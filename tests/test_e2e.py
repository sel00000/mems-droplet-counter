"""End-to-end pipeline tests against synthetic videos with known ground truth.

The synthetic videos are clean, so a correct pipeline converges to the
analytic crossing count.  When accuracy drifts the fix is to diagnose the
pipeline (rhythm / geometry / MOG2 warmup / band formula), never to loosen
these thresholds.
"""

from __future__ import annotations

import csv
import json
import threading

import cv2
import numpy as np
import pytest

from bubble_counter.config import CounterConfig, LineSpec
from bubble_counter.pipeline import CountResult, ProgressEvent, run_count
from bubble_counter.synth import SynthConfig, generate


def _accuracy_config(**overrides) -> CounterConfig:
    """The shared counting config for the accuracy E2E cases.

    band_px=64 satisfies the design's band-intersection rule for the synth
    defaults (band_px >= speed_max - 2*radius_min = 60 - 10 = 50), so no
    bubble is physically unobservable; warmup aligns with the 2s lead-in.
    """
    base = dict(line=LineSpec(pos=0.5, band_px=64), warmup_frames=60)
    base.update(overrides)
    return CounterConfig(**base)


@pytest.fixture(scope="session")
def mixed_video(tmp_path_factory):
    """Mixed-speed synth video (slow..fast bubbles), generated once."""
    out_dir = tmp_path_factory.mktemp("e2e_mixed")
    video = out_dir / "mixed.avi"
    cfg = SynthConfig(
        width=480, height=360, fps=30, duration_s=15, rate_per_s=2.5,
        speed_min=2, speed_max=60, radius_min=5, radius_max=10,
        lead_in_s=2, seed=1234,
    )
    gt = generate(cfg, video)
    return video, gt


@pytest.fixture(scope="session")
def fast_video(tmp_path_factory):
    """Fast-only synth video: every bubble crosses in 1-2 frames."""
    out_dir = tmp_path_factory.mktemp("e2e_fast")
    video = out_dir / "fast.avi"
    cfg = SynthConfig(
        width=480, height=360, fps=30, duration_s=12, rate_per_s=2.5,
        speed_min=40, speed_max=60, radius_min=5, radius_max=10,
        lead_in_s=2, seed=77,
    )
    gt = generate(cfg, video)
    return video, gt


def _assert_within_tolerance(total: int, target: int) -> None:
    tol = max(2, 0.05 * target)
    assert abs(total - target) <= tol, (
        f"count {total} vs ground truth {target} exceeds tolerance {tol}"
    )


def test_count_accuracy(mixed_video, tmp_path):
    video, gt = mixed_video
    result = run_count(video, tmp_path / "out", _accuracy_config())

    assert isinstance(result, CountResult)
    _assert_within_tolerance(result.total, gt["total_crossings"])


def test_fast_only_bubbles(fast_video, tmp_path):
    video, gt = fast_video
    result = run_count(video, tmp_path / "out", _accuracy_config())

    # Sanity: this really is a fast-only regime (every crossing is short-lived).
    assert gt["total_crossings"] > 0
    _assert_within_tolerance(result.total, gt["total_crossings"])


def test_annotate_matches_fast(mixed_video, tmp_path):
    video, _gt = mixed_video

    fast_result = run_count(video, tmp_path / "fast", _accuracy_config())

    ann_out = tmp_path / "ann"
    ann_result = run_count(
        video, ann_out, _accuracy_config(annotate=True, save_rhythm=True)
    )

    # fast vs annotate count the identical band rows -> identical totals.
    assert ann_result.total == fast_result.total

    annotated = ann_out / "annotated.avi"
    assert annotated.exists()
    cap = cv2.VideoCapture(str(annotated))
    n_frames = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        n_frames += 1
    cap.release()
    assert n_frames > 0

    rhythm_pngs = list(ann_out.glob("rhythm_*.png"))
    assert len(rhythm_pngs) >= 1


def test_outputs(mixed_video, tmp_path):
    video, _gt = mixed_video
    out = tmp_path / "out"
    result = run_count(video, out, _accuracy_config())

    # counts.csv: bucket counts sum to the total.
    with open(out / "counts.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sum(int(r["count"]) for r in rows) == result.total

    # marks.csv: one data row per counted mark.
    with open(out / "marks.csv", newline="", encoding="utf-8") as f:
        mark_rows = list(csv.DictReader(f))
    assert len(mark_rows) == result.total

    # summary.json: parses and agrees on the total.
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_count"] == result.total

    # max_frames bounds the processed count exactly.
    capped = run_count(
        video, tmp_path / "capped", _accuracy_config(max_frames=90)
    )
    assert capped.summary["processed_frames"] == 90


def test_cancel(mixed_video, tmp_path):
    video, _gt = mixed_video
    cancel_event = threading.Event()
    events: list[ProgressEvent] = []

    def progress_cb(event: ProgressEvent) -> None:
        events.append(event)
        if len(events) == 5:  # cancel mid-run (5th callback), not at frame 0
            cancel_event.set()

    out = tmp_path / "out"
    result = run_count(
        video, out, _accuracy_config(),
        progress_cb=progress_cb, progress_every=50, cancel_event=cancel_event,
    )

    assert result.summary["cancelled"] is True
    assert (out / "summary.json").exists()

    # Partial outputs are preserved and non-empty: enough crossings have
    # happened by the 5th callback (frame 200) that real marks were counted.
    assert result.total >= 1

    # marks.csv: exactly one data row per counted (partial) mark.
    with open(out / "marks.csv", newline="", encoding="utf-8") as f:
        mark_rows = list(csv.DictReader(f))
    assert len(mark_rows) == result.total

    # counts.csv: bucket counts sum to the partial total.
    with open(out / "counts.csv", newline="", encoding="utf-8") as f:
        count_rows = list(csv.DictReader(f))
    assert sum(int(r["count"]) for r in count_rows) == result.total


def _reencode_transposed(src, dst) -> bool:
    """Re-encode ``src`` with every frame transposed (H<->W) into ``dst``.

    Prefers the lossless FFV1 codec; if the writer will not open, falls back
    to MJPG.  Returns True when the lossless codec was used.
    """
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    lossless = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            transposed = frame.transpose(1, 0, 2)
            if writer is None:
                new_h, new_w = transposed.shape[:2]
                writer = cv2.VideoWriter(
                    str(dst), cv2.VideoWriter_fourcc(*"FFV1"), fps,
                    (new_w, new_h), isColor=True,
                )
                if not writer.isOpened():
                    lossless = False
                    writer = cv2.VideoWriter(
                        str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                        (new_w, new_h), isColor=True,
                    )
                assert writer.isOpened(), "no VideoWriter opened for transposed video"
            writer.write(transposed)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    assert writer is not None, "source video decoded no frames"
    return lossless


def test_v_axis_matches_h_axis_on_transposed_video(mixed_video, tmp_path):
    """The axis='v' pipeline branch must equal the axis='h' branch.

    A horizontal line over the original video is the same physical line as a
    vertical line over the video transposed (H<->W), so the two runs must count
    the same bubbles.  FFV1 re-encoding is lossless (exact equality); an MJPG
    fallback is lossy, so allow a +-1 difference there.
    """
    video, _gt = mixed_video

    h_result = run_count(video, tmp_path / "h", _accuracy_config())

    transposed = tmp_path / "mixed_transposed.avi"
    lossless = _reencode_transposed(video, transposed)

    v_line = LineSpec(axis="v", pos=0.5, band_px=64)
    v_fast = run_count(transposed, tmp_path / "v_fast", _accuracy_config(line=v_line))

    if lossless:
        assert v_fast.total == h_result.total
    else:
        assert abs(v_fast.total - h_result.total) <= 1

    # The annotate path decodes the identical transposed file, so its v-axis
    # total must equal the fast v-axis total exactly.
    v_ann = run_count(
        transposed, tmp_path / "v_ann",
        _accuracy_config(line=v_line, annotate=True),
    )
    assert v_ann.total == v_fast.total


def _write_boundary_blob_video(path, n_frames=130, size=64):
    """정지 배경(30) + 프레임 60부터 (x=32, y=50-(f-60))에서 상승하는 밝은 원.
    밴드 행 22..42(중앙 32, band 21) 기준 원(반지름 4)의 체류는 약 프레임 64..92."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (size, size), isColor=True)
    assert writer.isOpened()
    try:
        for f in range(n_frames):
            frame = np.full((size, size), 30, np.uint8)
            if f >= 60:
                y = 50 - (f - 60)
                if -4 <= y <= size + 4:
                    cv2.circle(frame, (32, y), 4, 255, -1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def test_rhythm_chunk_overlay_includes_boundary_spanning_mark(tmp_path, monkeypatch):
    """청크 flush 후에 finalize되는 마크도 그 청크 PNG에 박스로 나타나야 한다."""
    monkeypatch.setattr("bubble_counter.pipeline._RHYTHM_CHUNK_ROWS", 20)
    video = tmp_path / "blob.avi"
    _write_boundary_blob_video(video)

    out = tmp_path / "out"
    cfg = CounterConfig(
        line=LineSpec(pos=0.5, band_px=21), warmup_frames=60, save_rhythm=True,
    )
    result = run_count(video, out, cfg)

    assert len(result.marks) == 1
    mark = result.marks[0]
    # 청크 0은 프레임 [60, 80) — 마크가 그 경계를 실제로 걸치는지 확인
    assert mark.first_frame < 80 <= mark.last_frame

    img = cv2.imread(str(out / "rhythm_0000.png"))
    assert img is not None
    assert img.shape[0] == 20
    red = (img[:, :, 2] == 255) & (img[:, :, 1] == 0) & (img[:, :, 0] == 0)
    assert red.any(), "청크 0 PNG에 경계 걸침 마크의 빨간 박스가 없다"
