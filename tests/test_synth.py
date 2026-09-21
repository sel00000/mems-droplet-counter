"""Tests for bubble_counter.synth (synthetic bubble video + GT generator)."""

from __future__ import annotations

import json

import cv2
import numpy as np

from bubble_counter.synth import SynthConfig, crossing_frame, generate


# --- crossing_frame: pure analytic helper -----------------------------------


def test_crossing_frame_hand_computed_example_from_brief():
    # ceil((485-240)/49) = ceil(5.0) = 5 -> f0 + 5
    assert crossing_frame(f0=0, y0=485.0, speed=49.0, line_y=240.0) == 5


def test_crossing_frame_offsets_by_f0():
    assert crossing_frame(f0=100, y0=485.0, speed=49.0, line_y=240.0) == 105


def test_crossing_frame_rounds_up_when_not_exact():
    # delta=285, 285/50=5.7 -> ceil=6
    assert crossing_frame(f0=0, y0=485.0, speed=50.0, line_y=200.0) == 6


def test_crossing_frame_immediate_when_already_at_or_past_line():
    assert crossing_frame(f0=10, y0=240.0, speed=49.0, line_y=240.0) == 10
    assert crossing_frame(f0=10, y0=200.0, speed=49.0, line_y=240.0) == 10


def test_crossing_frame_none_when_speed_zero_and_above_line():
    assert crossing_frame(f0=0, y0=485.0, speed=0.0, line_y=240.0) is None


def test_crossing_frame_none_when_speed_negative_and_above_line():
    assert crossing_frame(f0=0, y0=485.0, speed=-5.0, line_y=240.0) is None


# --- generate(): shared small config for fast tests -------------------------
# Small frame size + short duration keeps tests fast while leaving
# brightness/alpha/speed/radius/drift/noise at their spec defaults, since
# those are the fields the rendering assertions actually care about.


def _small_cfg(**overrides) -> SynthConfig:
    base = dict(
        width=200, height=120, fps=30.0, duration_s=4.0,
        lead_in_s=1.0, rate_per_s=5.0, seed=7,
    )
    base.update(overrides)
    return SynthConfig(**base)


# --- generate(): determinism --------------------------------------------------


def test_generate_is_deterministic_for_same_seed(tmp_path):
    cfg = _small_cfg()
    video1 = tmp_path / "a.avi"
    gt1_path = tmp_path / "a.gt.json"
    video2 = tmp_path / "b.avi"
    gt2_path = tmp_path / "b.gt.json"

    gt1 = generate(cfg, video1, gt1_path)
    gt2 = generate(cfg, video2, gt2_path)

    assert gt1 == gt2
    assert gt1["crossings"] == gt2["crossings"]

    assert video1.exists() and video1.stat().st_size > 0
    assert video2.exists() and video2.stat().st_size > 0
    assert gt1_path.exists() and gt1_path.stat().st_size > 0
    assert gt2_path.exists() and gt2_path.stat().st_size > 0

    assert json.loads(gt1_path.read_text(encoding="utf-8")) == gt1


# --- generate(): GT consistency -----------------------------------------------


def test_gt_crossings_are_within_lead_in_and_n_frames_range(tmp_path):
    cfg = _small_cfg()
    gt = generate(cfg, tmp_path / "v.avi")

    lead_in_frames = round(cfg.lead_in_s * cfg.fps)
    assert gt["n_frames"] == round(cfg.duration_s * cfg.fps)
    assert len(gt["crossings"]) > 0  # sanity: this seed/config produces crossings
    for c in gt["crossings"]:
        assert lead_in_frames <= c["frame"] < gt["n_frames"]

    assert gt["total_crossings"] == len(gt["crossings"])


def test_gt_generated_with_large_speed_max_does_not_error(tmp_path):
    cfg = _small_cfg(speed_min=4000.0, speed_max=5000.0)
    gt = generate(cfg, tmp_path / "v.avi")

    assert gt["total_crossings"] == len(gt["crossings"])
    lead_in_frames = round(cfg.lead_in_s * cfg.fps)
    for c in gt["crossings"]:
        assert lead_in_frames <= c["frame"] < gt["n_frames"]


# --- generate(): video file is a valid, exact-length video --------------------


def test_generated_video_opens_with_correct_frame_count_and_size(tmp_path):
    cfg = _small_cfg()
    video_path = tmp_path / "v.avi"
    gt = generate(cfg, video_path)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    actual_frames = 0
    last_shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        actual_frames += 1
        last_shape = frame.shape
    cap.release()

    assert actual_frames == gt["n_frames"]
    assert last_shape == (cfg.height, cfg.width, 3)


# --- generate(): bubbles are actually rendered near the line ------------------


def test_bubble_is_visibly_rendered_near_line_at_crossing_frame(tmp_path):
    # Slow, narrow speed range: a fast bubble can legitimately overshoot
    # line_y by many pixels between two discrete frames (that's the
    # high-speed-bubble scenario the design targets), so it would not
    # necessarily be spatially "near" line_y at its crossing frame. A slow
    # bubble lands within ~1px of line_y at the crossing frame, which is
    # what this indirect rendering check needs.
    cfg = _small_cfg(speed_min=3.0, speed_max=6.0)
    video_path = tmp_path / "v.avi"
    gt = generate(cfg, video_path)
    assert len(gt["crossings"]) > 0

    crossing = gt["crossings"][0]
    line_y = gt["line_y"]
    band_lo, band_hi = max(0, line_y - 3), min(cfg.height, line_y + 4)

    cap = cv2.VideoCapture(str(video_path))

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        assert ok
        return frame

    lead_in_frame = read_frame(0)  # guaranteed bubble-free (before any f0)
    crossing_frame_img = read_frame(crossing["frame"])
    cap.release()

    baseline_max = lead_in_frame[band_lo:band_hi, :, 0].astype(np.float32).max()
    crossing_max = crossing_frame_img[band_lo:band_hi, :, 0].astype(np.float32).max()

    assert crossing_max > baseline_max + cfg.brightness * cfg.alpha * 0.3


# --- generate(): brightness saturation (clip) regression tests ---------------


def _read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        assert ret
        return frame
    finally:
        cap.release()


def _saturation_cfg(brightness):
    return SynthConfig(
        width=64, height=64, fps=10.0, duration_s=6.0, rate_per_s=1.5,
        speed_min=5.0, speed_max=8.0, radius_min=5, radius_max=6,
        brightness=brightness, noise_sigma=0.0, lead_in_s=1.0, seed=3,
    )


def test_extreme_positive_brightness_saturates_white_not_wraparound(tmp_path):
    """clip이 없으면 bg~90 + 500*0.85 ≈ 515 → uint8 랩어라운드로 어두운 값이
    된다. clip이 있으면 기포 중심은 255로 포화된다 (MJPG 손실 여유 ≥250)."""
    cfg = _saturation_cfg(brightness=500.0)
    video = tmp_path / "sat_pos.avi"
    gt = generate(cfg, video)

    # x가 가장자리인 crossing은 중심 픽셀 판독이 프레임 밖일 수 있어 제외
    c = next(c for c in gt["crossings"] if 6 <= c["x"] <= cfg.width - 7)
    frame = _read_frame(video, c["frame"])
    px = int(frame[gt["line_y"], int(round(c["x"])), 0])
    assert px >= 250, f"기포 중심이 포화 백색이 아니다 (랩어라운드 의심): {px}"


def test_extreme_negative_brightness_saturates_black_not_wraparound(tmp_path):
    cfg = _saturation_cfg(brightness=-500.0)
    video = tmp_path / "sat_neg.avi"
    gt = generate(cfg, video)

    c = next(c for c in gt["crossings"] if 6 <= c["x"] <= cfg.width - 7)
    frame = _read_frame(video, c["frame"])
    px = int(frame[gt["line_y"], int(round(c["x"])), 0])
    assert px <= 5, f"기포 중심이 포화 흑색이 아니다 (랩어라운드 의심): {px}"
