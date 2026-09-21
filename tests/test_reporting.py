"""Tests for bubble_counter.reporting — bucket aggregation and output files."""

import json
from dataclasses import dataclass, field

import pytest

from bubble_counter.rhythm import Mark
from bubble_counter.reporting import (
    bucket_counts,
    build_summary,
    write_counts_csv,
    write_marks_csv,
    write_summary_json,
)


def mark(first_frame, last_frame=None, position_px=1.0, width_px=2, area_px=4):
    if last_frame is None:
        last_frame = first_frame
    return Mark(first_frame=first_frame, last_frame=last_frame,
                position_px=position_px, width_px=width_px, area_px=area_px)


# build_summary takes any dataclass via dataclasses.asdict (duck-typed), so the
# tests use a local stand-in instead of depending on config.py.
@dataclass
class _FakeRoi:
    x: float = 0.0
    w: float = 1.0


@dataclass
class _FakeConfig:
    roi: _FakeRoi = field(default_factory=_FakeRoi)
    band_px: int = 5
    fps_override: float | None = None


# ---------------------------------------------------------------------------
# bucket_counts
# ---------------------------------------------------------------------------

def test_bucket_counts_basic():
    marks = [mark(0), mark(29), mark(30), mark(89)]
    got = bucket_counts(marks, fps=30.0, bucket_seconds=1.0, total_frames=90)
    assert got == [(0.0, 2), (1.0, 1), (2.0, 1)]


def test_bucket_counts_empty_marks_gives_zero_buckets():
    got = bucket_counts([], fps=30.0, bucket_seconds=1.0, total_frames=90)
    assert got == [(0.0, 0), (1.0, 0), (2.0, 0)]


def test_bucket_counts_includes_partial_last_bucket():
    # 100 frames @ 30 fps = 3.33 s -> buckets 0..3, last one partial.
    got = bucket_counts([mark(99)], fps=30.0, bucket_seconds=1.0, total_frames=100)
    assert got == [(0.0, 0), (1.0, 0), (2.0, 0), (3.0, 1)]


def test_bucket_counts_at_least_one_bucket_for_short_video():
    got = bucket_counts([mark(0)], fps=30.0, bucket_seconds=1.0, total_frames=1)
    assert got == [(0.0, 1)]


def test_bucket_counts_no_frames_gives_no_buckets():
    assert bucket_counts([], fps=30.0, bucket_seconds=1.0, total_frames=0) == []


def test_bucket_counts_non_unit_bucket_seconds():
    marks = [mark(0), mark(74), mark(75)]
    got = bucket_counts(marks, fps=30.0, bucket_seconds=2.5, total_frames=90)
    assert got == [(0.0, 2), (2.5, 1)]


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def test_write_counts_csv(tmp_path):
    path = tmp_path / "counts.csv"
    write_counts_csv(path, [(0.0, 2), (1.0, 0), (2.5, 1)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "bucket_start_s,count"
    assert lines[1:] == ["0.000,2", "1.000,0", "2.500,1"]


def test_write_marks_csv_sorted_by_first_frame(tmp_path):
    path = tmp_path / "marks.csv"
    marks = [  # deliberately out of order
        mark(30, last_frame=32, position_px=4.5, width_px=2, area_px=6),
        mark(0, last_frame=0, position_px=1.0, width_px=1, area_px=1),
    ]
    write_marks_csv(path, marks, fps=30.0)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "index,first_frame,time_s,duration_frames,position_px,width_px,area_px"
    assert lines[1] == "0,0,0.000,1,1.000,1,1"
    assert lines[2] == "1,30,1.000,3,4.500,2,6"
    assert len(lines) == 3


def test_write_marks_csv_empty(tmp_path):
    path = tmp_path / "marks.csv"
    write_marks_csv(path, [], fps=30.0)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["index,first_frame,time_s,duration_frames,position_px,width_px,area_px"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_build_summary_is_json_safe_and_complete():
    summary = build_summary(
        video_path="in/기포.mp4", out_dir="out", cfg=_FakeConfig(),
        total_count=10, processed_frames=360, counted_frames=300,
        fps=30.0, duration_s=12.0, elapsed_s=1.5, processing_fps=240.0)
    text = json.dumps(summary)  # must not raise
    assert isinstance(text, str)
    assert summary["total_count"] == 10
    assert summary["video_path"] == "in/기포.mp4"
    assert summary["processed_frames"] == 360
    assert summary["counted_frames"] == 300
    assert summary["fps"] == 30.0
    assert summary["duration_s"] == 12.0
    assert summary["elapsed_s"] == 1.5
    assert summary["processing_fps"] == 240.0
    # counted duration = 300 / 30 = 10 s -> 10 bubbles / 10 s = 1.0 /s
    assert summary["mean_rate_per_s"] == pytest.approx(1.0)
    # cfg embedded via dataclasses.asdict, recursively
    assert summary["config"]["band_px"] == 5
    assert summary["config"]["roi"] == {"x": 0.0, "w": 1.0}
    assert summary["config"]["fps_override"] is None


def test_build_summary_zero_counted_frames_no_division_error():
    summary = build_summary(
        video_path="v", out_dir="o", cfg=_FakeConfig(),
        total_count=0, processed_frames=0, counted_frames=0,
        fps=30.0, duration_s=0.0, elapsed_s=0.1, processing_fps=0.0)
    assert summary["mean_rate_per_s"] == 0.0


def test_write_summary_json_korean_path_and_no_ascii_escape(tmp_path):
    out_dir = tmp_path / "결과"
    out_dir.mkdir()
    path = out_dir / "summary.json"
    write_summary_json(path, {"video_path": "기포.mp4", "total_count": 3})
    raw = path.read_text(encoding="utf-8")
    assert "기포.mp4" in raw          # ensure_ascii=False
    assert "\\u" not in raw
    assert "\n  " in raw              # indent=2
    assert json.loads(raw) == {"video_path": "기포.mp4", "total_count": 3}


def test_bucket_counts_nonpositive_bucket_seconds_raises():
    with pytest.raises(ValueError, match="bucket_seconds"):
        bucket_counts([], fps=30.0, bucket_seconds=0.0, total_frames=100)


def test_bucket_counts_negative_bucket_seconds_raises():
    with pytest.raises(ValueError, match="bucket_seconds"):
        bucket_counts([], fps=30.0, bucket_seconds=-1.0, total_frames=100)
