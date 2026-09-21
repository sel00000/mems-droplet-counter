"""Bucket aggregation and output files (counts.csv, marks.csv, summary.json).

Depends only on the standard library and bubble_counter.rhythm.Mark; the
config object passed to build_summary is any dataclass (serialized with
dataclasses.asdict), so config.py is not imported at runtime.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from bubble_counter.rhythm import Mark

if TYPE_CHECKING:
    from bubble_counter.config import CounterConfig

_MARKS_HEADER = "index,first_frame,time_s,duration_frames,position_px,width_px,area_px"


def bucket_counts(marks: Sequence[Mark], fps: float, bucket_seconds: float,
                  total_frames: int) -> list[tuple[float, int]]:
    """Count marks per time bucket over the whole video span.

    Buckets cover [0, total_frames / fps]; each entry is (bucket_start_s,
    count).  A mark belongs to bucket int((first_frame / fps) /
    bucket_seconds).  Empty buckets are included, the trailing partial
    bucket too; at least 1 bucket whenever total_frames > 0.
    """
    if not bucket_seconds > 0:  # 음수/0/NaN 모두 거부
        raise ValueError(f"bucket_seconds는 0보다 커야 합니다: {bucket_seconds}")
    if total_frames <= 0:
        return []
    duration_s = total_frames / fps
    n_buckets = max(1, math.ceil(duration_s / bucket_seconds))
    counts = [0] * n_buckets
    for m in marks:
        idx = int((m.first_frame / fps) / bucket_seconds)
        counts[min(idx, n_buckets - 1)] += 1
    return [(i * bucket_seconds, c) for i, c in enumerate(counts)]


def write_counts_csv(path: Path, buckets: list[tuple[float, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("bucket_start_s,count\n")
        for start_s, count in buckets:
            f.write(f"{start_s:.3f},{count}\n")


def write_marks_csv(path: Path, marks: Sequence[Mark], fps: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_MARKS_HEADER + "\n")
        for index, m in enumerate(sorted(marks, key=lambda m: m.first_frame)):
            duration = m.last_frame - m.first_frame + 1
            f.write(f"{index},{m.first_frame},{m.first_frame / fps:.3f},"
                    f"{duration},{m.position_px:.3f},{m.width_px},{m.area_px}\n")


def build_summary(*, video_path: str, out_dir: str, cfg: "CounterConfig", total_count: int,
                  processed_frames: int, counted_frames: int, fps: float,
                  duration_s: float, elapsed_s: float, processing_fps: float) -> dict:
    """Assemble the JSON-safe run summary.

    mean_rate_per_s is total_count over the counted (post-warmup) duration.
    """
    counted_duration_s = counted_frames / fps
    return {
        "video_path": video_path,
        "out_dir": out_dir,
        "config": dataclasses.asdict(cfg),
        "total_count": total_count,
        "processed_frames": processed_frames,
        "counted_frames": counted_frames,
        "fps": fps,
        "duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "processing_fps": processing_fps,
        "mean_rate_per_s": total_count / max(counted_duration_s, 1e-9),
    }


def write_summary_json(path: Path, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
