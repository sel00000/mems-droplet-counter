"""End-to-end counting pipeline (the GUI/CLI entry point).

Decode every frame (never skipped) -> MOG2 foreground -> collapse the band
rows into one boolean line -> streaming RhythmCounter.  The fast path runs
MOG2 only on the band+padding strip; the annotate path runs MOG2 on the
full processing frame but counts from the identical band rows (MOG2 is a
per-pixel GMM, so both yield the same band foreground).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

from bubble_counter import reporting
from bubble_counter.annotate import AnnotatedWriter, redraw_rhythm_marks, save_rhythm_strip
from bubble_counter.geometry import extract_band_row, line_center_px, strip_bounds
from bubble_counter.live import LiveEmitter
from bubble_counter.resmon import ResourceSampler, make_reader
from bubble_counter.rhythm import Mark, RhythmCounter
from bubble_counter.video_io import VideoSource, resolve_fps

if TYPE_CHECKING:
    import threading

    from bubble_counter.config import CounterConfig

_RHYTHM_CHUNK_ROWS = 2000


@dataclass
class ProgressEvent:
    """A periodic progress snapshot handed to ``progress_cb``."""

    frame_idx: int
    total_frames: int  # <=0 if unknown
    processing_fps: float
    count: int
    eta_s: float | None
    # 자원 사용률(실행 중). 측정 비활성/실패 시 None → 완전 하위호환(trailing default).
    cpu_pct: float | None = None
    cpu_cores: float | None = None
    rss_mb: float | None = None
    sys_ram_pct: float | None = None
    gpu_pct: float | None = None


@dataclass
class CountResult:
    """The full result of a counting run (also mirrored to disk)."""

    total: int
    marks: list[Mark]
    buckets: list[tuple[float, int]]
    summary: dict
    out_dir: Path


def run_count(
    video_path: "str | Path",
    out_dir: "str | Path",
    cfg: "CounterConfig",
    progress_cb: "Callable[[ProgressEvent], None] | None" = None,
    progress_every: int = 200,
    resmon_interval: float = 0.5,
    cancel_event: "threading.Event | None" = None,
    live_cb: "Callable | None" = None,
    live_control=None,
    live_label: str = "",
) -> CountResult:
    """Run the counting pipeline and write the output files.

    Writes summary.json / counts.csv / marks.csv (plus annotated.avi and
    rhythm_*.png when enabled) into ``out_dir`` (created if missing).  If
    ``cancel_event`` is set the run stops at the next progress checkpoint,
    still writes the partial outputs, and marks summary["cancelled"]=True.

    ``resmon_interval`` (seconds) drives inline resource sampling: a progress
    checkpoint fires every ``progress_every`` frames OR every
    ``resmon_interval`` seconds (whichever comes first), so slow hardware still
    updates in real time.  ``<= 0`` disables both the monitor and the
    time-based checkpoint (bench uses 0 for pure speed measurement); the
    sampled CPU/RAM/GPU stats are attached to each ProgressEvent and recorded
    under summary["resources"].  Measurement failures never stop the count.
    """
    out_dir = Path(out_dir)
    progress_every = max(1, progress_every)

    source = VideoSource(video_path, cfg.roi, cfg.scale)
    writer: AnnotatedWriter | None = None
    sampler: ResourceSampler | None = None
    try:
        fps = resolve_fps(source.info, cfg.fps_override)
        out_w, out_h = source.out_size

        if cfg.line.axis == "h":
            roi_len, line_len = out_h, out_w
        else:
            roi_len, line_len = out_w, out_h

        center = line_center_px(roi_len, cfg.line.pos)
        pad = max(cfg.morph_kernel + 2, 3)
        strip_lo, strip_hi, band_lo, band_hi = strip_bounds(
            roi_len, center, cfg.line.band_px, pad
        )
        abs_band_lo = strip_lo + band_lo
        abs_band_hi = strip_lo + band_hi

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=False
        )
        counter = RhythmCounter(
            line_len, min_width_px=cfg.min_width_px, min_area_px=cfg.min_area_px,
            merge_gap_frames=cfg.merge_gap_frames,
        )

        morph_kernel = None
        if cfg.morph_kernel > 0:
            morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
            )
        row_close_kernel = None
        if cfg.row_close_px > 1:
            row_close_kernel = np.ones((1, cfg.row_close_px), np.uint8)

        annotate = cfg.annotate
        save_rhythm = cfg.save_rhythm
        if annotate or save_rhythm:
            out_dir.mkdir(parents=True, exist_ok=True)
        if annotate:
            writer = AnnotatedWriter(out_dir / "annotated.avi", (out_w, out_h), fps)
        line_overlay = (cfg.line.axis, center, abs_band_lo, abs_band_hi)

        warmup = cfg.warmup_frames
        max_frames = cfg.max_frames
        all_marks: list[Mark] = []
        rhythm_rows: list[np.ndarray] = []
        rhythm_seg_start = warmup
        rhythm_index = 0
        rhythm_chunks: list[tuple[Path, int]] = []  # (png path, seg start) — 종료 시 재드로잉 대상

        total_meta = source.info.frame_count
        if max_frames is not None:
            total_hint = min(total_meta, max_frames) if total_meta > 0 else max_frames
        else:
            total_hint = total_meta

        processed = 0
        cancelled = False
        sampler = ResourceSampler(make_reader(), enabled=resmon_interval > 0)
        live = LiveEmitter(live_cb, live_control)
        start = time.perf_counter()
        last_cp = start

        for frame_idx, gray, bgr in source.frames(with_color=annotate):
            if max_frames is not None and frame_idx >= max_frames:
                break

            now_cp = time.perf_counter()
            due = (processed % progress_every == 0) or (
                resmon_interval > 0 and now_cp - last_cp >= resmon_interval)
            if due:
                last_cp = now_cp
                snap = sampler.sample()
                if progress_cb is not None:
                    elapsed = now_cp - start
                    pfps = processed / elapsed if elapsed > 0 else 0.0
                    eta = (max(0.0, (total_hint - processed) / pfps)
                           if total_hint > 0 and pfps > 0 else None)
                    progress_cb(ProgressEvent(
                        frame_idx, total_hint, pfps, counter.total_count, eta,
                        cpu_pct=snap.cpu_pct if snap else None,
                        cpu_cores=snap.cpu_cores if snap else None,
                        rss_mb=snap.rss_mb if snap else None,
                        sys_ram_pct=snap.sys_ram_pct if snap else None,
                        gpu_pct=snap.gpu_pct if snap else None))
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break

            if annotate:
                fg = mog2.apply(gray)
                if morph_kernel is not None:
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
                if cfg.line.axis == "h":
                    row = extract_band_row(fg, abs_band_lo, abs_band_hi)
                else:
                    row = fg[:, abs_band_lo:abs_band_hi].any(axis=1)
                contours, _ = cv2.findContours(
                    fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(c) for c in contours]
            else:
                if cfg.line.axis == "h":
                    strip = gray[strip_lo:strip_hi, :]
                else:
                    strip = np.ascontiguousarray(gray[:, strip_lo:strip_hi].T)
                fg = mog2.apply(strip)
                if morph_kernel is not None:
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel)
                row = extract_band_row(fg, band_lo, band_hi)

            if row_close_kernel is not None:
                row = cv2.morphologyEx(
                    row.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, row_close_kernel
                )[0] > 0

            if frame_idx >= warmup:
                new_marks = counter.feed(row, frame_idx)
                if new_marks:
                    all_marks.extend(new_marks)
                if save_rhythm:
                    rhythm_rows.append(np.where(row, 255, 0).astype(np.uint8))
                    if len(rhythm_rows) >= _RHYTHM_CHUNK_ROWS:
                        rhythm_index, chunk_path = _flush_rhythm(
                            out_dir, rhythm_rows, rhythm_seg_start, all_marks, rhythm_index)
                        rhythm_chunks.append((chunk_path, rhythm_seg_start))
                        rhythm_seg_start += len(rhythm_rows)
                        rhythm_rows = []

            if writer is not None:
                writer.write(bgr, boxes, line_overlay, counter.total_count, frame_idx)

            processed += 1
            if live.enabled:
                elapsed_live = time.perf_counter() - start
                pfps_live = processed / elapsed_live if elapsed_live > 0 else 0.0
                live.pace()
                live.emit(
                    frame_idx, total_hint, gray,
                    {"total": counter.total_count},
                    live_label or str(video_path), pfps_live)

        all_marks.extend(counter.flush())
        if save_rhythm and rhythm_rows:
            _flush_rhythm(out_dir, rhythm_rows, rhythm_seg_start, all_marks, rhythm_index)
        for chunk_path, chunk_start in rhythm_chunks:
            redraw_rhythm_marks(chunk_path, chunk_start, all_marks)

        resources = sampler.result()
        resources["interval_s"] = resmon_interval  # 스펙 §6.3: 샘플링 주기 기록
        elapsed = time.perf_counter() - start
        processing_fps = processed / elapsed if elapsed > 0 else 0.0
        counted_frames = max(0, processed - warmup)
        duration_s = processed / fps if fps > 0 else 0.0

        out_dir.mkdir(parents=True, exist_ok=True)
        buckets = reporting.bucket_counts(all_marks, fps, cfg.bucket_seconds, processed)
        reporting.write_counts_csv(out_dir / "counts.csv", buckets)
        reporting.write_marks_csv(out_dir / "marks.csv", all_marks, fps)
        summary = reporting.build_summary(
            video_path=str(video_path), out_dir=str(out_dir), cfg=cfg,
            total_count=counter.total_count, processed_frames=processed,
            counted_frames=counted_frames, fps=fps, duration_s=duration_s,
            elapsed_s=elapsed, processing_fps=processing_fps,
        )
        summary["cancelled"] = cancelled
        summary["resources"] = resources
        reporting.write_summary_json(out_dir / "summary.json", summary)

        return CountResult(
            total=counter.total_count, marks=all_marks, buckets=buckets,
            summary=summary, out_dir=out_dir,
        )
    finally:
        if sampler is not None:
            sampler.close()
        if writer is not None:
            writer.close()
        source.close()


def _flush_rhythm(out_dir: Path, rows: list[np.ndarray], seg_start: int,
                  marks: list[Mark], index: int) -> tuple[int, Path]:
    strip = np.array(rows, dtype=np.uint8)
    path = out_dir / f"rhythm_{index:04d}.png"
    save_rhythm_strip(path, strip, seg_start, marks)
    return index + 1, path
