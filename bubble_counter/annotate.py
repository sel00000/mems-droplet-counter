"""Verification outputs: annotated MJPG video and rhythm-strip PNGs.

Internal to the pipeline (only ``bubble_counter.pipeline`` imports this).
None of this affects the count; it renders the ``--annotate`` overlay video
and the ``--save-rhythm`` space-time strips used to eyeball the marks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from bubble_counter.imgio import imread_unicode, imwrite_unicode

if TYPE_CHECKING:
    from bubble_counter.rhythm import Mark

# BGR colors.
_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_CYAN = (255, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


class AnnotatedWriter:
    """MJPG writer overlaying blob boxes, the counting line/band, and text."""

    def __init__(self, path: str | Path, size_wh: tuple[int, int], fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size_wh, isColor=True)
        if not self._writer.isOpened():
            self._writer.release()
            raise IOError(f"주석 영상을 생성할 수 없습니다: {path}")

    def write(self, bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
              line_overlay: tuple[str, int, int, int], count: int,
              frame_idx: int) -> None:
        """Draw one annotated frame and append it to the video.

        boxes: (x, y, w, h) blob rectangles.  line_overlay: (axis, center,
        band_lo, band_hi) in processing-scale pixels along the axis
        perpendicular to the line.
        """
        frame = bgr.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), _GREEN, 1)
        _draw_line_overlay(frame, line_overlay)
        text = f"count={count}  frame={frame_idx}"
        cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _WHITE, 1, cv2.LINE_AA)
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


def _draw_line_overlay(frame: np.ndarray,
                       line_overlay: tuple[str, int, int, int]) -> None:
    axis, center, band_lo, band_hi = line_overlay
    h, w = frame.shape[:2]
    if axis == "h":
        cv2.line(frame, (0, center), (w - 1, center), _YELLOW, 1)
        cv2.line(frame, (0, band_lo), (w - 1, band_lo), _CYAN, 1)
        cv2.line(frame, (0, band_hi - 1), (w - 1, band_hi - 1), _CYAN, 1)
    else:
        cv2.line(frame, (center, 0), (center, h - 1), _YELLOW, 1)
        cv2.line(frame, (band_lo, 0), (band_lo, h - 1), _CYAN, 1)
        cv2.line(frame, (band_hi - 1, 0), (band_hi - 1, h - 1), _CYAN, 1)


def save_rhythm_strip(path: str | Path, rows_u8_2d: np.ndarray, start_frame: int,
                      marks: Sequence["Mark"]) -> None:
    """Write a (time x line) rhythm strip PNG with red mark boxes.

    rows_u8_2d: (n_rows, line_len) uint8 in {0, 255}; strip row r is frame
    ``start_frame + r``.  Each mark overlapping [start_frame, start_frame +
    n_rows) gets an approximate red rectangle (x spans position_px +-
    width_px/2, y spans first_frame..last_frame clipped to the strip).
    """
    if rows_u8_2d.ndim != 2:
        raise ValueError(f"rows_u8_2d must be 2-D, got shape {rows_u8_2d.shape}")
    bgr = cv2.cvtColor(rows_u8_2d, cv2.COLOR_GRAY2BGR)
    _draw_mark_boxes(bgr, start_frame, marks)
    imwrite_unicode(path, bgr)


def _draw_mark_boxes(bgr: np.ndarray, start_frame: int,
                     marks: Sequence["Mark"]) -> None:
    """Draw the red mark rectangles in place. Deterministic per (mark,
    start_frame, shape), so repeating a draw is pixel-identical — the
    redraw path below relies on this."""
    n_rows, line_len = bgr.shape[:2]
    end_frame = start_frame + n_rows
    for m in marks:
        if m.last_frame < start_frame or m.first_frame >= end_frame:
            continue
        y0 = max(0, m.first_frame - start_frame)
        y1 = min(n_rows - 1, m.last_frame - start_frame)
        half = m.width_px / 2.0
        x0 = max(0, int(round(m.position_px - half)))
        x1 = min(line_len - 1, int(round(m.position_px + half)))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), _RED, 1)


def redraw_rhythm_marks(path: str | Path, start_frame: int,
                        marks: Sequence["Mark"]) -> None:
    """Re-draw mark boxes on an existing rhythm strip PNG.

    Used after the counter has flushed: chunks written mid-run only had the
    marks finalized by flush time, so marks still open at a chunk boundary
    were missing.  PNG round-trips losslessly and box drawing is
    deterministic, so repainting with the final list adds the missing boxes
    and repaints existing ones identically.
    """
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"rhythm 이미지를 다시 열 수 없습니다: {path}")
    _draw_mark_boxes(bgr, start_frame, marks)
    imwrite_unicode(path, bgr)
