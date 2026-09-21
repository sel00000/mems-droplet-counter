"""측정 중 라이브 뷰 — 관찰 전용 스냅샷/제어 (계수 결과 무영향 L4).

표시 MVP: 전체 영상 미리보기 + 카운트 메타. 마스크/패치/수동 플래그는 범위 밖.
일시정지(display_paused)는 GUI만 사용 — 엔진은 읽지 않는다(D3).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# UI 스로틀 기본값 (초당 스냅샷 상한)
DEFAULT_MAX_UI_FPS = 15.0
# 미리보기 긴 변 상한 (px)
DEFAULT_MAX_PREVIEW_EDGE = 640
# 저속 관찰 시 wall-clock 기준 목표 처리 fps = BASE_PACE_FPS * speed
BASE_PACE_FPS = 30.0


@dataclass(frozen=True)
class LiveSnapshot:
    """워커 → GUI 전달용 한 프레임 관찰 스냅샷."""

    frame_idx: int
    total_frames: int
    jpeg: bytes
    counts: dict          # multiway: {1: n, ...} / 단일선: {"total": n}
    label: str
    processing_fps: float
    preview_w: int
    preview_h: int
    # 원본 프레임 크기(오버레이 분율 좌표 매핑용). 미지이면 preview와 동일.
    source_w: int = 0
    source_h: int = 0


class LiveControl:
    """스레드 안전 관찰 제어. display_paused는 엔진이 무시한다(D3)."""

    def __init__(self, speed: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._speed = 1.0
        self._display_paused = False
        self.set_speed(speed)

    def get_speed(self) -> float:
        with self._lock:
            return self._speed

    def set_speed(self, speed: float) -> None:
        s = float(speed)
        if not (s > 0):
            raise ValueError(f"speed는 0보다 커야 합니다: {speed}")
        with self._lock:
            self._speed = s

    def is_display_paused(self) -> bool:
        with self._lock:
            return self._display_paused

    def set_display_paused(self, paused: bool) -> None:
        with self._lock:
            self._display_paused = bool(paused)


class LiveEmitter:
    """프레임 루프에서 호출: 저속 배압 + UI 스로틀 + JPEG 인코드 + live_cb.

    live_cb 예외·실패는 삼킨다 — 계수 경로와 완전 분리(L4).
    """

    def __init__(
        self,
        live_cb: "Callable[[LiveSnapshot], None] | None",
        live_control: "LiveControl | None" = None,
        *,
        max_ui_fps: float = DEFAULT_MAX_UI_FPS,
        max_preview_edge: int = DEFAULT_MAX_PREVIEW_EDGE,
        jpeg_quality: int = 70,
    ) -> None:
        self._cb = live_cb
        self._control = live_control
        self._max_ui_fps = max(1.0, float(max_ui_fps))
        self._max_edge = max(32, int(max_preview_edge))
        self._jpeg_quality = int(jpeg_quality)
        self._last_emit_t = 0.0
        self._last_pace_t = 0.0
        self._enabled = live_cb is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def pace(self) -> None:
        """저속(speed<1)일 때만 wall-clock 배압. speed>=1이면 no-op."""
        if not self._enabled:
            return
        speed = self._control.get_speed() if self._control is not None else 1.0
        if speed >= 1.0:
            self._last_pace_t = time.perf_counter()
            return
        target_fps = max(0.5, BASE_PACE_FPS * speed)
        min_dt = 1.0 / target_fps
        now = time.perf_counter()
        if self._last_pace_t > 0:
            wait = min_dt - (now - self._last_pace_t)
            if wait > 0:
                time.sleep(wait)
        self._last_pace_t = time.perf_counter()

    def emit(
        self,
        frame_idx: int,
        total_frames: int,
        image: np.ndarray,
        counts: dict,
        label: str,
        processing_fps: float,
    ) -> None:
        """image: gray (H,W) 또는 BGR (H,W,3). UI fps 스로틀. 실패 무시."""
        if not self._enabled:
            return
        now = time.perf_counter()
        min_interval = 1.0 / self._max_ui_fps
        if self._last_emit_t > 0 and (now - self._last_emit_t) < min_interval:
            return
        try:
            src_h, src_w = int(image.shape[0]), int(image.shape[1])
            jpeg, pw, ph = encode_preview(
                image, max_edge=self._max_edge, quality=self._jpeg_quality)
            snap = LiveSnapshot(
                frame_idx=int(frame_idx),
                total_frames=int(total_frames),
                jpeg=jpeg,
                counts=dict(counts),
                label=str(label),
                processing_fps=float(processing_fps),
                preview_w=pw,
                preview_h=ph,
                source_w=src_w,
                source_h=src_h,
            )
            self._cb(snap)
            self._last_emit_t = time.perf_counter()
        except Exception:
            # 관찰 경로 실패는 계수에 영향 없음
            return


def encode_preview(
    image: np.ndarray,
    *,
    max_edge: int = DEFAULT_MAX_PREVIEW_EDGE,
    quality: int = 70,
) -> tuple[bytes, int, int]:
    """gray/BGR → 축소 BGR JPEG bytes + (w, h)."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    h, w = bgr.shape[:2]
    edge = max(h, w)
    if edge > max_edge:
        scale = max_edge / float(edge)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        w, h = nw, nh
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes(), w, h


def queue_put_drop(q, item) -> bool:
    """maxsize 큐에 put; 가득 차면 한 칸 비우고 최신만 남김. 성공 True."""
    import queue as _queue

    for _ in range(4):
        try:
            q.put_nowait(item)
            return True
        except _queue.Full:
            try:
                q.get_nowait()
            except _queue.Empty:
                pass
    return False
