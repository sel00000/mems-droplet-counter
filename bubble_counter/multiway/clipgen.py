"""온디맨드 슬로모션 클립 생성 (R0 §3.5, P34-1/P34-2).

한글 경로 주의: cv2.VideoWriter는 비ASCII 경로에서 실패할 수 있어
ASCII 임시 폴더에 쓴 뒤 shutil.move로 최종 경로에 옮긴다.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..video_io import VideoSource
from .model import PointSpec

DEFAULT_HALF_WINDOW = 25          # ±25프레임 (UI에서 ±5~±100 조절)
DEFAULT_FPS_OUT = 10.0            # 300fps 촬영 → 1/30 슬로모션


def clip_path(folder, point: int, frame: int) -> Path:
    """결과 폴더 기준 클립 경로. 패딩 없음(D6 예시 P5_f6084.avi가 정본)."""
    return Path(folder) / "clips" / f"P{point}_f{frame}.avi"


@dataclass(frozen=True)
class ClipOverlay:
    """ROI·계수선·이벤트 마커 오버레이 스펙 (P34-2에서 사용). global_direction은
    현재 드로잉엔 불필요하나 R0 계약 필드로 보존(향후 흐름방향 표식 예약)."""

    points: "tuple[PointSpec, ...]"
    frame_w: int
    frame_h: int
    global_direction: str
    highlight_point: "int | None" = None
    event_frame: "int | None" = None


def render_clip(video_path, out_path, center_frame, *,
                half_window: int = DEFAULT_HALF_WINDOW, fps_out: float = DEFAULT_FPS_OUT,
                overlay: "ClipOverlay | None" = None,
                progress_cb: "Callable[[int, int], None] | None" = None) -> Path:
    """[center−hw, center+hw] 구간을 MJPG 슬로모션 클립으로 렌더. out_path 반환.

    경계는 [max(0, c−hw), min(n−1, c+hw)]로 클램프. 구간이 비면(영상 밖 center)
    ValueError. 존재 파일은 덮어쓴다(재렌더). progress_cb는 (쓴 프레임, 총 프레임).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = VideoSource(str(video_path))             # 기본 = 전체 프레임, scale=1.0
    tmp_dir = Path(tempfile.mkdtemp(prefix="clip_"))  # ASCII 임시 경로 (한글 경로 우회)
    tmp_avi = tmp_dir / "clip.avi"
    try:
        n = source.info.frame_count
        w = source.info.width
        h = source.info.height
        lo = max(0, center_frame - half_window)
        hi = center_frame + half_window if n <= 0 else min(n - 1, center_frame + half_window)
        if lo > hi:
            raise ValueError(f"클립 구간이 영상 범위 밖입니다: center={center_frame}, 총 {n}프레임")
        total = hi - lo + 1

        writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"),
                                 fps_out, (w, h), isColor=True)
        if not writer.isOpened():
            writer.release()
            raise IOError(f"클립 영상을 생성할 수 없습니다: {out_path}")
        written = 0
        try:
            source.seek(lo)
            for local_idx, _gray, bgr in source.frames(with_color=True):
                if local_idx >= total:
                    break
                real_frame = lo + local_idx
                frame = bgr if overlay is None else _draw_overlay(bgr.copy(), real_frame, overlay)
                writer.write(frame)
                written += 1
                if progress_cb is not None:
                    progress_cb(written, total)
        finally:
            writer.release()
        if written == 0:
            raise ValueError(f"쓴 프레임이 없습니다(디코드 실패?): {video_path}")
        # ASCII 임시 → 한글 최종 경로. 재렌더 덮어쓰기 위해 기존 파일 선삭제.
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(tmp_avi), str(out_path))
        return out_path
    finally:
        source.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _mid(a, b) -> "tuple[int, int]":
    return (int(round((a[0] + b[0]) / 2)), int(round((a[1] + b[1]) / 2)))


def _roi_polygon(p: PointSpec, w: int, h: int) -> "np.ndarray":
    """corners_px 순서 [(-u,-n),(-u,+n),(+u,-n),(+u,+n)] → 폴리곤 순서 [c0,c1,c3,c2]."""
    c = p.corners_px(w, h)
    return np.array([c[0], c[1], c[3], c[2]], dtype=np.int32).reshape((-1, 1, 2))


def _draw_overlay(frame: "np.ndarray", real_frame: int, ov: ClipOverlay) -> "np.ndarray":
    """P34-2: ROI(초록)·계수선(노랑, 폭축 중앙선)·프레임 번호·event_frame 강조(빨강)."""
    for p in ov.points:
        cv2.polylines(frame, [_roi_polygon(p, ov.frame_w, ov.frame_h)], True, (0, 255, 0), 1)
        c = p.corners_px(ov.frame_w, ov.frame_h)
        # 계수선 = length 중앙에서 채널을 가로지르는 폭축 선분: mid(c0,c2) ↔ mid(c1,c3)
        cv2.line(frame, _mid(c[0], c[2]), _mid(c[1], c[3]), (0, 255, 255), 1)
    if (ov.event_frame is not None and real_frame == ov.event_frame
            and ov.highlight_point is not None):
        for p in ov.points:
            if p.number == ov.highlight_point:
                cv2.polylines(frame, [_roi_polygon(p, ov.frame_w, ov.frame_h)], True,
                              (0, 0, 255), 2)
    text = f"frame={real_frame}"
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


__all__ = ["DEFAULT_HALF_WINDOW", "DEFAULT_FPS_OUT", "ClipOverlay", "clip_path", "render_clip"]
