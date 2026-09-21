"""Video reading: ROI crop + downscale, without ever skipping frames."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bubble_counter.config import RoiSpec


@dataclass
class VideoInfo:
    """Full-frame metadata as reported by the container (may be unreliable)."""

    fps: float  # as reported by container (may be 0/nan)
    frame_count: int  # may be <= 0 if unknown
    width: int
    height: int


class VideoSource:
    """OpenCV video reader with ROI crop + downscale. NEVER skips frames."""

    def __init__(self, path: str | Path, roi: RoiSpec = RoiSpec(), scale: float = 1.0):
        if not (0 < scale <= 1):
            raise ValueError(f"scale은 0보다 크고 1 이하여야 합니다: {scale}")
        if not Path(path).exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다: {path}")

        self._cap = cap
        self._scale = scale
        self.roi = roi

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info = VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=width,
            height=height,
        )

        self.roi_px = roi.to_pixels(width, height)
        _, _, crop_w, crop_h = self.roi_px
        self.out_size = (
            max(1, round(crop_w * scale)),
            max(1, round(crop_h * scale)),
        )

    def frames(self, with_color: bool = False):
        """Yield (frame_idx, gray, bgr_or_None). frame_idx from 0, consecutive.

        gray: uint8 (out_h, out_w) — cropped, INTER_AREA-resized (scale<1),
        grayscale. bgr: same geometry BGR frame when with_color else None.
        """
        x0, y0, w, h = self.roi_px
        out_w, out_h = self.out_size
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            cropped = frame[y0 : y0 + h, x0 : x0 + w]
            if self._scale < 1.0:
                resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cropped
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            bgr = resized if with_color else None
            yield frame_idx, gray, bgr
            frame_idx += 1

    def seek(self, frame_idx: int) -> None:
        """디코더를 frame_idx로 이동. 인트라 온리 코덱(FFV1/rawvideo/MJPG)에서 정확 시킹."""
        if frame_idx < 0:
            raise ValueError(f"frame_idx는 0 이상이어야 합니다: {frame_idx}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def sample_gray(self, indices: "list[int]") -> "list[np.ndarray]":
        """indices의 프레임들을 frames()와 동일 지오메트리 gray로 수집.

        프라이밍 1차 패스 전용.  호출이 끝나면 seek(0)으로 되감아 2차 패스의
        frames()가 프레임 0부터 순차 디코드하도록 보장한다.  읽기 실패(범위 밖 등)
        인덱스는 조용히 건너뛴다(반환 길이가 줄어들 수 있음).
        """
        x0, y0, w, h = self.roi_px
        out_w, out_h = self.out_size
        result: "list[np.ndarray]" = []
        for idx in indices:
            self.seek(int(idx))
            ret, frame = self._cap.read()
            if not ret:
                continue
            cropped = frame[y0 : y0 + h, x0 : x0 + w]
            if self._scale < 1.0:
                resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cropped
            result.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        self.seek(0)
        return result

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


def resolve_fps(info: VideoInfo, override: float | None) -> float:
    """Resolve the fps to use for time calculations.

    override(>0) wins; else info.fps if finite>0; else raise ValueError
    ('영상에 fps 정보가 없습니다. 직접 지정하세요 (CLI: --fps, GUI: 'fps 강제' 칸).').
    """
    if override is not None and override > 0:
        return override
    if math.isfinite(info.fps) and info.fps > 0:
        return info.fps
    raise ValueError("영상에 fps 정보가 없습니다. 직접 지정하세요 (CLI: --fps, GUI: 'fps 강제' 칸).")


def probe_video(path: "str | Path") -> VideoInfo:
    """영상을 열어 메타데이터(VideoInfo)만 읽고 즉시 닫는다 — 프레임 소비 없음.

    GUI의 영상 정보 라벨과 밴드 자동 환산용. 실패 예외는 VideoSource와 동일
    (FileNotFoundError / IOError)이라 호출자는 한 가지 계약만 알면 된다.
    """
    with VideoSource(path) as source:
        return source.info
