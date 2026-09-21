"""GUI 입력 사전 검증 — tkinter를 import하지 않는 순수 stdlib 모듈.

여기의 범위는 gui.py 위젯(Spinbox from_/to)과 동일한 GUI 전용 정책이다.
CLI에는 적용하지 않는다 (CLI는 config/파이프라인의 자체 검증만 따른다).
tkinter가 없는 환경(예: WSL dev 머신)에서도 단위 테스트할 수 있도록
gui.py에서 분리되어 있다.
"""

from __future__ import annotations

import math

# (필드 라벨, 최소, 최대) — gui.py의 Spinbox from_/to와 동기.
_RANGES = {
    "band_px": ("밴드 폭(px)", 1, 512),
    "bucket_seconds": ("시간 버킷(초)", 0.1, 3600.0),
    "var_threshold": ("var-threshold", 1.0, 500.0),
    "merge_gap_frames": ("merge-gap(프레임)", 0, 1000),
}


def _check_range(field: str, value: float) -> None:
    label, lo, hi = _RANGES[field]
    if not (lo <= value <= hi):
        raise ValueError(f"{label}은(는) {lo}~{hi} 범위여야 합니다: {value}")


def validate_params(band_px: int, bucket_seconds: float, var_threshold: float,
                    merge_gap_frames: int, fps_text: str) -> float | None:
    """Spinbox/Entry 값을 GUI 범위로 검증하고 fps_override를 반환한다.

    fps_text: 'fps 강제' 칸의 원문 (빈칸이면 자동 = None 반환).
    범위 밖/파싱 불가 값은 한국어 ValueError — gui.py의 기존
    messagebox 경로("입력값을 확인하세요")가 그대로 잡는다.
    """
    _check_range("band_px", band_px)
    _check_range("bucket_seconds", bucket_seconds)
    _check_range("var_threshold", var_threshold)
    _check_range("merge_gap_frames", merge_gap_frames)

    text = fps_text.strip()
    if not text:
        return None
    try:
        fps = float(text)
    except ValueError:
        raise ValueError(f"fps 강제 값은 숫자여야 합니다: {text!r}") from None
    if not math.isfinite(fps):
        raise ValueError(f"fps 강제 값은 유한한 숫자여야 합니다: {text!r}")
    if fps <= 0:
        raise ValueError(f"fps 강제 값은 0보다 커야 합니다: {fps}")
    return fps
