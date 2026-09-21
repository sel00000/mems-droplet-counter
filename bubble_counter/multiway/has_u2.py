"""HAS-U2 고속카메라 내장 스펙표(설계 §1.2) — fps 매트릭스·모드 판별·fps 후보.

컨테이너 fps는 절대 신뢰하지 않으므로(rawvideo AVI가 가짜 15fps를 보고함),
사용자가 측정 전 RecordingSpec을 선행 선택한다. 이 모듈은 그 선택 UI(트랙 B
위저드 1단계)와 프리플라이트(트랙 A)가 참조하는 공식 제원 상수·헬퍼다.

값은 DITECT 공식 표 이미지 판독본이며 원본 이미지를 docs/에 함께 커밋해
구현 시 재대조한다.
"""

from __future__ import annotations

from .model import VALID_MODES

# (width, height) → 모드별 최대 fps. 선택 가능한 fps 후보는 FPS_STEPS 중 최대치 이하 전부.
HAS_U2_MAX_FPS: dict[tuple[int, int], dict[str, int]] = {
    (2592, 2048): {"dma": 60,   "memory": 100},
    (1920, 1080): {"dma": 150,  "memory": 250},
    (1600, 1200): {"dma": 150,  "memory": 250},
    (1280, 1024): {"dma": 200,  "memory": 300},
    (1024, 1024): {"dma": 250,  "memory": 400},
    (1280, 768):  {"dma": 300,  "memory": 500},   # ← 이번 6way 영상 12개
    (1024, 768):  {"dma": 300,  "memory": 500},
    (800, 600):   {"dma": 500,  "memory": 800},
    (640, 480):   {"dma": 800,  "memory": 1000},
    (512, 480):   {"dma": 1000, "memory": 1500},
}

MEMORY_BYTES = 2 * 1024 ** 3       # 내장 메모리 2GB

FPS_STEPS = (60, 100, 150, 200, 250, 300, 400, 500, 800, 1000, 1500, 2000)


def infer_mode(width: int, height: int, n_frames: int) -> str:
    """프레임수 × 프레임크기(w×h×3) > 2GB → 내장메모리로는 물리적으로 불가 → 'dma'."""
    frame_bytes = width * height * 3
    if n_frames * frame_bytes > MEMORY_BYTES:
        return "dma"
    return "memory"


def max_fps(width: int, height: int, mode: str) -> "int | None":
    """(해상도, 모드)의 최대 fps. 표에 없는 해상도면 None(직접 입력 폴백)."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode는 {VALID_MODES} 중 하나여야 합니다: {mode!r}")
    entry = HAS_U2_MAX_FPS.get((width, height))
    if entry is None:
        return None
    return entry[mode]


def fps_candidates(width: int, height: int, mode: str) -> tuple[int, ...]:
    """최대 fps 이하의 표준 스텝 전부. 표에 없으면 빈 튜플(GUI가 직접 입력으로 폴백)."""
    top = max_fps(width, height, mode)
    if top is None:
        return ()
    return tuple(f for f in FPS_STEPS if f <= top)
