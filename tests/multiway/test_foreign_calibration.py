"""이물 캘리브레이션 게이트 테스트 (R0 §6.2)."""

import cv2
import numpy as np
import pytest

from bubble_counter.multiway import foreign
from bubble_counter.multiway.model import (
    BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec,
)

FIXTURE_DIR = "tests/multiway/fixtures/median_frames/"


# 1_3 canonical Points (분리이동 구간 6way_1_3 기준)
CANONICAL_POINTS = tuple(PointSpec(
    number=i, cx=0.15 + 0.15 * (i - 1), cy=0.5,
    length=0.08, width=0.06, angle_deg=0.0
) for i in range(1, 7))

# 1_2 전용 Points (1_2 칩 레이아웃은 1_3과 다름 - 상단 채널 영역)
# 1_2 파티클 (1118, 63) 위치 기반 - U2 통과 기포 분열 관련
POINTS_1_2 = (
    PointSpec(1, 0.15, 0.10, 0.08, 0.06, 0.0),   # 상단 좌측 채널
    PointSpec(2, 0.30, 0.10, 0.08, 0.06, 0.0),
    PointSpec(3, 0.45, 0.10, 0.08, 0.06, 0.0),
    PointSpec(4, 0.60, 0.10, 0.08, 0.06, 0.0),
    PointSpec(5, 0.75, 0.10, 0.08, 0.06, 0.0),
    PointSpec(6, 0.90, 0.10, 0.08, 0.06, 0.0),   # 상단 우측 - 파티클 (1118,63) 커버
)


def _cfg(points):
    return MultiwayConfig(
        RecordingSpec(1280, 768, 300.0, "dma"),
        points,
        BubbleSizeSpec("circle", 24.0, 24.0),
    )


def _load_median_std(chapter_stem: str):
    """픽스처에서 median.png + std4.png 로드."""
    med = cv2.imread(f"{FIXTURE_DIR}{chapter_stem}.png", cv2.IMREAD_GRAYSCALE)
    std4 = cv2.imread(f"{FIXTURE_DIR}{chapter_stem}_std4.png", cv2.IMREAD_GRAYSCALE)
    assert med is not None, f"median not found: {chapter_stem}.png"
    assert std4 is not None, f"std4 not found: {chapter_stem}_std4.png"
    std = std4.astype(np.float32) / 4.0
    return med, std


CHAPTERS = [
    ("6way_1_1", _cfg(CANONICAL_POINTS), 1),
    ("6way_1_2", _cfg(POINTS_1_2), 2),  # 1_2 전용 Points
    ("6way_1_3", _cfg(CANONICAL_POINTS), 3),
    ("6way_1_4", _cfg(CANONICAL_POINTS), 4),
    ("6way_2_1", _cfg(CANONICAL_POINTS), 5),
    ("6way_2_2", _cfg(CANONICAL_POINTS), 6),
    ("6way_2_3", _cfg(CANONICAL_POINTS), 7),
    ("6way_2_4", _cfg(CANONICAL_POINTS), 8),
    ("6way_3_1", _cfg(CANONICAL_POINTS), 9),
    ("6way_3_2", _cfg(CANONICAL_POINTS), 10),
    ("6way_3_3", _cfg(CANONICAL_POINTS), 11),
    ("6way_3_4", _cfg(CANONICAL_POINTS), 12),
]


@pytest.mark.parametrize("stem,cfg,ch_num", CHAPTERS)
def test_foreign_calibration(stem, cfg, ch_num):
    """각 챕터별 이물 스캔 검증."""
    med, std = _load_median_std(stem)
    out = foreign.scan_stats(med, std, cfg, source="review-scan")

    near_objs = [fo for fo in out if fo.near_points]

    if ch_num == 2:  # 6way_1_2 = 1_2 챕터 (파티클 (1118,63))
        assert len(near_objs) >= 1, f"{stem}: 1_2 파티클 near-검출 0개 (예상 ≥1)"
        particle_found = any(
            abs(fo.x_px - 1118) <= 15 and abs(fo.y_px - 63) <= 15
            for fo in near_objs
        )
        assert particle_found, f"{stem}: 1_2 파티클 (1118,63)±15px 근처에 검출 없음"
    else:
        assert len(near_objs) == 0, f"{stem}: near-오검출 {len(near_objs)}개 (예상 0)"


def test_canonical_points_roi_overlap():
    """canonical 6 ROI가 겹치지 않는지 검증 (프리플라이트 기하 검증과 정합)."""
    from bubble_counter.multiway.engine import _geometry_check
    rep = _geometry_check(_cfg(CANONICAL_POINTS))
    overlap_warnings = [w for w in rep.warnings if "겹칩니다" in w]
    assert len(overlap_warnings) == 0, f"canonical ROI 겹침 경고: {overlap_warnings}"
