"""has_u2 단위 테스트: 스펙표·모드 판별·fps 후보 (설계 §1.2)."""

from __future__ import annotations

import pytest

from bubble_counter.multiway.has_u2 import (
    FPS_STEPS,
    HAS_U2_MAX_FPS,
    MEMORY_BYTES,
    fps_candidates,
    infer_mode,
    max_fps,
)


def test_memory_bytes_is_2gib():
    assert MEMORY_BYTES == 2 * 1024 ** 3


def test_table_spot_values():
    assert HAS_U2_MAX_FPS[(1280, 768)] == {"dma": 300, "memory": 500}
    assert HAS_U2_MAX_FPS[(2592, 2048)] == {"dma": 60, "memory": 100}
    assert HAS_U2_MAX_FPS[(512, 480)] == {"dma": 1000, "memory": 1500}
    assert len(HAS_U2_MAX_FPS) == 10


def test_fps_steps_are_sorted_standard():
    assert FPS_STEPS == (60, 100, 150, 200, 250, 300, 400, 500, 800, 1000, 1500, 2000)


def test_max_fps_lookup():
    assert max_fps(1280, 768, "dma") == 300
    assert max_fps(1280, 768, "memory") == 500


def test_max_fps_unknown_resolution_is_none():
    # 표에 없는 해상도(ROI 크롭 등) → None = 직접 입력 폴백 신호
    assert max_fps(999, 555, "dma") is None


def test_max_fps_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        max_fps(1280, 768, "ram")


def test_infer_mode_small_clip_fits_memory():
    # 1280x768x3 = 2,949,120 B/frame; 2GiB / that ≈ 728.17 → 728프레임까지 메모리 가능
    assert infer_mode(1280, 768, 728) == "memory"


def test_infer_mode_large_clip_forces_dma():
    assert infer_mode(1280, 768, 729) == "dma"


def test_infer_mode_zero_frames_is_memory():
    assert infer_mode(1280, 768, 0) == "memory"


def test_fps_candidates_dma_1280x768():
    assert fps_candidates(1280, 768, "dma") == (60, 100, 150, 200, 250, 300)


def test_fps_candidates_memory_1280x768():
    assert fps_candidates(1280, 768, "memory") == (60, 100, 150, 200, 250, 300, 400, 500)


def test_fps_candidates_unknown_resolution_is_empty():
    # None(표에 없음) → 후보 없음 = GUI가 직접 입력 폴백으로 처리
    assert fps_candidates(999, 555, "dma") == ()
