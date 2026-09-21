"""annotate.py 단위 테스트: rhythm 마크 박스의 재드로잉 등가성."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bubble_counter.annotate import redraw_rhythm_marks, save_rhythm_strip
from bubble_counter.rhythm import Mark


def _strip(seed: int = 5, rows: int = 30, width: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random((rows, width)) < 0.3, 255, 0).astype(np.uint8)


def _full_marks() -> list[Mark]:
    # 스트립 범위는 프레임 [100, 130). 안쪽/시작 걸침/끝 걸침/완전 밖 각 1개.
    return [
        Mark(first_frame=105, last_frame=110, position_px=10.0, width_px=5, area_px=20),
        Mark(first_frame=95, last_frame=102, position_px=30.0, width_px=4, area_px=12),
        Mark(first_frame=128, last_frame=140, position_px=20.0, width_px=6, area_px=30),
        Mark(first_frame=50, last_frame=60, position_px=5.0, width_px=3, area_px=9),
    ]


def test_redraw_equals_drawing_full_list_from_scratch(tmp_path):
    """부분 마크로 저장한 PNG를 전체 마크로 redraw하면, 처음부터 전체 마크로
    그린 PNG와 픽셀 단위로 동일해야 한다 (idempotent redraw 계약)."""
    strip = _strip()
    full = _full_marks()
    partial = full[:2]

    repaired = tmp_path / "repaired.png"
    save_rhythm_strip(repaired, strip, 100, partial)
    redraw_rhythm_marks(repaired, 100, full)

    reference = tmp_path / "reference.png"
    save_rhythm_strip(reference, strip, 100, full)

    a = cv2.imread(str(repaired))
    b = cv2.imread(str(reference))
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_redraw_missing_file_raises_ioerror(tmp_path):
    with pytest.raises(IOError, match="다시 열 수 없습니다"):
        redraw_rhythm_marks(tmp_path / "nope.png", 0, [])
