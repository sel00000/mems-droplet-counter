"""multiway/corrections.py 테스트."""

import tempfile
from pathlib import Path

from bubble_counter.multiway.corrections import (
    make_correction, load_corrections, append_correction,
    append_corrections, remove_correction, counts_with,
)
from bubble_counter.multiway.model import Correction


def test_make_correction():
    c = make_correction(1, 100, +1, "수동 +")
    assert c.point == 1
    assert c.frame == 100
    assert c.delta == 1
    assert c.reason == "수동 +"
    assert c.evidence_path == ""
    assert c.corrected_at  # ISO timestamp


def test_load_empty_folder(tmp_path):
    assert load_corrections(tmp_path) == []


def test_append_single_correction(tmp_path):
    c = make_correction(1, 100, +1, "test")
    append_correction(tmp_path, c)
    loaded = load_corrections(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].point == 1
    assert loaded[0].delta == 1


def test_append_multiple_corrections(tmp_path):
    cs = [make_correction(i, 100, +1, f"c{i}") for i in range(1, 4)]
    append_corrections(tmp_path, cs)
    loaded = load_corrections(tmp_path)
    assert len(loaded) == 3


def test_remove_correction_undo(tmp_path):
    cs = [make_correction(i, 100, +1, f"c{i}") for i in range(1, 4)]
    append_corrections(tmp_path, cs)
    assert len(load_corrections(tmp_path)) == 3

    # 중간 것 제거
    remove_correction(tmp_path, 1)  # index 1 = c2
    loaded = load_corrections(tmp_path)
    assert len(loaded) == 2
    assert loaded[0].frame == 100
    assert loaded[1].frame == 100
    assert loaded[0].reason == "c1"
    assert loaded[1].reason == "c3"


def test_counts_with():
    auto = {1: 10, 2: 11, 3: 9}
    cs = [
        Correction(1, 100, +1, "r1", "t"),
        Correction(1, 200, -1, "r2", "t"),
        Correction(3, 300, +1, "r3", "t"),
    ]
    final = counts_with(auto, cs)
    assert final == {1: 10, 2: 11, 3: 10}  # 1: +1-1=0, 3: +1


def test_counts_with_original_unchanged():
    auto = {1: 10}
    cs = [Correction(1, 100, +1, "r", "t")]
    final = counts_with(auto, cs)
    assert final == {1: 11}
    assert auto == {1: 10}  # 원본 불변


def test_correction_delta_validation():
    from bubble_counter.multiway.corrections import make_correction
    import pytest
    # delta는 +1 또는 -1만 가능 (모델에서 검증)
    from bubble_counter.multiway.model import Correction
    with pytest.raises(ValueError, match="delta"):
        Correction(1, 100, 0, "r", "t")
    with pytest.raises(ValueError, match="delta"):
        Correction(1, 100, 2, "r", "t")