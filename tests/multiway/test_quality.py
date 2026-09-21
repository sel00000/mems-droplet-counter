"""multiway/quality.py — 신뢰 지표 빌더 (R0 §2.2)."""

import pytest

from bubble_counter.multiway.model import (
    BubbleEvent, BubbleSizeSpec, ChapterResult, MultiwayConfig, PointSpec, RecordingSpec)
from bubble_counter.multiway import quality


def _cfg(way=3):
    pts = tuple(PointSpec(i, 0.1 * i + 0.1, 0.5, 0.05, 0.05, 0.0) for i in range(1, way + 1))
    return MultiwayConfig(RecordingSpec(1280, 768, 300.0, "dma"), pts,
                          BubbleSizeSpec("circle", 44.0, 44.0))


def _ch(events=(), filtered=(), counts=None, suspects=(), truncated=False, sets=0):
    return ChapterResult(
        video_path="x.avi", recording=RecordingSpec(1280, 768, 300.0, "dma"),
        counts_auto=dict(counts or {}), corrections=[], events=list(events),
        violations=[], suspects=list(suspects), sets_completed=sets,
        processed_frames=1000, filtered_out=list(filtered), suspects_truncated=truncated)


def test_sibling_agreement_and_set_conversion():
    q = quality.build_quality(_ch(counts={1: 100, 2: 100, 3: 50}, sets=30), _cfg(3))
    assert q["sibling_agreement"] == pytest.approx((100 - 50) / 100)   # median 100
    assert q["set_conversion"] == pytest.approx(30 * 3 / 250)


def test_null_when_median_or_denominator_zero():
    q = quality.build_quality(_ch(counts={1: 0, 2: 0, 3: 0}), _cfg(3))
    assert q["sibling_agreement"] is None and q["set_conversion"] is None


def test_filtered_by_reason_and_total():
    filtered = [(1, 10, 12.0, "undersize"), (2, 20, 8.0, "undersize"),
                (1, 30, 40.0, "duplicate"), (3, 40, 5.0, "boundary-incomplete")]
    q = quality.build_quality(_ch(filtered=filtered), _cfg(3))
    assert q["filtered_total"] == 4
    assert q["filtered_by_reason"] == {"undersize": 2, "ring-residual": 0,
                                       "duplicate": 1, "boundary-incomplete": 1}


def test_per_point_flags_counted():
    evs = [BubbleEvent(1, 100, 0.0, ("no-upstream",), 50.0, 44.0),
           BubbleEvent(1, 200, 0.0, ("weak-bubble", "boundary-size"), 50.0, 20.0)]
    q = quality.build_quality(_ch(events=evs, counts={1: 2}), _cfg(1))
    assert q["per_point"]["1"]["count_final"] == 2
    assert q["per_point"]["1"]["flags"] == {"no-upstream": 1, "weak-bubble": 1,
                                            "merge-suspect": 0, "boundary-size": 1}


def test_split_pair_candidates_greedy():
    evs = [BubbleEvent(1, 100, 0.0, (), 50.0, 30.0), BubbleEvent(2, 102, 0.0, (), 50.0, 25.0),
           BubbleEvent(3, 500, 0.0, (), 50.0, 30.0)]   # 3은 짝 없음(간격 초과)
    q = quality.build_quality(_ch(events=evs, counts={1: 1, 2: 1, 3: 1}), _cfg(3))
    assert q["split_pair_candidates"] == 1


def test_audit_seed_constant_matches_r0():
    assert quality.build_quality(_ch(), _cfg(1))["audit_seed"] == 20260711