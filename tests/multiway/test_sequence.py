"""multiway/sequence.py — 순서·set 상태기계."""

from collections import Counter

import pytest

from bubble_counter.multiway.model import BubbleEvent, Correction
from bubble_counter.multiway.sequence import (
    SequenceOutcome, analyze, apply_corrections, gate_analyze,
)


def ev(point, frame, sub=0.0):
    return BubbleEvent(point=point, frame=frame, subframe=sub)


def perfect(way, sets, step=10):
    out, f = [], 0
    for _ in range(sets):
        for p in range(1, way + 1):
            out.append(ev(p, f)); f += step
    return out


def _identity_holds(events, outcome, way):
    total = sum(Counter(e.point for e in events).values())
    return total == outcome.sets_completed * way + sum(outcome.residual.values()) + outcome.absorbed


class TestAnalyze:
    def test_perfect_three_sets(self):
        events = perfect(6, 3)
        o = analyze(events, way=6)
        assert o.sets_completed == 3
        assert o.violations == []
        assert o.residual == {} and o.absorbed == 0
        assert _identity_holds(events, o, 6)

    def test_partial_final_set_is_residual(self):
        events = perfect(6, 1) + [ev(1, 100), ev(2, 110)]   # 2개짜리 미완성 사이클
        o = analyze(events, way=6)
        assert o.sets_completed == 1
        assert o.residual == {1: 1, 2: 1}
        assert _identity_holds(events, o, 6)

    def test_missing_point_makes_violation_and_resyncs(self):
        # way=4, 순서 1,2,4  → 3을 기대했는데 4 도착
        events = [ev(1, 0), ev(2, 10), ev(4, 20)]
        o = analyze(events, way=4)
        assert len(o.violations) == 1
        v = o.violations[0]
        assert (v.expected_point, v.actual_point, v.frame) == (3, 4, 20)
        assert o.absorbed == 2   # D-5: 위반 시 끊긴 런([1,2], 길이2)이 absorbed로 흡수됨
        assert _identity_holds(events, o, 4)

    def test_tolerance_reorders_within_window(self):
        # 4,5,6이 순서 뒤엉킴(같은 창): 5,4,6 → tolerance로 재정렬해 위반 0
        events = [ev(1, 0), ev(2, 5), ev(3, 10), ev(5, 20), ev(4, 22), ev(6, 24)]
        o_strict = analyze(events, way=6, tolerance_frames=0)
        o_tol = analyze(events, way=6, tolerance_frames=5)
        assert len(o_strict.violations) >= 1
        assert o_tol.violations == [] and o_tol.sets_completed == 1
        assert _identity_holds(events, o_tol, 6)

    def test_identity_random(self):
        import random
        rng = random.Random(4)
        events = [ev(rng.randint(1, 6), i * 3) for i in range(200)]
        o = analyze(events, way=6, tolerance_frames=3)
        assert _identity_holds(events, o, 6)


class TestViolationClearsRun:
    """리뷰 D-5 (Critical) 회귀: 위반 발생 시 진행 중이던 cycle이 absorbed로 빠지고
    위반 이벤트 자신이 새 런의 첫 원소가 되어야 한다.  그렇지 않으면 위반 이전
    잔존 point가 위반 이후 재개된 청정 전진과 섞여, point가 중복/누락된 채로
    set 완성이 오탐될 수 있다(수정 전 버그)."""

    def test_violation_then_resumed_run_completes_with_distinct_points(self):
        # 리뷰 재현 사례: P1@0(매치) → P3@10(위반,expected=2) → P1@20,P2@30(새 런 3→1→2 완성)
        events = [ev(1, 0), ev(3, 10), ev(1, 20), ev(2, 30)]
        o = analyze(events, way=3)
        assert len(o.violations) == 1
        v = o.violations[0]
        assert (v.expected_point, v.actual_point, v.frame) == (2, 3, 10)
        assert o.absorbed == 1                # 위반 이전 잔존 cycle=[1] (길이1)만 흡수
        assert o.sets_completed == 1           # 3→1→2: 서로 다른 point 3개로 완성(중복 없음)
        assert o.residual == {}
        assert _identity_holds(events, o, 3)

    def test_violation_leaves_new_run_as_residual_when_incomplete(self):
        events = [ev(1, 0), ev(3, 10), ev(1, 20)]
        o = analyze(events, way=3)
        assert o.sets_completed == 0
        assert o.absorbed == 1
        assert o.residual == {3: 1, 1: 1}      # 위반 이벤트(3)+뒤이은 매치(1) = 새 런
        assert _identity_holds(events, o, 3)

    def test_no_violation_two_cycles_way3(self):
        events = perfect(3, 2)
        o = analyze(events, way=3)
        assert o.sets_completed == 2
        assert o.violations == []
        assert o.residual == {} and o.absorbed == 0
        assert _identity_holds(events, o, 3)


class TestGateAnalyze:
    """D-21 순서 게이트: 수락만 +1, count error는 미계수·expected 유지."""

    def test_perfect_sets_all_accepted(self):
        events = perfect(6, 2)
        g = gate_analyze(events, way=6)
        assert len(g.accepted) == 12
        assert g.count_errors == []
        assert g.sets_completed == 2
        assert g.residual == {}

    def test_wave_without_one_all_count_errors(self):
        # 질문1-B: 2→3→4→5→6 (1 없음) → 전부 count error
        events = [ev(p, p * 10) for p in range(2, 7)]
        g = gate_analyze(events, way=6)
        assert g.accepted == []
        assert len(g.count_errors) == 5
        assert all(ce.expected_point == 1 for ce in g.count_errors)
        assert g.sets_completed == 0

    def test_jump_keeps_expected(self):
        # 질문2-B: 1→2→3 후 5 → count error, 기대 4 유지, 이후 4 수락
        events = [ev(1, 0), ev(2, 10), ev(3, 20), ev(5, 30), ev(4, 40), ev(5, 50), ev(6, 60)]
        g = gate_analyze(events, way=6)
        assert [e.point for e in g.accepted] == [1, 2, 3, 4, 5, 6]
        assert len(g.count_errors) == 1
        assert (g.count_errors[0].expected_point, g.count_errors[0].actual_point,
                g.count_errors[0].frame) == (4, 5, 30)
        assert g.sets_completed == 1

    def test_gate_identity_accepted_only(self):
        events = perfect(4, 1) + [ev(2, 100), ev(3, 110)]  # 2,3 without leading 1
        g = gate_analyze(events, way=4)
        assert len(g.accepted) == 4
        assert len(g.count_errors) == 2
        assert len(g.accepted) == g.sets_completed * 4 + sum(g.residual.values())


class TestApplyCorrections:
    def test_plus_one_inserts_synthetic_event(self):
        events = [ev(1, 0), ev(2, 10)]        # way=3 미완성
        o = apply_corrections(events, [Correction(3, 15, +1, "누락", "t")], way=3)
        assert o.sets_completed == 1
        assert list(events) == [ev(1, 0), ev(2, 10)]     # 원본 불변

    def test_minus_one_removes_nearest(self):
        events = [ev(1, 0), ev(1, 100), ev(2, 10), ev(3, 20)]
        # point1의 frame 90 최근접(=100) 제거 → 1,2,3 완성 set 1
        o = apply_corrections(events, [Correction(1, 90, -1, "파티클", "t")], way=3)
        assert o.sets_completed == 1
        assert len(events) == 4                # 원본 불변

    def test_minus_one_tie_break_prefers_smaller_frame(self):
        # 리뷰 D-5 (Important) 회귀: frame=100을 리스트 앞에 둬 입력 순서 의존성이
        # 없음을 확인한다. 거리 동률(|100-90|==|80-90|==10) → frame이 작은 80이
        # 제거되고 100이 생존해야 한다.
        events = [ev(1, 100), ev(1, 80), ev(2, 95)]
        o = apply_corrections(events, [Correction(1, 90, -1, "동률", "t")], way=2)
        # 80이 제거되면 정렬 순서가 [ev(2,95), ev(1,100)]이 되어 2가 먼저 도착 →
        # 위반 1건(expected=1인데 actual=2) 발생 후 1이 매치돼 set 완성.
        # (100이 제거돼 80이 남았다면 1,2 순서가 청정해 위반 0이 됐을 것.)
        assert len(o.violations) == 1
        v = o.violations[0]
        assert (v.expected_point, v.actual_point, v.frame) == (1, 2, 95)
        assert o.sets_completed == 1
