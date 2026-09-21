"""multiway/audit.py — 표본 감사 테스트."""

import json
import pytest
import tempfile
from pathlib import Path

from bubble_counter.multiway.audit import (
    select_samples, load_audit, append_round, estimates,
    DEFAULT_SEED, EMPTY_MIN_GAP,
)
from bubble_counter.multiway.model import BubbleEvent, ChapterResult, AuditSample


def _events():
    # 3 Point, 300프레임
    return [
        BubbleEvent(1, 100, 0.0, (), 50.0, 44.0),
        BubbleEvent(2, 200, 0.0, (), 50.0, 44.0),
        BubbleEvent(3, 300, 0.0, (), 50.0, 44.0),
        BubbleEvent(1, 400, 0.0, (), 50.0, 44.0),
        BubbleEvent(2, 500, 0.0, (), 50.0, 44.0),
        BubbleEvent(3, 600, 0.0, (), 50.0, 44.0),
    ]


class TestSelectSamples:
    def test_event_sampling_uniform(self):
        events = _events()
        samples = select_samples(events, 300, way=3, round_no=1, n_events=2, n_empty=2)
        event_samples = [s for s in samples if s.kind == "event"]
        assert len(event_samples) == 2
        # seed 고정 → 결정론적
        samples2 = select_samples(events, 300, way=3, round_no=1, n_events=2, n_empty=2)
        assert [(s.point, s.frame) for s in event_samples] == [(s.point, s.frame) for s in [s for s in samples2 if s.kind == "event"]]

    def test_empty_sampling_respects_gap(self):
        events = _events()
        samples = select_samples(events, 300, way=3, round_no=1, n_events=0, n_empty=6)
        empty = [s for s in samples if s.kind == "empty"]
        assert len(empty) == 6
        # 각 empty 프레임은 이벤트에서 EMPTY_MIN_GAP(12) 이상 떨어짐
        for e in empty:
            for ev in _events():
                if ev.point == e.point:
                    assert abs(ev.frame - e.frame) >= 12

    def test_exclude_works(self):
        events = _events()
        exclude = frozenset([("event", 1, 100), ("empty", 2, 150)])
        # exclude된 이벤트/empty는 선택되지 않아야 함
        samples = select_samples(events, 300, way=3, round_no=1, n_events=10, n_empty=10, exclude=exclude)
        event_points = [(s.point, s.frame) for s in samples if s.kind == "event"]
        assert (1, 100) not in event_points
        empty_points = [(s.point, s.frame) for s in samples if s.kind == "empty"]
        assert (2, 150) not in empty_points

    def test_round_no_changes_seed(self):
        events = _events()
        s1 = select_samples(events, 300, way=3, round_no=1, n_events=2, n_empty=2)
        s2 = select_samples(events, 300, way=3, round_no=2, n_events=2, n_empty=2)
        e1 = [(s.point, s.frame) for s in s1 if s.kind == "event"]
        e2 = [(s.point, s.frame) for s in s2 if s.kind == "event"]
        # round_no 다르면 다른 표본 선택 (seed + round_no)
        assert e1 != e2


class TestEstimates:
    def test_zero_wrong_gives_upper(self):
        rounds = [{"round": 1, "samples": [
            {"kind": "event", "point": 1, "frame": 100, "verdict": "O"},
            {"kind": "empty", "point": 2, "frame": 200, "verdict": "O"},
        ]}]
        est = estimates(rounds)
        assert est["overcount_rate"] == 0.0
        assert est["overcount_upper95"] == 3.0 / 1  # rule-of-three
        assert est["miss_rate"] == 0.0
        assert est["miss_upper95"] == 3.0 / 1

    def test_some_wrong_point_estimate(self):
        rounds = [{"round": 1, "samples": [
            {"kind": "event", "point": 1, "frame": 100, "verdict": "O"},
            {"kind": "event", "point": 2, "frame": 200, "verdict": "X"},
            {"kind": "empty", "point": 1, "frame": 300, "verdict": "O"},
            {"kind": "empty", "point": 2, "frame": 400, "verdict": "X"},
        ]}]
        est = estimates(rounds)
        assert est["overcount_rate"] == 0.5  # 1/2
        assert est["overcount_upper95"] is None  # k>0
        assert est["miss_rate"] == 0.5  # 1/2
        assert est["miss_upper95"] is None

    def test_moalm_excluded(self):
        rounds = [{"round": 1, "samples": [
            {"kind": "event", "point": 1, "frame": 100, "verdict": "모름"},
            {"kind": "empty", "point": 2, "frame": 200, "verdict": "O"},
        ]}]
        est = estimates(rounds)
        # 모름은 분모에서 제외
        assert est["overcount_rate"] == 0.0
        assert est["miss_rate"] == 0.0


class TestAuditIO:
    def test_append_round_atomic(self, tmp_path):
        from bubble_counter.multiway.audit import append_round
        folder = tmp_path / "audit_test"
        folder.mkdir()

        append_round(folder, 1, [
            {"kind": "event", "point": 1, "frame": 100, "verdict": "O"},
            {"kind": "empty", "point": 2, "frame": 200, "verdict": "X"},
        ])

        data = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        assert data["rounds"][0]["round"] == 1
        assert len(data["rounds"][0]["samples"]) == 2
        assert data["estimates"]["event_audited"] == 1
        assert data["estimates"]["empty_audited"] == 1
        # 1 empty 검증 중 1개 X → miss_rate=1.0, upper95=None (k>0이므로)
        assert data["estimates"]["miss_rate"] == 1.0
        assert data["estimates"]["miss_upper95"] is None

    def test_round_accumulation(self, tmp_path):
        from bubble_counter.multiway.audit import append_round, load_audit
        folder = tmp_path / "audit"
        folder.mkdir()

        append_round(folder, 1, [{"kind": "event", "point": 1, "frame": 100, "verdict": "O"}])
        append_round(folder, 2, [{"kind": "event", "point": 2, "frame": 200, "verdict": "X"}])

        audit = load_audit(folder)
        assert len(audit["rounds"]) == 2
        # estimates는 전 라운드 누적
        assert audit["estimates"]["event_audited"] == 2
        assert audit["estimates"]["event_wrong"] == 1