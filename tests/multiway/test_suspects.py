"""multiway/suspects.py — 의심 판정·자기진단."""

from bubble_counter.rhythm import Mark
from bubble_counter.multiway.model import BubbleEvent, PointSpec, RecordingSpec, BubbleSizeSpec, MultiwayConfig
from bubble_counter.multiway.sequence import SequenceOutcome
from bubble_counter.multiway import suspects as S


def _cfg():
    return MultiwayConfig(RecordingSpec(200, 120, 300.0, "dma"),
                          (PointSpec(1, 0.5, 0.5, 0.5, 0.5, 0.0),),
                          BubbleSizeSpec("circle", 44.0, 44.0))


def mk(first, last, pos=30.0, width=44, area=100):
    return Mark(first_frame=first, last_frame=last, position_px=pos, width_px=width, area_px=area)


class TestDetect:
    def test_merge_flags_oversized_area(self):
        # merge_suspect_px = max(44,44) * 1.8 = 79.2px
        # 정상 마크 폭=44(<79.2), 대형 마크 폭=100(>79.2)
        marks = [mk(t, t, width=44) for t in range(0, 100, 10)] + [mk(200, 205, width=100)]
        sus, trunc = S.detect({1: marks}, [], _cfg(), 300.0)
        assert any(s.kind == "merge" and s.point == 1 for s in sus)
        assert trunc is False

    def test_slow_flags_long_dwell(self):
        marks = [mk(t, t + 2) for t in range(0, 100, 10)] + [mk(200, 240)]   # 지속 41 ≫ 중앙값 3×3
        sus, _ = S.detect({1: marks}, [], _cfg(), 300.0)
        assert any(s.kind == "slow" for s in sus)

    def test_gap_anomaly(self):
        firsts = [0, 10, 20, 30, 200]        # 마지막 간격 170 ≫ 중앙값 10×3
        marks = [mk(f, f) for f in firsts]
        sus, _ = S.detect({1: marks}, [], _cfg(), 300.0)
        assert any(s.kind == "gap-anomaly" for s in sus)

    def test_undersize_from_boundary_flag(self):
        events = [BubbleEvent(1, 50, 0.2, flags=("boundary-size",))]
        sus, _ = S.detect({1: []}, events, _cfg(), 300.0)
        assert any(s.kind == "undersize" and s.frame == 50 for s in sus)

    def test_cap_and_truncated(self):
        # 기준선(과반, 150개)에 대해 대형 마크(소수, 105개 — 상한 100 초과)를 이상치로
        # 배치한다. 다수파를 이상치로 만들면 모집단 중앙값 자체가 그 값으로 옮겨가
        # "중앙값×배수 초과" 판정이 구조적으로 성립 불가(중앙값 기반 탐지의 수학적
        # 한계) — 브리프 원안(기준선 10개 vs 대형 150개)은 이 함정에 해당해 소수/다수
        # 비율을 뒤집었다 (Deviations 참조).
        # merge_suspect_px = 79.2 → 정상 폭 44, 대형 폭 100(>79.2)
        marks = [mk(t, t, width=44) for t in range(0, 1500, 10)]            # 기준선 150개
        marks += [mk(2000 + t, 2000 + t, width=100) for t in range(0, 105)]  # 대형 105개(소수)
        sus, trunc = S.detect({1: marks}, [], _cfg(), 300.0)
        assert len(sus) == S.CHAPTER_SUSPECT_CAP
        assert trunc is True


class TestSelfCheck:
    def test_deviation_warns(self):
        o = SequenceOutcome([], 90, {}, 0)
        msgs = S.self_check({1: 100, 2: 90, 3: 100}, o, way=3)
        assert any("Point 2" in m for m in msgs)

    def test_arithmetic_mismatch_reports_internal_error(self):
        # counts합 10, 그러나 outcome은 sets2×way3=6 + 잔여0 + 흡수0 = 6 → 불일치
        o = SequenceOutcome([], 2, {}, 0)
        msgs = S.self_check({1: 4, 2: 3, 3: 3}, o, way=3)
        assert any("내부 오류" in m for m in msgs)

    def test_consistent_no_internal_error(self):
        o = SequenceOutcome([], 3, {}, 0)          # 3×3=9
        msgs = S.self_check({1: 3, 2: 3, 3: 3}, o, way=3)
        assert not any("내부 오류" in m for m in msgs)
