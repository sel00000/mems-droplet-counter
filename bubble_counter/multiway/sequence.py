"""이벤트 스트림 → 순서·set 상태기계 (설계 결정 4) + 순서 게이트(D-21).

analyze (order_gate=False, 구 동작):
  expected=1에서 청정 전진, 불일치 시 expected=actual+1 재동기. 이벤트는 모두
  이미 계수된 전제로 set만 분석. 항등식: sum==sets×way+residual+absorbed.

gate_analyze (order_gate=True, 사용자 확정):
  순서에 맞는 통과만 수락(+1). 불일치=count error(미계수), expected 불변.
  set 시작은 반드시 1. 2→…→6(1 없음)은 전부 count error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import BubbleEvent, Correction, SequenceViolation


@dataclass
class SequenceOutcome:
    violations: list[SequenceViolation]
    sets_completed: int
    residual: dict[int, int]       # 미완성 사이클 잔여 (point → 개수)
    absorbed: int                  # 위반으로 흡수된 이벤트 수


@dataclass
class GateOutcome:
    """순서 게이트 결과: 수락 이벤트 + count error(미계수) + set."""

    accepted: list[BubbleEvent]
    count_errors: list[SequenceViolation]
    sets_completed: int
    residual: dict[int, int]


def gate_analyze(events: "list[BubbleEvent]", way: int,
                 tolerance_frames: int = 0) -> GateOutcome:
    """순서 게이트: expected만 수락, 그 외 count error·expected 유지 (D-21).

    way>=1, Point 번호 1..way 가정. tolerance_frames>0이면 창 내 expected를
    먼저 소비(재정렬)한 뒤 현재 이벤트를 재평가 — 구 analyze와 동일 창 의미.
    """
    if way < 1:
        raise ValueError(f"way는 1 이상이어야 합니다: {way}")
    order = sorted(events, key=lambda e: (e.frame, e.subframe, e.point))
    n = len(order)
    used = [False] * n
    accepted: list[BubbleEvent] = []
    count_errors: list[SequenceViolation] = []
    sets_completed = 0
    cycle: list[int] = []
    expected = 1
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        e = order[i]
        actual = e.point
        if actual == expected:
            accepted.append(e)
            used[i] = True
            cycle.append(actual)
            expected = expected + 1 if expected < way else 1
            if len(cycle) == way:
                sets_completed += 1
                cycle = []
            i += 1
            continue
        pulled = -1
        if tolerance_frames > 0:
            j = i
            while j < n and order[j].frame - e.frame <= tolerance_frames:
                if not used[j] and order[j].point == expected:
                    pulled = j
                    break
                j += 1
        if pulled >= 0:
            accepted.append(order[pulled])
            used[pulled] = True
            cycle.append(expected)
            expected = expected + 1 if expected < way else 1
            if len(cycle) == way:
                sets_completed += 1
                cycle = []
            continue
        # count error: 미계수, expected 불변
        count_errors.append(SequenceViolation(
            frame=e.frame, expected_point=expected, actual_point=actual))
        used[i] = True
        i += 1
    residual: dict[int, int] = {}
    for p in cycle:
        residual[p] = residual.get(p, 0) + 1
    return GateOutcome(accepted=accepted, count_errors=count_errors,
                       sets_completed=sets_completed, residual=residual)


def analyze(events: "list[BubbleEvent]", way: int, tolerance_frames: int = 0) -> SequenceOutcome:
    order = sorted(events, key=lambda e: (e.frame, e.subframe, e.point))
    n = len(order)
    used = [False] * n
    violations: list[SequenceViolation] = []
    sets_completed = 0
    absorbed = 0
    cycle: list[int] = []          # 현재 사이클의 청정 전진 point들
    expected = 1
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        e = order[i]
        actual = e.point
        if actual == expected:
            cycle.append(actual)
            used[i] = True
            expected = expected + 1 if expected < way else 1
            if len(cycle) == way:
                sets_completed += 1
                cycle = []
            i += 1
            continue
        # tolerance: 창 내 미사용 expected 이벤트를 먼저 소비 (재정렬)
        pulled = -1
        if tolerance_frames > 0:
            j = i
            while j < n and order[j].frame - e.frame <= tolerance_frames:
                if not used[j] and order[j].point == expected:
                    pulled = j
                    break
                j += 1
        if pulled >= 0:
            cycle.append(expected)
            used[pulled] = True
            expected = expected + 1 if expected < way else 1
            if len(cycle) == way:
                sets_completed += 1
                cycle = []
            continue               # i 전진 안 함 — 새 expected로 order[i] 재평가
        # 위반: 끊긴 런(cycle)은 absorbed로 흡수, 위반 이벤트 자신은 새 런의 첫 원소로 재동기
        violations.append(SequenceViolation(
            frame=e.frame, expected_point=expected, actual_point=actual))
        absorbed += len(cycle)
        cycle = [actual]
        used[i] = True
        expected = actual + 1 if actual < way else 1
        i += 1
    residual: dict[int, int] = {}
    for p in cycle:
        residual[p] = residual.get(p, 0) + 1
    return SequenceOutcome(violations=violations, sets_completed=sets_completed,
                            residual=residual, absorbed=absorbed)


def apply_corrections(events: "list[BubbleEvent]", corrections: "list[Correction]",
                       way: int, tolerance_frames: int = 0) -> SequenceOutcome:
    """+1 합성 이벤트 삽입 / −1 최근접 제거 후 automaton 재실행.  원본 events 불변.

    −1 최근접 제거는 프레임 거리로 정렬하고, 거리 동률이면 frame이 작은 쪽,
    그마저 같으면 subframe이 작은 쪽을 제거한다(입력 리스트 순서에 의존하지
    않는 결정론적 규칙 — 리뷰 D-5).
    """
    work = _apply_correction_edits(events, corrections)
    return analyze(work, way, tolerance_frames)


def apply_corrections_gated(events: "list[BubbleEvent]", corrections: "list[Correction]",
                            way: int, tolerance_frames: int = 0) -> GateOutcome:
    """보정 반영 후 순서 게이트 재실행 (order_gate=True 경로)."""
    work = _apply_correction_edits(events, corrections)
    return gate_analyze(work, way, tolerance_frames)


def _apply_correction_edits(events: "list[BubbleEvent]",
                            corrections: "list[Correction]") -> list[BubbleEvent]:
    work = list(events)
    for c in corrections:
        if c.delta == 1:
            work.append(BubbleEvent(point=c.point, frame=c.frame, subframe=0.0,
                                     flags=("manual",)))
        elif c.delta == -1:
            cands = [k for k, e in enumerate(work) if e.point == c.point]
            if cands:
                j = min(cands, key=lambda k: (abs(work[k].frame - c.frame),
                                               work[k].frame, work[k].subframe))
                work.pop(j)
    return work
