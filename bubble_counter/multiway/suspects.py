"""의심 구간 판정 + 자기진단 (설계 결정 5·6).

merge/slow/gap-anomaly/undersize를 Point별 중앙값 대비 배수로 판정.
(역류 탐지는 폐기 — 엔진·여기서 생성하지 않음.) detail은 한국어, 시각은
frame ÷ fps(RecordingSpec 기준) 파생 표기.
"""

from __future__ import annotations

import statistics

from bubble_counter.rhythm import Mark
from .model import SuspectMoment
from .sequence import SequenceOutcome

SLOW_DURATION_RATIO = 3.0
GAP_RATIO = 3.0
CHAPTER_SUSPECT_CAP = 100


def _t(frame: int, fps: float) -> float:
    return frame / fps if fps > 0 else 0.0


def detect(marks_by_point, events, cfg, fps) -> "tuple[list[SuspectMoment], bool]":
    scored: list[tuple[float, SuspectMoment]] = []     # (심각도, moment)
    merge_suspect = cfg.bubble.merge_suspect_px() if cfg.bubble is not None else 0.0
    for point, marks in marks_by_point.items():
        if not marks:
            continue
        med_dur = statistics.median((m.last_frame - m.first_frame + 1) for m in marks)
        for m in marks:
            if merge_suspect > 0 and m.width_px > merge_suspect:
                sev = m.width_px / merge_suspect
                scored.append((sev, SuspectMoment(
                    kind="merge", point=point, frame=m.last_frame,
                    detail=f"겹침 의심: 폭 {m.width_px}px (기준 {merge_suspect:.0f}px 초과) "
                           f"@ {_t(m.last_frame, fps):.2f}s", snapshot=None)))
            dur = m.last_frame - m.first_frame + 1
            if med_dur > 0 and dur > med_dur * SLOW_DURATION_RATIO:
                sev = dur / med_dur
                scored.append((sev, SuspectMoment(
                    kind="slow", point=point, frame=m.last_frame,
                    detail=f"정체 의심: 밴드 체류 {dur}프레임(중앙값의 {sev:.1f}배) "
                           f"@ {_t(m.last_frame, fps):.2f}s", snapshot=None)))
        ordered = sorted(marks, key=lambda m: m.first_frame)
        gaps = [ordered[k].first_frame - ordered[k - 1].first_frame
                for k in range(1, len(ordered))]
        if gaps:
            med_gap = statistics.median(gaps)
            for k in range(1, len(ordered)):
                g = ordered[k].first_frame - ordered[k - 1].first_frame
                if med_gap > 0 and g > med_gap * GAP_RATIO:
                    sev = g / med_gap
                    scored.append((sev, SuspectMoment(
                        kind="gap-anomaly", point=point, frame=ordered[k].first_frame,
                        detail=f"도착 간격 이상: {g}프레임(중앙값의 {sev:.1f}배) — 누락 후보 "
                               f"@ {_t(ordered[k].first_frame, fps):.2f}s", snapshot=None)))
    for e in events:
        if "boundary-size" in e.flags:
            scored.append((1.0, SuspectMoment(
                kind="undersize", point=e.point, frame=e.frame,
                detail=f"경계 크기: 인정 하한 ±20% 구간 @ {_t(e.frame, fps):.2f}s",
                snapshot=None)))
    scored.sort(key=lambda t: t[0], reverse=True)
    truncated = len(scored) > CHAPTER_SUSPECT_CAP
    capped = [s for _, s in scored[:CHAPTER_SUSPECT_CAP]]
    return capped, truncated


def self_check(counts_final, outcome: SequenceOutcome, way: int) -> list[str]:
    msgs: list[str] = []
    vals = [counts_final.get(n, 0) for n in range(1, way + 1)]
    if vals and max(vals) > 0:
        hi, lo = max(vals), min(vals)
        if (hi - lo) / hi > 0.05:
            low_pt = min(range(1, way + 1), key=lambda n: counts_final.get(n, 0))
            msgs.append(
                f"Point {low_pt}이(가) 다른 Point 대비 낮습니다 — ROI 위치 확인 권장 "
                f"(최대 {hi} · 최소 {lo})")
    total = sum(counts_final.values())
    expected = outcome.sets_completed * way + sum(outcome.residual.values()) + outcome.absorbed
    if total != expected:
        msgs.append(
            f"내부 오류: 산술 정합 불일치 (개수합 {total} ≠ "
            f"set×way+잔여+흡수 {expected}) — 개발자에게 보고 요망")
    return msgs
