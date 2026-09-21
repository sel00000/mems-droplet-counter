"""검토 센터 신뢰 지표(quality 블록) 빌더 (R0 §2.2, D9·⑪). 순수 함수 — cv2 불요."""

from __future__ import annotations

import statistics

from .model import ChapterResult, MultiwayConfig


# P34 audit.DEFAULT_SEED와 동일 값을 의도적으로 복제한다(트랙 병렬 — audit import 금지, R0 §2.2).
# audit.py 변경 시 두 상수를 함께 맞춘다.
AUDIT_SEED = 20260711
_FLAG_KEYS = ("no-upstream", "weak-bubble", "merge-suspect", "boundary-size")
_REASON_KEYS = ("undersize", "ring-residual", "duplicate", "boundary-incomplete")
_SPLIT_FRAME_GAP = 3        # D13 초기값 — 12챕터 베이스라인 캘리브 후 확정(스펙 Deviations 재량)
_SPLIT_SIZE_RATIO = 0.9


def build_quality(chapter: ChapterResult, config: MultiwayConfig) -> dict:
    way = config.way
    counts_final = chapter.counts_final()
    vals = [counts_final.get(n, 0) for n in range(1, way + 1)]

    med = statistics.median(vals) if vals else 0
    sibling_agreement = (max(vals) - min(vals)) / med if (vals and med) else None
    denom = sum(vals)
    set_conversion = (chapter.sets_completed * way / denom) if denom else None

    per_point = {}
    for n in range(1, way + 1):
        fc = {k: 0 for k in _FLAG_KEYS}
        for e in chapter.events:
            if e.point == n:
                for f in e.flags:
                    if f in fc:
                        fc[f] += 1
        per_point[str(n)] = {"count_final": counts_final.get(n, 0), "flags": fc}

    by_reason = {k: 0 for k in _REASON_KEYS}
    for row in chapter.filtered_out:
        if row[3] in by_reason:                  # row = (point, frame, size_px, reason)
            by_reason[row[3]] += 1

    return {
        "sibling_agreement": sibling_agreement,
        "set_conversion": set_conversion,
        "split_pair_candidates": _count_split_pairs(chapter, config),
        "per_point": per_point,
        "suspects_total": len(chapter.suspects),
        "suspects_truncated": chapter.suspects_truncated,
        "filtered_total": len(chapter.filtered_out),
        "filtered_by_reason": by_reason,
        "audit_seed": AUDIT_SEED,
    }


def _count_split_pairs(chapter: ChapterResult, config: MultiwayConfig) -> int:
    """D13: 같은 프레임 근방(≤_SPLIT_FRAME_GAP) 서로 다른 Point의 소형 이벤트 쌍 수.
    프레임 오름차순 그리디 — 각 이벤트 최대 1쌍(결정론). 집계 전용(계수 무영향)."""
    if config.bubble is None:
        return 0
    thr = _SPLIT_SIZE_RATIO * min(config.bubble.major_px, config.bubble.minor_px)
    small = sorted((e for e in chapter.events if 0.0 <= e.size_px < thr),
                   key=lambda e: (e.frame, e.point))
    used = [False] * len(small)
    pairs = 0
    for i in range(len(small)):
        if used[i]:
            continue
        for j in range(i + 1, len(small)):
            if used[j]:
                continue
            if small[j].frame - small[i].frame > _SPLIT_FRAME_GAP:
                break                            # 정렬돼 있으므로 이후 j는 전부 간격 초과
            if small[j].point != small[i].point:
                used[i] = used[j] = True
                pairs += 1
                break
    return pairs