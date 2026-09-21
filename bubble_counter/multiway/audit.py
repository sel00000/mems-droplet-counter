"""무작위 표본 감사 (R0 §3.5, P34).

- 이벤트 표본: 전 이벤트에서 균등 추출 (이전 라운드 exclude 제외)
- 무이벤트 표본: Point 균등 순환 + 그 Point 이벤트와 EMPTY_MIN_GAP 이상 떨어진 프레임 균등 추출
- 표본 부족 시 있는 만큼 (개수 부족은 UI에 명시 — 조용한 축소 금지)
- seed 고정: DEFAULT_SEED + round_no (라운드 누적)
- verdict: "O" | "X" | "모름" — "모름"은 분모에서 제외
- rule-of-three: k=0 관측 시 95% 상한 = 3/n. k>0이면 상한 null + 점추정 k/n
- 라운드 누적: 새 라운드는 이전 라운드 표본 제외 후 같은 seed 유도 (seed + round_no 사용)
- 페이즈 풀링(챕터 합산)은 화면 표시만 (파일은 챕터별)
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from .model import AuditSample, BubbleEvent


DEFAULT_SEED = 20260711
EMPTY_MIN_GAP = 12               # "무이벤트 창" = 그 Point 이벤트로부터 ±12프레임 이상
EVENTS_PER_ROUND = 10            # 이벤트 표본 수
EMPTY_PER_ROUND = 10             # 무이벤트 표본 수


def snapshot_name(round_no: int, kind: str, i: int, point: int, frame: int) -> str:
    """snapshots/audit_r{round}_{e|n}{i:02d}_p{point}_f{frame:06d}.png"""
    return f"snapshots/audit_r{round_no}_{'e' if kind == 'event' else 'n'}{i:02d}_p{point}_f{frame:06d}.png"


def select_samples(
    events: list[BubbleEvent],
    processed_frames: int,
    way: int,
    *,
    seed: int = DEFAULT_SEED,
    round_no: int = 1,
    n_events: int = EVENTS_PER_ROUND,
    n_empty: int = EMPTY_PER_ROUND,
    exclude: Optional[frozenset[tuple[str, int, int]]] = None,
) -> list[AuditSample]:
    """결정론적 표본 선정 (seed 고정)."""
    if exclude is None:
        exclude = frozenset()

    rng = random.Random(seed + round_no)
    samples: list[AuditSample] = []

    # ── 이벤트 표본 ──
    available_events = [e for e in events if ("event", e.point, e.frame) not in exclude]

    if available_events and n_events > 0:
        # round_no를 seed에 반영해 round별 다른 표본이 나오도록 무작위 선택
        event_samples = rng.sample(available_events, min(n_events, len(available_events)))
        for e in event_samples:
            samples.append(AuditSample(
                kind="event", point=e.point, frame=e.frame,
                window=None,
            ))

    # ── 무이벤트 표본 ──
    # Point별로 이벤트 프레임들 모음
    events_by_point: dict[int, list[int]] = {}
    for e in events:
        events_by_point.setdefault(e.point, []).append(e.frame)
    for p in events_by_point:
        events_by_point[p].sort()

    empty_candidates: list[tuple[int, int]] = []  # (point, frame)
    for p in range(1, way + 1):
        ev_frames = events_by_point.get(p, [])
        if not ev_frames:
            for f in range(processed_frames):
                empty_candidates.append((p, f))
            continue
        first = ev_frames[0]
        if first > EMPTY_MIN_GAP:
            for f in range(EMPTY_MIN_GAP, first - EMPTY_MIN_GAP + 1):
                empty_candidates.append((p, f))
        for i in range(len(ev_frames) - 1):
            gap = ev_frames[i + 1] - ev_frames[i]
            if gap > 2 * EMPTY_MIN_GAP:
                for f in range(ev_frames[i] + EMPTY_MIN_GAP, ev_frames[i + 1] - EMPTY_MIN_GAP + 1):
                    empty_candidates.append((p, f))
        last = ev_frames[-1]
        if processed_frames - last > EMPTY_MIN_GAP:
            for f in range(last + EMPTY_MIN_GAP, processed_frames - EMPTY_MIN_GAP + 1):
                empty_candidates.append((p, f))

    # 제외된 empty 표본 제거
    available_empty = [(p, f) for p, f in empty_candidates
                       if ("empty", p, f) not in exclude]

    if available_empty and n_empty > 0:
        empty_samples = rng.sample(available_empty, min(n_empty, len(available_empty)))
        for p, f in empty_samples:
            samples.append(AuditSample(
                kind="empty", point=p, frame=f,
                window=(f - EMPTY_MIN_GAP, f + EMPTY_MIN_GAP),
            ))

    return samples


def load_audit(folder: Path) -> dict | None:
    """audit.json 로드 (없으면 None)."""
    path = Path(folder) / "audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_round(folder: Path, round_no: int, samples_with_verdicts: list[dict]) -> None:
    """audit.json 원자적 갱신 + estimates 재계산(전 라운드 누적)."""
    from .audit import estimates  # 순환 import 방지

    path = Path(folder) / "audit.json"
    data = {"tool_version": "0.1.0", "seed": DEFAULT_SEED, "rounds": []}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    # 라운드 추가
    data["rounds"].append({"round": round_no, "audited_at": datetime.utcnow().isoformat(),
                           "samples": samples_with_verdicts})

    # estimates 재계산 (전 라운드 누적)
    data["estimates"] = estimates(data["rounds"])

    # 원자적 쓰기
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def estimates(rounds: list[dict]) -> dict:
    """§2.3 estimates 산식 (모름 제외, rule-of-three)."""
    event_audited = event_wrong = 0
    empty_audited = empty_missed = 0

    for r in rounds:
        for s in r.get("samples", []):
            v = s.get("verdict", "")
            if v not in ("O", "X"):
                continue
            if s.get("kind") == "event":
                event_audited += 1
                if v == "X":
                    event_wrong += 1
            elif s.get("kind") == "empty":
                empty_audited += 1
                if v == "X":
                    empty_missed += 1

    def rate(k, n):
        if n == 0:
            return 0.0, 0.0  # 표본 없으면 0% + 상한 0 (UI에서 별도 처리)
        if k == 0:
            return 0.0, 3.0 / n
        return k / n, None

    o_rate, o_upper = rate(event_wrong, event_audited)
    m_rate, m_upper = rate(empty_missed, empty_audited)

    return {
        "event_audited": event_audited, "event_wrong": event_wrong,
        "overcount_rate": o_rate, "overcount_upper95": o_upper,
        "empty_audited": empty_audited, "empty_missed": empty_missed,
        "miss_rate": m_rate, "miss_upper95": m_upper,
    }


def load_audit(folder: Path) -> dict | None:
    path = Path(folder) / "audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))