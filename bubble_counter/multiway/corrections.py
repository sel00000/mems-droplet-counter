"""보정 이력 (원본 불변 + append + undo)."""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .model import Correction


LOG_FILE = "corrections_log.csv"
HEADER = ["corrected_at", "point", "frame", "delta", "reason", "evidence_path"]


def make_correction(point: int, frame: int, delta: int, reason: str,
                    evidence_path: str = "") -> Correction:
    """Correction 팩토리 (현재 시각 주입)."""
    return Correction(
        point=point,
        frame=frame,
        delta=delta,
        reason=reason,
        corrected_at=datetime.utcnow().isoformat(timespec="seconds"),
        evidence_path=evidence_path,
    )


def load_corrections(folder: Path) -> list[Correction]:
    path = folder / LOG_FILE
    if not path.exists():
        return []
    cs: list[Correction] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cs.append(Correction(
                point=int(row["point"]),
                frame=int(row["frame"]),
                delta=int(row["delta"]),
                reason=row["reason"],
                corrected_at=row["corrected_at"],
                evidence_path=row.get("evidence_path", ""),
            ))
    return cs


def append_correction(folder: Path, c: Correction) -> None:
    """단일 보정 append (헤더 없으면 생성)."""
    path = folder / LOG_FILE
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(HEADER)
        writer.writerow([c.corrected_at, c.point, c.frame, c.delta, c.reason, c.evidence_path])


def append_corrections(folder: Path, cs: Iterable[Correction]) -> None:
    """일괄 append (1-fsync)."""
    path = folder / LOG_FILE
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(HEADER)
        for c in cs:
            writer.writerow([c.corrected_at, c.point, c.frame, c.delta, c.reason, c.evidence_path])


def remove_correction(folder: Path, index: int) -> None:
    """undo — 해당 행 제거 후 전체 원자적 재작성 (mkstemp+os.replace)."""
    path = folder / LOG_FILE
    cs = load_corrections(folder)
    if not (0 <= index < len(cs)):
        raise IndexError("보정 인덱스 범위 밖")
    del cs[index]
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for c in cs:
                writer.writerow([c.corrected_at, c.point, c.frame, c.delta, c.reason, c.evidence_path])
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def counts_with(counts_auto: dict[int, int], cs: Iterable[Correction]) -> dict[int, int]:
    """자동 카운트 + 보정 델타 = 최종 카운트."""
    final = dict(counts_auto)
    for c in cs:
        final[c.point] = final.get(c.point, 0) + c.delta
    return final