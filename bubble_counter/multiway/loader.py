"""결과 폴더 → 뷰모델 로더 (R0 §3.2, P2)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2

from .model import (
    BubbleEvent,
    ChapterResult,
    Correction,
    MultiwayConfig,
    PhaseResult,
    RecordingSpec,
    SequenceViolation,
    SuspectMoment,
    BubbleSizeSpec,
    PointSpec,
    ForeignObject,
    StabilizationStats,
)
from .profiles import profile_to_config
from bubble_counter import __version__


@dataclass
class ChapterView:
    """챕터 1개 분량의 검토용 뷰 데이터."""

    folder: Path
    summary: dict                       # 원본 multiway_summary.json
    events: list[BubbleEvent]           # 구버전: position_px/size_px = -1.0
    filtered: list[tuple[int, int, float, str]]  # 구버전: reason = "" (화면 "미기록")
    suspects: list[SuspectMoment]       # 구버전 6컬럼 CSV: confidence = ""
    corrections: list[Correction]       # corrections_log.csv (없으면 [])
    foreign: dict | None                # foreign_objects.json 원본 (없으면 None)
    audit: dict | None                  # audit.json 원본 (없으면 None)
    config: MultiwayConfig | None       # summary["config"] 복원 (구버전 None)
    video_path: Path | None             # 존재 확인된 원본 영상 (없으면 None = 스냅샷 전용 모드)
    quality: dict | None                # summary["quality"] (구버전 None)
    violations: list[SequenceViolation] = field(default_factory=list)  # count_errors.csv (없으면 violations.csv 폴백)
    counts_auto: dict[int, int] = field(default_factory=dict)          # point_counts.csv의 point→count_auto
    recording: RecordingSpec | None = None                            # summary["recording"] (없음/검증 실패 시 None)


def find_video(summary: dict, folder: Path) -> Path | None:
    """원본 영상 경로 탐색 (R0 §3.2)."""
    # 1) summary["video_path"] 존재 시 그 절대경로
    vp = summary.get("video_path")
    if vp and Path(vp).exists():
        return Path(vp)

    # 2) 폴더명 역산: "X_기포계수_멀티웨이[_rN]" → 부모/X.avi
    stem = folder.name
    for suffix in ("_기포계수_멀티웨이", "_기포계수_멀티웨이_r"):
        if stem.endswith(suffix) or any(stem.endswith(f"{suffix}{n}") for n in range(2, 10)):
            base = stem.rsplit("_기포계수_멀티웨이", 1)[0]
            cand = folder.parent / f"{base}.avi"
            if cand.exists():
                return cand
            # _rN 버전도 체크
            for n in range(2, 10):
                cand = folder.parent / f"{base}_r{n}.avi"
                if cand.exists():
                    return cand
                cand = folder.parent / f"{base}.avi"
                if cand.exists():
                    return cand

    # 3) 없음
    return None


def is_review_folder(folder: Path) -> bool:
    """진입 다이얼로그 검증용: multiway_summary.json 존재 여부."""
    return (folder / "multiway_summary.json").exists()


def _load_csv_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def _parse_events_csv(path: Path) -> list[BubbleEvent]:
    rows = _load_csv_rows(path)
    if not rows:
        return []
    header = rows[0]
    has_pos = "position_px" in header
    has_size = "size_px" in header
    events = []
    for row in rows[1:]:
        if len(row) < 5:
            continue
        point = int(row[0])
        frame = int(row[1])
        subframe = float(row[2])
        time_s = float(row[3])
        flags = tuple(f for f in row[4].split("|") if f) if row[4] else ()
        position_px = float(row[5]) if has_pos and len(row) > 5 else -1.0
        size_px = float(row[6]) if has_size and len(row) > 6 else -1.0
        events.append(BubbleEvent(
            point=point, frame=frame, subframe=subframe, flags=flags,
            position_px=position_px, size_px=size_px
        ))
    return events


def _parse_filtered_csv(path: Path) -> list[tuple[int, int, float, str]]:
    """구버전 3컬럼/신버전 4컬럼 모두 수용."""
    rows = _load_csv_rows(path)
    if not rows:
        return []
    header = rows[0]
    has_reason = "reason" in header
    filtered = []
    for row in rows[1:]:
        if not row or row[0].startswith("#"):
            continue
        if len(row) < 3:
            continue
        point = int(row[0])
        frame = int(row[1])
        size_px = float(row[2])
        reason = row[3] if has_reason and len(row) > 3 else ""
        filtered.append((point, frame, size_px, reason))
    return filtered


def _parse_suspects_csv(path: Path) -> list[SuspectMoment]:
    rows = _load_csv_rows(path)
    if not rows:
        return []
    header = rows[0]
    has_conf = "confidence" in header
    suspects = []
    for row in rows[1:]:
        if len(row) < 6:
            continue
        kind, point, frame, time_s, detail, snapshot = row[:6]
        confidence = row[6] if has_conf and len(row) > 6 else ""
        suspects.append(SuspectMoment(
            kind=kind, point=int(point), frame=int(frame),
            detail=detail, snapshot=snapshot or None, confidence=confidence
        ))
    return suspects


def _parse_corrections_csv(path: Path) -> list[Correction]:
    rows = _load_csv_rows(path)
    if not rows:
        return []
    corrections = []
    for row in rows[1:]:
        if len(row) < 5:
            continue
        corrected_at, point, frame, delta, reason = row[:5]
        evidence = row[5] if len(row) > 5 else ""
        corrections.append(Correction(
            point=int(point), frame=int(frame), delta=int(delta),
            reason=reason, corrected_at=corrected_at, evidence_path=evidence
        ))
    return corrections


def _parse_foreign_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_audit_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_violations_csv(path: Path) -> list[SequenceViolation]:
    """count_errors.csv/violations.csv(같은 write 함수의 별칭) → SequenceViolation 목록.

    헤더: frame,time_s,expected_point,actual_point (time_s는 파생값이라 버린다).
    헤더 기반 파싱 — 컬럼 위치는 하드코딩하지 않는다. 파일 없음/헤더 불일치 시 빈 목록.
    """
    rows = _load_csv_rows(path)
    if not rows:
        return []
    header = rows[0]
    try:
        i_frame = header.index("frame")
        i_expected = header.index("expected_point")
        i_actual = header.index("actual_point")
    except ValueError:
        return []
    violations = []
    for row in rows[1:]:
        if not row or row[0].startswith("#"):
            continue
        if len(row) <= max(i_frame, i_expected, i_actual):
            continue
        violations.append(SequenceViolation(
            frame=int(row[i_frame]),
            expected_point=int(row[i_expected]),
            actual_point=int(row[i_actual]),
        ))
    return violations


def _parse_point_counts_csv(path: Path) -> dict[int, int]:
    """point_counts.csv → {point: count_auto}.

    헤더 기반 파싱(point·count_auto 컬럼만 사용, 나머지 컬럼은 무시). 파일 없음/헤더 불일치 시 빈 dict.
    """
    rows = _load_csv_rows(path)
    if not rows:
        return {}
    header = rows[0]
    try:
        i_point = header.index("point")
        i_count = header.index("count_auto")
    except ValueError:
        return {}
    counts: dict[int, int] = {}
    for row in rows[1:]:
        if not row or row[0].startswith("#"):
            continue
        if len(row) <= max(i_point, i_count):
            continue
        counts[int(row[i_point])] = int(row[i_count])
    return counts


def _parse_recording(summary: dict) -> RecordingSpec | None:
    """summary["recording"]의 {width,height,fps,mode} → RecordingSpec.

    키 없음/타입 오류/검증 실패(RecordingSpec.__post_init__)는 삼키고 None을 반환한다.
    """
    rec = summary.get("recording")
    if not isinstance(rec, dict):
        return None
    try:
        return RecordingSpec(
            width=rec["width"], height=rec["height"],
            fps=rec["fps"], mode=rec["mode"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_chapter(folder: Path) -> ChapterView:
    """챕터 폴더 → ChapterView (R0 §3.2)."""
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"폴더가 없습니다: {folder}")

    # summary.json
    summary_path = folder / "multiway_summary.json"
    if not summary_path.exists():
        raise ValueError(f"multiway_summary.json이 없습니다: {folder}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # CSV들
    events = _parse_events_csv(folder / "events.csv")
    filtered = _parse_filtered_csv(folder / "filtered_out.csv")
    suspects = _parse_suspects_csv(folder / "suspects.csv")
    corrections = _parse_corrections_csv(folder / "corrections_log.csv")

    # JSON들
    foreign = _parse_foreign_json(folder / "foreign_objects.json")
    audit = _parse_audit_json(folder / "audit.json")

    # config 복원 (구버전 None)
    config = None
    if "config" in summary:
        config = profile_to_config(summary["config"])

    # quality (구버전 None)
    quality = summary.get("quality")

    # 원본 영상
    video_path = find_video(summary, folder)

    # count error / point counts / recording (검토 센터 count error 탭·counts_auto 소스)
    ce_path = folder / "count_errors.csv"
    violations = _parse_violations_csv(ce_path if ce_path.exists() else folder / "violations.csv")
    counts_auto = _parse_point_counts_csv(folder / "point_counts.csv")
    recording = _parse_recording(summary)

    return ChapterView(
        folder=folder,
        summary=summary,
        events=events,
        filtered=filtered,
        suspects=suspects,
        corrections=corrections,
        foreign=foreign,
        audit=audit,
        config=config,
        video_path=video_path,
        quality=quality,
        violations=violations,
        counts_auto=counts_auto,
        recording=recording,
    )


def load_phase(folder: Path) -> PhaseResult:
    """페이즈 폴더(챕터들 합) → PhaseResult."""
    chapters = []
    for child in sorted(folder.iterdir()):
        if child.is_dir() and is_review_folder(child):
            chapters.append(load_chapter(child))
    return PhaseResult(name=folder.name, chapters=chapters)