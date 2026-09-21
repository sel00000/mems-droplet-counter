"""멀티웨이 출력 파일 — 챕터 폴더·페이즈 폴더 전부 (설계 §1.5).

frame이 1차 좌표, time_s = frame ÷ RecordingSpec.fps 파생 표기(컨테이너 fps
불사용). JSON은 ensure_ascii=False·allow_nan=False. suspects/ 스냅샷 PNG는
엔진이 생성하고 여기서는 상대경로만 참조한다. count_error.png만 이 모듈이
OpenCV로 직접 그린다(imgio 경유 저장).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..imgio import imwrite_unicode
from .model import ChapterResult, MultiwayConfig, PhaseResult, ForeignObject
from . import profiles

from bubble_counter import __version__

FILTERED_OUT_ROW_CAP = 1000


def build_quality(chapter: ChapterResult, config: MultiwayConfig) -> dict:
    """quality.py가 없을 때 fallback: 빈 quality 블록 반환 (Task 5 완료 시 교체)."""
    return {
        "sibling_agreement": None,
        "set_conversion": None,
        "split_pair_candidates": 0,
        "per_point": {},
        "suspects_total": len(chapter.suspects),
        "suspects_truncated": chapter.suspects_truncated,
        "filtered_total": len(chapter.filtered_out),
        "filtered_by_reason": {},
        "audit_seed": 20260711,
    }


def time_s(frame: int, fps: float) -> float:
    """프레임 번호를 초로 환산(파생 표기). RecordingSpec.fps만 사용."""
    return frame / fps


def _write_rows(path, header: list[str], rows: Iterable[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_point_counts_csv(path, chapter: ChapterResult, config: MultiwayConfig) -> None:
    final = chapter.counts_final()
    auto = chapter.counts_auto
    rows = []
    for p in sorted(config.points, key=lambda p: p.number):
        n = p.number
        a = auto.get(n, 0)
        fnl = final.get(n, 0)
        rows.append([n, a, fnl - a, fnl, p.target, fnl - p.target])
    _write_rows(path, ["point", "count_auto", "corrections", "count_final", "target", "error"], rows)


def write_events_csv(path, chapter: ChapterResult) -> None:
    fps = chapter.recording.fps
    events = sorted(chapter.events, key=lambda e: (e.frame, e.subframe, e.point))
    rows = [
        [e.point, e.frame, f"{e.subframe:.4f}", f"{time_s(e.frame, fps):.3f}",
         "|".join(e.flags), f"{e.position_px:.1f}", f"{e.size_px:.1f}"]
        for e in events
    ]
    _write_rows(path, ["point", "frame", "subframe", "time_s", "flags",
                       "position_px", "size_px"], rows)


def write_violations_csv(path, chapter: ChapterResult) -> None:
    """순서 위반 / count error (order_gate ON 시 미계수 사건) CSV."""
    fps = chapter.recording.fps
    rows = [
        [v.frame, f"{time_s(v.frame, fps):.3f}", v.expected_point, v.actual_point]
        for v in sorted(chapter.violations, key=lambda v: v.frame)
    ]
    _write_rows(path, ["frame", "time_s", "expected_point", "actual_point"], rows)


def write_count_errors_csv(path, chapter: ChapterResult) -> None:
    """D-21: count error = 순서 불일치 미계수 (violations와 동일 행, 파일명 분리)."""
    write_violations_csv(path, chapter)


def write_suspects_csv(path, chapter: ChapterResult) -> None:
    fps = chapter.recording.fps
    rows = [
        [s.kind, s.point, s.frame, f"{time_s(s.frame, fps):.3f}", s.detail,
         s.snapshot or "", s.confidence or ""]
        for s in sorted(chapter.suspects, key=lambda s: (s.frame, s.point))
    ]
    _write_rows(path, ["kind", "point", "frame", "time_s", "detail", "snapshot", "confidence"],
                rows)


def write_filtered_out_csv(path, filtered_out: "Iterable[tuple[int, int, float, str]]", *,
                           cap: int = FILTERED_OUT_ROW_CAP) -> None:
    """크기 미달·중복·경계 배제 blob 감사 로그(ch.filtered_out = (point,frame,size_px,reason) 튜플).

    행 수 상한 + 항상 요약행(조용한 잘림 금지).
    """
    blobs = sorted(filtered_out, key=lambda b: (b[1], b[0]))   # (frame, point)
    total = len(blobs)
    rows = [[b[0], b[1], f"{b[2]:.1f}", b[3]] for b in blobs[:cap]]
    if total > cap:
        rows.append([f"# 총 {total}건 중 {cap}건만 기록 (감사 로그 상한)", "", "", ""])
    else:
        rows.append([f"# 총 {total}건 기록 (감사 로그)", "", "", ""])
    _write_rows(path, ["point", "frame", "size_px", "reason"], rows)


def _write_json(path, obj) -> None:
    """JSON을 원자적으로 쓴다(gui_settings.save_settings와 동일 관례).

    같은 폴더의 임시 파일에 먼저 쓰고 os.replace로 교체 — allow_nan=False가
    직렬화 도중 예외를 던져도(비유한값 등) 기존에 유효했던 대상 파일은 잘린
    채로 손상되지 않는다. 임시 파일은 실패 시 정리하고 원래 예외를 재raise한다.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def build_multiway_summary(chapter: ChapterResult, config: MultiwayConfig) -> dict:
    r = chapter.recording
    final = chapter.counts_final()
    auto = chapter.counts_auto
    points = []
    for p in sorted(config.points, key=lambda p: p.number):
        n = p.number
        fnl = final.get(n, 0)
        points.append({"number": n, "target": p.target, "count_auto": auto.get(n, 0),
                       "count_final": fnl, "error": fnl - p.target})
    targets_ok = all(final.get(p.number, 0) == p.target for p in config.points)
    summary = {
        "video_path": chapter.video_path,
        "tool_version": __version__,
        "way": config.way,
        "recording": {"width": r.width, "height": r.height, "fps": r.fps, "mode": r.mode},
        "points": points,
        "sets_completed": chapter.sets_completed,
        "violations": len(chapter.violations),
        "count_errors": len(chapter.violations),  # D-21: order_gate 시 미계수 사건 수(=violations)
        "order_gate": bool(getattr(config, "order_gate", True)),
        "perfect": bool(targets_ok and len(chapter.violations) == 0),
        "suspects": len(chapter.suspects),
        "suspects_high": sum(1 for s in chapter.suspects if s.confidence == "높음"),  # D-15
        "corrections": [dataclasses.asdict(c) for c in chapter.corrections],
        "stabilization": dataclasses.asdict(chapter.stabilization),  # D-16: 평행이동만(회전·줌 제외)
        "processed_frames": chapter.processed_frames,
        "cancelled": chapter.cancelled,
        "suspects_truncated": chapter.suspects_truncated,
        "self_check_messages": chapter.self_check_messages,
        "config": profiles.config_to_profile(config),
        "time_base": f"frame ÷ {r.fps:g}fps (RecordingSpec 기준, 컨테이너 fps 불사용)",
    }
    # quality 블록: quality.py가 구현되면 Task 5에서 채워짐 — 여기서 import 실패 대비
    try:
        from . import quality
        summary["quality"] = quality.build_quality(chapter, config)
    except ImportError:
        pass
    return summary


def write_multiway_summary_json(path, chapter: ChapterResult, config: MultiwayConfig) -> None:
    _write_json(path, build_multiway_summary(chapter, config))


def write_foreign_objects_json(path, objects: "list[ForeignObject]", *, source: str,
                               scanned_at: str, sample_frames: int,
                               band_margin_px: float) -> None:
    """정지 이물 목록 → foreign_objects.json (R0 §2.3). near_points는 list로 직렬화."""
    from .model import ForeignObject
    import dataclasses
    payload = {
        "tool_version": __version__,
        "scan": {"source": source, "scanned_at": scanned_at,
                 "sample_frames": sample_frames, "band_margin_px": band_margin_px},
        "objects": [{**dataclasses.asdict(fo), "near_points": list(fo.near_points)}
                    for fo in objects],
    }
    _write_json(path, payload)


def chapter_dir_name(video_path, revision: int = 1) -> str:
    """revision 1 → 기존 이름, n≥2 → …_멀티웨이_r{n} (R0 §2.4, D14)."""
    base = f"{Path(video_path).stem}_기포계수_멀티웨이"
    return base if revision <= 1 else f"{base}_r{revision}"


def next_revision(out_root, video_path) -> int:
    """out_root에서 기존 회차 폴더 스캔 → 다음 회차 번호(기존 없으면 1)."""
    root = Path(out_root)
    base = f"{Path(video_path).stem}_기포계수_멀티웨이"
    if not root.exists():
        return 1
    highest = 1 if (root / base).is_dir() else 0
    import re
    pat = re.compile(re.escape(base) + r"_r(\d+)$")
    for child in root.iterdir():
        m = pat.match(child.name)
        if m and child.is_dir():
            highest = max(highest, int(m.group(1)))
    return highest + 1


# 설계 §2.5 고정 팔레트(validate_palette 4검사 PASS — hex 변경 금지).
# zero(정확 달성) 색은 D-4 결정으로 B트랙 GUI 그래프 플랜과 표현을 통일.
_PALETTES = {
    "light": {"bg": "#ffffff", "pos": "#e34948", "neg": "#2a78d6", "zero": "#8a8f98",
              "axis": "#333333", "text": "#1a1a1a"},
    "dark":  {"bg": "#181a1f", "pos": "#e66767", "neg": "#3987e5", "zero": "#9aa0aa",
              "axis": "#c8c8c8", "text": "#e6e6e6"},
}


def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _error_label(e: int) -> str:
    """값 라벨 문자열 — 0은 무부호 "0"(중립), 그 외엔 부호 포함 "{e:+d}"(D-4)."""
    return "0" if e == 0 else f"{e:+d}"


def _label_color(e: int, pal: dict) -> tuple[int, int, int]:
    """값 라벨 색 — 0(정확 달성)은 zero 색, 그 외엔 text 색(D-4).

    라벨은 LINE_AA로 그려져 픽셀 정확 일치 검증이 신뢰할 수 없어(실측:
    "0" 글자 렌더 시 zero 색 정확 일치 픽셀 0개) 색 선택 로직을 분리해
    렌더링 없이 단위검증한다.
    """
    return pal["zero"] if e == 0 else pal["text"]


def render_count_error_png(path, errors: dict[int, int], *, theme: str = "light",
                           title: "str | None" = None) -> bool:
    """count error 다이버징 막대 그래프를 OpenCV로 그려 imgio로 저장.

    막대는 안티앨리어싱 없이 단색 채움(색 판정·테스트가 정확하도록), 텍스트만
    LINE_AA. 값 라벨(f"{e:+d}", 단 0은 무부호 "0")을 막대 끝에 항상 병기한다
    (색에만 의존 금지). 0(정확 달성)은 라벨을 zero 색으로 구분 표시한다(D-4).
    """
    if theme not in _PALETTES:
        raise ValueError(f"theme는 'light' 또는 'dark'여야 합니다: {theme!r}")
    pal = {k: _hex_to_bgr(v) for k, v in _PALETTES[theme].items()}
    numbers = sorted(errors)
    n = len(numbers)

    col_w = 90
    left, right, top, bottom = 64, 30, 54, 56
    plot_h = 300
    width = left + right + max(1, n) * col_w
    height = top + plot_h + bottom
    baseline_y = top + plot_h // 2

    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = pal["bg"]
    cv2.line(img, (left, baseline_y), (width - right, baseline_y), pal["axis"], 2)  # 0 기준선

    max_abs = max(1, max((abs(errors[k]) for k in numbers), default=1))
    unit = (plot_h // 2 - 24) / max_abs
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, k in enumerate(numbers):
        e = errors[k]
        cx = left + i * col_w + col_w // 2
        bar_w = int(col_w * 0.5)
        y_end = int(round(baseline_y - e * unit))
        color = pal["pos"] if e > 0 else pal["neg"] if e < 0 else pal["axis"]
        if e != 0:
            cv2.rectangle(img, (cx - bar_w // 2, min(baseline_y, y_end)),
                          (cx + bar_w // 2, max(baseline_y, y_end)), color, -1)
        label = _error_label(e)
        label_color = _label_color(e, pal)
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ly = y_end - 6 if e >= 0 else y_end + th + 6
        cv2.putText(img, label, (cx - tw // 2, ly), font, 0.5, label_color, 1, cv2.LINE_AA)
        plabel = f"Point {k}"
        (pw, _ph), _ = cv2.getTextSize(plabel, font, 0.44, 1)
        cv2.putText(img, plabel, (cx - pw // 2, height - 20), font, 0.44,
                    pal["text"], 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (left, 32), font, 0.6, pal["text"], 1, cv2.LINE_AA)
    return imwrite_unicode(path, img)


def phase_dir_name(phase_name: str) -> str:
    return f"{phase_name}_페이즈결과"


def build_phase_summary(phase: PhaseResult, config: MultiwayConfig) -> dict:
    n_ch = len(phase.chapters)
    sum_final = phase.sum_counts_final()
    points = []
    for p in sorted(config.points, key=lambda p: p.number):
        tp = p.target * n_ch
        fnl = sum_final.get(p.number, 0)
        points.append({"number": p.number, "target_phase": tp,
                       "count_final": fnl, "error": fnl - tp})
    return {
        "name": phase.name,
        "chapters": [
            {"video_path": ch.video_path, "sets_completed": ch.sets_completed,
             "violations": len(ch.violations), "suspects": len(ch.suspects),
             "processed_frames": ch.processed_frames}
            for ch in phase.chapters
        ],
        "points": points,
        "sum_sets": phase.sum_sets(),
    }


def write_chapter_outputs(out_dir, chapter: ChapterResult, config: MultiwayConfig, *,
                          theme: str = "light") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_multiway_summary_json(out / "multiway_summary.json", chapter, config)
    write_point_counts_csv(out / "point_counts.csv", chapter, config)
    write_events_csv(out / "events.csv", chapter)
    write_violations_csv(out / "violations.csv", chapter)
    write_count_errors_csv(out / "count_errors.csv", chapter)  # D-21 별칭
    write_suspects_csv(out / "suspects.csv", chapter)
    write_filtered_out_csv(out / "filtered_out.csv", chapter.filtered_out)
    errors = chapter.errors({p.number: p.target for p in config.points})
    # 파일명 count_error.png 유지, 의미=목표 오차(질문3-A)
    render_count_error_png(out / "count_error.png", errors, theme=theme,
                            title=f"{Path(chapter.video_path).stem} — 목표 오차")


def write_phase_outputs(out_dir, phase: PhaseResult, config: MultiwayConfig, *,
                        theme: str = "light") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "phase_summary.json", build_phase_summary(phase, config))
    n_ch = len(phase.chapters)
    sum_final = phase.sum_counts_final()
    errors = {p.number: sum_final.get(p.number, 0) - p.target * n_ch for p in config.points}
    render_count_error_png(out / "phase_count_error.png", errors, theme=theme,
                            title=f"{phase.name} — 페이즈 목표 오차")
