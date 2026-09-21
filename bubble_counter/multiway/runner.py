"""챕터 순차 실행 + 페이즈 합산 오케스트레이션 (CLI/GUI 공용 배선 지점)."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from ..imgio import imwrite_unicode
from .engine import PreflightReport, preflight, run_multiway_chapter
from .has_u2 import infer_mode
from .model import MultiwayConfig, PhaseResult
from .report import (chapter_dir_name, next_revision, phase_dir_name,
                     write_chapter_outputs, write_phase_outputs, write_foreign_objects_json)


class PreflightBlocked(RuntimeError):
    """프리플라이트 차단 항목 발생 — CLI 종료코드 2에 대응."""


def _snapshot_saver(out_dir: Path):
    def save(rel_path: str, img) -> None:
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(str(path), img)
    return save


def run_phase(videos, cfg: MultiwayConfig, phase_name: str, out_root,
              progress_cb=None, cancel_event=None, revision: int = 1,
              live_cb=None, live_control=None) -> PhaseResult:
    """revision=1(기본) → 기존 동작 동일. n≥2 → chapter_dir_name(video, revision)로 재분석 폴더 생성.

    live_cb/live_control: 관찰 전용(L4). 챕터마다 live_label을 '이름 (i/N)'으로 넘긴다.
    """
    out_root = Path(out_root)
    chapters = []
    n_vid = len(videos)
    for i, video in enumerate(videos):
        out_dir = out_root / chapter_dir_name(video, revision)
        pf = preflight(video, cfg, mode_checker=infer_mode)
        if pf.blocking:
            raise PreflightBlocked("; ".join(pf.blocking))
        out_dir.mkdir(parents=True, exist_ok=True)

        # 프리플라이트에서 이물 검출 시 foreign_objects.json 사전 저장 (R0 §2.3)
        if pf.foreign_objects:
            write_foreign_objects_json(
                out_dir / "foreign_objects.json",
                pf.foreign_objects,
                source="preflight",
                scanned_at=datetime.utcnow().isoformat(),
                sample_frames=60,
                band_margin_px=25.0
            )

        live_label = f"{Path(video).name} ({i + 1}/{n_vid})"
        ch = run_multiway_chapter(
            video, cfg, out_dir, progress_cb=progress_cb,
            cancel_event=cancel_event, snapshot_save_fn=_snapshot_saver(out_dir),
            live_cb=live_cb, live_control=live_control, live_label=live_label)
        write_chapter_outputs(out_dir, ch, cfg)
        chapters.append(ch)
        if cancel_event is not None and cancel_event.is_set():
            break
    phase = PhaseResult(name=phase_name, chapters=chapters)
    phase_dir = out_root / phase_dir_name(phase_name)
    phase_dir.mkdir(parents=True, exist_ok=True)
    write_phase_outputs(phase_dir, phase, cfg)
    return phase
