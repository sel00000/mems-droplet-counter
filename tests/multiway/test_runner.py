"""runner 배선 테스트 — 엔진/리포트를 가짜로 주입해 오케스트레이션만 검증."""

from pathlib import Path

import pytest

from bubble_counter.multiway import runner
from bubble_counter.multiway.report import next_revision
from bubble_counter.multiway.model import (
    BubbleSizeSpec, ChapterResult, MultiwayConfig, PointSpec, RecordingSpec,
)


def _cfg():
    return MultiwayConfig(
        recording=RecordingSpec(1280, 768, 300.0, "dma"),
        points=(PointSpec(1, 0.5, 0.5, 0.05, 0.1, 0.0),),
        bubble=BubbleSizeSpec("circle", 50.0, 50.0),
    )


def _fake_chapter(video_path, cfg, out_dir, **kw):
    return ChapterResult(
        video_path=str(video_path), recording=cfg.recording,
        counts_auto={1: 7}, corrections=[], events=[], violations=[],
        suspects=[], sets_completed=7, processed_frames=100,
    )


def test_run_phase_aggregates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_multiway_chapter",
                        lambda *a, **k: calls.append(a[0]) or _fake_chapter(*a, **k))
    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightReport(blocking=[], warnings=[], size_histogram={}))
    monkeypatch.setattr(runner, "chapter_dir_name", lambda video, revision=1: f"{Path(video).stem}_기포계수_멀티웨이")
    written = []
    monkeypatch.setattr(runner, "write_chapter_outputs", lambda out, ch, cfg: written.append(("ch", out)))
    monkeypatch.setattr(runner, "write_phase_outputs", lambda out, ph, cfg: written.append(("ph", out)))
    ph = runner.run_phase(["a.avi", "b.avi"], _cfg(), "6way_1", tmp_path)
    assert [c for c in calls] == ["a.avi", "b.avi"]          # 챕터 순서 = 파일 순서
    assert ph.sum_counts_final() == {1: 14}
    assert [w[0] for w in written] == ["ch", "ch", "ph"]


def test_run_phase_blocks_on_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "preflight",
                        lambda *a, **k: runner.PreflightReport(
                            blocking=["해상도가 설정과 다릅니다"], warnings=[], size_histogram={}))
    monkeypatch.setattr(runner, "chapter_dir_name", lambda video, revision=1: f"{Path(video).stem}_기포계수_멀티웨이")
    with pytest.raises(runner.PreflightBlocked, match="해상도"):
        runner.run_phase(["a.avi"], _cfg(), "p", tmp_path)
    # (참고: 번호 연속성은 MultiwayConfig 생성 시 ValueError로 차단됨(CLI 종료 3) —
    #  preflight 차단(종료 2)은 영상 대조 항목만. 교차 리뷰 L-3 반영)
