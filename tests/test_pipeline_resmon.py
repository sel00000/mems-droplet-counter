"""run_count 자원 모니터 통합 테스트.

resmon.make_reader를 pipeline 네임스페이스에서 monkeypatch로 가짜 리더로 바꿔
결정론적으로 검증한다(실측값은 비결정적). 검증 항목:
- summary["resources"] 블록 존재·형식, ProgressEvent에 자원 필드 전달.
- resmon_interval=0 이면 자원 필드 None·집계 0(하위호환, 시간 체크포인트 비활성).
- 진행 체크포인트의 시간 OR("progress_every프레임 OR resmon_interval초")가
  프레임 조건과 독립으로 발동.
"""
import json

from bubble_counter import pipeline
from bubble_counter.config import CounterConfig, LineSpec
from bubble_counter.resmon import Snapshot
from bubble_counter.synth import SynthConfig, generate


def _tiny_video(tmp_path):
    video = tmp_path / "v.avi"
    generate(SynthConfig(width=160, height=120, fps=30, duration_s=2,
                         rate_per_s=2, lead_in_s=0.5, seed=5), video)
    return video


def _fixed_reader():
    def read():
        return Snapshot(cpu_pct=25.0, cpu_cores=1.0, rss_mb=200.0, sys_ram_pct=40.0)
    return read


def test_run_count_writes_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "make_reader", _fixed_reader)  # pipeline 네임스페이스로 import됨
    video = _tiny_video(tmp_path)
    cfg = CounterConfig(line=LineSpec(axis="h", pos=0.5, band_px=8), warmup_frames=5)
    events = []
    res = pipeline.run_count(video, tmp_path / "out", cfg,
                             progress_cb=events.append, progress_every=1,
                             resmon_interval=0.5)
    assert "resources" in res.summary
    assert res.summary["resources"]["cpu_pct_peak"] == 25.0
    assert res.summary["resources"]["interval_s"] == 0.5   # 스펙 §6.3
    assert any(e.cpu_pct == 25.0 for e in events)     # ProgressEvent에 실림
    disk = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert disk["resources"]["samples"] >= 1


def test_resmon_interval_zero_disables(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "make_reader", _fixed_reader)
    video = _tiny_video(tmp_path)
    cfg = CounterConfig(line=LineSpec(axis="h", pos=0.5, band_px=8), warmup_frames=5)
    events = []
    res = pipeline.run_count(video, tmp_path / "out2", cfg,
                             progress_cb=events.append, progress_every=1,
                             resmon_interval=0.0)
    assert res.summary["resources"]["samples"] == 0
    assert all(e.cpu_pct is None for e in events)


def test_time_based_checkpoint_fires_independent_of_frames(tmp_path, monkeypatch):
    """시간 OR가 프레임 조건과 독립으로 발동하는지("200프레임 OR 0.5초"의 시간 축).

    progress_every를 전체 프레임 수(=60)보다 크게 두면 프레임 조건은 frame 0에서만
    1회 발동한다. 그럼에도 아주 짧은 resmon_interval을 주면 시간 OR가 이후 프레임들에서
    발동해 콜백이 여러 번 온다. 60프레임 디코드/MOG2 시간(수십 ms) ≫ 1ms 간격이라
    안정적으로 참(가짜 시계 불필요).
    """
    monkeypatch.setattr(pipeline, "make_reader", _fixed_reader)
    video = _tiny_video(tmp_path)  # 60 frames
    cfg = CounterConfig(line=LineSpec(axis="h", pos=0.5, band_px=8), warmup_frames=5)
    events = []
    pipeline.run_count(video, tmp_path / "out3", cfg,
                       progress_cb=events.append, progress_every=100000,
                       resmon_interval=0.001)
    assert len(events) >= 2  # 시간 OR가 없으면 frame 0에서 딱 1회뿐
    assert all(e.cpu_pct == 25.0 for e in events)  # 시간 OR 발동에도 샘플이 실림
