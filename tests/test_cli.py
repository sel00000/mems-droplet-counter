"""CLI subcommand tests (synth / count / bench / error handling)."""

from __future__ import annotations

import json as _json
import sys
import types

import pytest

from bubble_counter import cli
from bubble_counter.cli import main
from bubble_counter.synth import SynthConfig, generate


@pytest.fixture(scope="module")
def small_video(tmp_path_factory):
    """A tiny valid video shared by the count/bench CLI tests."""
    out_dir = tmp_path_factory.mktemp("cli_video")
    video = out_dir / "v.avi"
    cfg = SynthConfig(
        width=320, height=240, fps=30, duration_s=4, rate_per_s=2,
        lead_in_s=1, seed=5,
    )
    generate(cfg, video)
    return video


def test_synth_subcommand_creates_video_and_gt(tmp_path):
    out = tmp_path / "s.avi"
    rc = main(["synth", str(out), "--duration", "3", "--rate", "1"])

    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "s.avi.gt.json").exists()


def test_count_subcommand_writes_summary(small_video, tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["count", str(small_video), "-o", str(out), "--band", "32"])

    assert rc == 0
    assert (out / "summary.json").exists()
    captured = capsys.readouterr()
    assert "처리 크기" in captured.err


def test_bench_subcommand_runs(small_video):
    rc = main(["bench", str(small_video), "--seconds", "2"])

    assert rc == 0


def test_bench_bad_roi_returns_nonzero_with_korean_error(small_video, capsys):
    rc = main(["bench", str(small_video), "--seconds", "1",
               "--roi", "0", "0", "1", "2"])

    assert rc != 0
    captured = capsys.readouterr()
    assert "설정 오류" in (captured.err + captured.out)


def test_count_missing_file_returns_nonzero_with_korean_error(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.avi"
    rc = main(["count", str(missing), "-o", str(tmp_path / "out")])

    assert rc != 0
    captured = capsys.readouterr()
    assert "찾을 수 없습니다" in (captured.err + captured.out)


def test_count_bad_bucket_fails_fast_with_korean_error(small_video, tmp_path, capsys):
    # --bucket 0은 이제 config 생성 시점에 거부된다(영상 전체 처리 전 fail-fast).
    # "설정 오류:" 접두는 late run_count 경로("오류:")가 아니라 조기 config 경로임을 증명.
    out = tmp_path / "out"
    rc = main(["count", str(small_video), "-o", str(out), "--bucket", "0"])

    assert rc != 0
    captured = capsys.readouterr()
    assert "설정 오류" in (captured.err + captured.out)
    assert not (out / "summary.json").exists()  # 처리 전 실패 → 산출물 없음


def test_bench_bad_fps_returns_nonzero_with_korean_error(small_video, capsys):
    rc = main(["bench", str(small_video), "--seconds", "1", "--fps", "inf"])

    assert rc != 0
    captured = capsys.readouterr()
    assert "설정 오류" in (captured.err + captured.out)


def test_bench_bad_seconds_returns_nonzero_with_korean_error(small_video, capsys):
    # --fps inf와 동일 버그 클래스(int(seconds*fps)): 조기 "설정 오류"로 거부.
    rc = main(["bench", str(small_video), "--seconds", "inf"])

    assert rc != 0
    captured = capsys.readouterr()
    assert "설정 오류" in (captured.err + captured.out)


def _point(number=1, cx=0.5, cy=0.5):
    return {"number": number, "cx": cx, "cy": cy, "length": 0.05, "width": 0.05,
            "angle_deg": 0.0, "direction": None, "target": 500}


def _points_file(tmp_path, pts=None):
    p = tmp_path / "points.json"
    p.write_text(_json.dumps(pts or [_point()]), encoding="utf-8")
    return p


def _video(tmp_path, name="6way_1_1.avi"):
    v = tmp_path / name
    v.write_bytes(b"x")     # 존재만 하면 됨(runner는 스텁)
    return v


def test_multiway_happy_path_builds_config_and_calls_runner(tmp_path, monkeypatch):
    v = _video(tmp_path)
    captured = {}
    monkeypatch.setattr(cli, "_run_multiway",
                        lambda cfg, vids, name, out: captured.update(cfg=cfg, vids=vids, name=name) or 0)
    rc = cli.main(["multiway", str(v), "--recording", "1280x768@300:dma",
                   "--points", str(_points_file(tmp_path)), "--bubble", "circle:50",
                   "--phase-name", "6way_1", "-o", str(tmp_path)])
    assert rc == 0
    assert captured["cfg"].way == 1 and captured["cfg"].recording.fps == 300.0
    assert captured["vids"] == [str(v)] and captured["name"] == "6way_1"


def _profile_cfg(count_half=False):
    from bubble_counter.multiway.model import (
        BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec)
    return MultiwayConfig(RecordingSpec(1280, 768, 300.0, "dma"),
                          (PointSpec(1, 0.5, 0.5, 0.05, 0.05, 0.0),),
                          BubbleSizeSpec("circle", 50.0, 50.0, count_half_bubbles=count_half))


def test_multiway_profile_path_no_bubble_needed(tmp_path, monkeypatch):
    v = _video(tmp_path)
    cfg = _profile_cfg()
    monkeypatch.setattr(cli, "_load_profile_config", lambda name: cfg if name == "칩A" else None)
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda c, *a: got.update(cfg=c) or 0)
    rc = cli.main(["multiway", str(v), "--profile", "칩A", "--phase-name", "p"])  # --bubble 불필요
    assert rc == 0 and got["cfg"] is cfg          # 프로파일의 bubble 그대로


def test_multiway_profile_bubble_override(tmp_path, monkeypatch):
    v = _video(tmp_path)
    monkeypatch.setattr(cli, "_load_profile_config", lambda name: _profile_cfg())
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda c, *a: got.update(b=c.bubble) or 0)
    cli.main(["multiway", str(v), "--profile", "칩A", "--bubble", "ellipse:60x45",
              "--phase-name", "p"])
    assert got["b"].major_px == 60.0 and got["b"].minor_px == 45.0   # override 적용


def test_multiway_profile_bubble_override_preserves_count_half_when_unspecified(tmp_path, monkeypatch):
    # 프로파일에 count_half_bubbles=True가 저장돼 있고 --bubble로 크기만 override할 때
    # --count-half-bubbles/--no-count-half-bubbles를 둘 다 지정하지 않으면 프로파일 값을 유지해야
    # 한다(인터뷰 Q3 확정 토글의 침묵 리셋 방지).
    v = _video(tmp_path)
    monkeypatch.setattr(cli, "_load_profile_config", lambda name: _profile_cfg(count_half=True))
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda c, *a: got.update(b=c.bubble) or 0)
    cli.main(["multiway", str(v), "--profile", "칩A", "--bubble", "ellipse:60x45",
              "--phase-name", "p"])
    assert got["b"].count_half_bubbles is True


def test_multiway_profile_bubble_override_explicit_no_count_half_wins(tmp_path, monkeypatch):
    # 프로파일이 True를 저장하고 있어도 --no-count-half-bubbles를 명시하면 그 값이 우선한다.
    v = _video(tmp_path)
    monkeypatch.setattr(cli, "_load_profile_config", lambda name: _profile_cfg(count_half=True))
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda c, *a: got.update(b=c.bubble) or 0)
    cli.main(["multiway", str(v), "--profile", "칩A", "--bubble", "ellipse:60x45",
              "--no-count-half-bubbles", "--phase-name", "p"])
    assert got["b"].count_half_bubbles is False


def test_multiway_direct_path_count_half_defaults_false(tmp_path, monkeypatch):
    # 직접 경로(--recording+--points)에서 토글 미지정 → False(기존 회귀 동작 유지).
    v = _video(tmp_path)
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda cfg, *a: got.update(b=cfg.bubble) or 0)
    cli.main(["multiway", str(v), "--recording", "1280x768@300:dma",
              "--points", str(_points_file(tmp_path)), "--bubble", "circle:50",
              "--phase-name", "p"])
    assert got["b"].count_half_bubbles is False


def test_multiway_ellipse_bubble(tmp_path, monkeypatch):
    v = _video(tmp_path)
    got = {}
    monkeypatch.setattr(cli, "_run_multiway", lambda cfg, *a: got.update(b=cfg.bubble) or 0)
    cli.main(["multiway", str(v), "--recording", "1280x768@300:dma",
              "--points", str(_points_file(tmp_path)), "--bubble", "ellipse:60x45",
              "--phase-name", "p"])
    assert got["b"].major_px == 60.0 and got["b"].minor_px == 45.0


def test_multiway_profile_and_recording_conflict_is_3(tmp_path):
    v = _video(tmp_path)
    rc = cli.main(["multiway", str(v), "--profile", "x", "--recording", "1280x768@300:dma",
                   "--bubble", "circle:50", "--phase-name", "p"])
    assert rc == 3


def test_multiway_neither_source_is_3(tmp_path):
    rc = cli.main(["multiway", str(_video(tmp_path)), "--bubble", "circle:50", "--phase-name", "p"])
    assert rc == 3


def test_multiway_missing_bubble_is_3(tmp_path):
    rc = cli.main(["multiway", str(_video(tmp_path)), "--recording", "1280x768@300:dma",
                   "--points", str(_points_file(tmp_path)), "--phase-name", "p"])
    assert rc == 3


def test_multiway_bad_recording_syntax_is_3(tmp_path):
    rc = cli.main(["multiway", str(_video(tmp_path)), "--recording", "1280-768-300",
                   "--points", str(_points_file(tmp_path)), "--bubble", "circle:50",
                   "--phase-name", "p"])
    assert rc == 3


def test_multiway_unknown_profile_is_3(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_profile_config", lambda name: None)
    rc = cli.main(["multiway", str(_video(tmp_path)), "--profile", "없음", "--phase-name", "p"])
    assert rc == 3


def test_multiway_number_gap_in_points_is_3(tmp_path):
    # {1, 3} → MultiwayConfig ValueError → 설정 모순 3
    pts = _points_file(tmp_path, [_point(1, cx=0.3), _point(3, cx=0.6)])
    rc = cli.main(["multiway", str(_video(tmp_path)), "--recording", "1280x768@300:dma",
                   "--points", str(pts), "--bubble", "circle:50", "--phase-name", "p"])
    assert rc == 3


def test_multiway_missing_video_is_3(tmp_path):
    rc = cli.main(["multiway", str(tmp_path / "none.avi"), "--recording", "1280x768@300:dma",
                   "--points", str(_points_file(tmp_path)), "--bubble", "circle:50",
                   "--phase-name", "p"])
    assert rc == 3


def test_multiway_corrupt_profile_type_error_is_3(tmp_path, monkeypatch):
    # 손상된 프로파일(스키마 불일치)은 profile_to_config에서 TypeError를 던진다
    # (실측: BubbleSizeSpec.__init__()에 알 수 없는 키워드 인자) — 설정 모순 3으로 처리해야
    # 트레이스백이 사용자에게 노출되지 않는다.
    def _raise(name):
        raise TypeError("BubbleSizeSpec.__init__() got an unexpected keyword argument 'x'")
    monkeypatch.setattr(cli, "_load_profile_config", _raise)
    rc = cli.main(["multiway", str(_video(tmp_path)), "--profile", "손상됨", "--phase-name", "p"])
    assert rc == 3


def test_run_multiway_maps_preflight_block_to_2(tmp_path, monkeypatch):
    # runner(트랙 I)를 가짜 모듈로 주입해 exit-2 매핑만 검증(라이브는 I1).
    fake = types.ModuleType("bubble_counter.multiway.runner")

    class PreflightBlocked(RuntimeError):
        pass

    def run_phase(*a, **k):
        raise PreflightBlocked("3번이 없습니다")

    fake.PreflightBlocked = PreflightBlocked
    fake.run_phase = run_phase
    monkeypatch.setitem(sys.modules, "bubble_counter.multiway.runner", fake)
    from bubble_counter.multiway.model import (
        BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec)
    cfg = MultiwayConfig(RecordingSpec(1280, 768, 300.0, "dma"),
                         (PointSpec(1, 0.5, 0.5, 0.05, 0.05, 0.0),),
                         BubbleSizeSpec("circle", 50.0, 50.0))
    assert cli._run_multiway(cfg, ["a.avi"], "p", str(tmp_path)) == 2


def test_run_multiway_maps_value_error_to_3(tmp_path, monkeypatch, capsys):
    # runner(트랙 I) 실행 중 설정 모순(ValueError)은 exit-3으로 매핑(I1 리뷰 후속).
    fake = types.ModuleType("bubble_counter.multiway.runner")

    class PreflightBlocked(RuntimeError):
        pass

    def run_phase(*a, **k):
        raise ValueError("배경 추정 샘플이 비어 있습니다")

    fake.PreflightBlocked = PreflightBlocked
    fake.run_phase = run_phase
    monkeypatch.setitem(sys.modules, "bubble_counter.multiway.runner", fake)
    from bubble_counter.multiway.model import (
        BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec)
    cfg = MultiwayConfig(RecordingSpec(1280, 768, 300.0, "dma"),
                         (PointSpec(1, 0.5, 0.5, 0.05, 0.05, 0.0),),
                         BubbleSizeSpec("circle", 50.0, 50.0))
    assert cli._run_multiway(cfg, ["a.avi"], "p", str(tmp_path)) == 3
    captured = capsys.readouterr()
    assert "설정 오류" in captured.err


def test_run_multiway_maps_io_error_to_3(tmp_path, monkeypatch, capsys):
    # runner(트랙 I) 실행 중 영상 접근 실패(IOError)도 exit-3으로 매핑(I1 리뷰 후속).
    fake = types.ModuleType("bubble_counter.multiway.runner")

    class PreflightBlocked(RuntimeError):
        pass

    def run_phase(*a, **k):
        raise IOError("영상을 열 수 없습니다: a.avi")

    fake.PreflightBlocked = PreflightBlocked
    fake.run_phase = run_phase
    monkeypatch.setitem(sys.modules, "bubble_counter.multiway.runner", fake)
    from bubble_counter.multiway.model import (
        BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec)
    cfg = MultiwayConfig(RecordingSpec(1280, 768, 300.0, "dma"),
                         (PointSpec(1, 0.5, 0.5, 0.05, 0.05, 0.0),),
                         BubbleSizeSpec("circle", 50.0, 50.0))
    assert cli._run_multiway(cfg, ["a.avi"], "p", str(tmp_path)) == 3
    captured = capsys.readouterr()
    assert "설정 오류" in captured.err
