"""Unit tests for bubble_counter.resmon (stdlib-only resource monitor).

Fake read_fn + injected clocks + fixed parser inputs keep everything
deterministic, per the design spec's test seam strategy.
"""

import os

import pytest

from bubble_counter.resmon import Snapshot, ResourceSampler, format_line, make_reader
from bubble_counter import resmon as R


# --- Task 1: Snapshot · ResourceSampler · format_line -----------------------

def _seq_reader(snaps):
    it = iter(snaps)

    def read():
        return next(it)

    return read


def test_accumulator_peak_avg_samples():
    snaps = [
        Snapshot(cpu_pct=20.0, cpu_cores=0.8, rss_mb=100.0, sys_ram_pct=40.0),
        Snapshot(cpu_pct=26.0, cpu_cores=1.04, rss_mb=120.0, sys_ram_pct=42.0),
    ]
    s = ResourceSampler(_seq_reader(snaps))
    assert s.sample() == snaps[0]
    assert s.sample() == snaps[1]
    r = s.result()
    assert r["samples"] == 2
    assert r["cpu_pct_peak"] == 26.0
    assert abs(r["cpu_pct_avg"] - 23.0) < 1e-9
    assert r["proc_rss_mb_peak"] == 120.0
    assert r["cpu_pct_normalized"] is True


def test_read_fn_exception_is_swallowed():
    def boom():
        raise RuntimeError("measure fail")

    s = ResourceSampler(boom)
    assert s.sample() is None          # count는 안 죽는다
    assert s.result()["samples"] == 0


def test_none_metrics_skipped_in_stats():
    snaps = [Snapshot(cpu_pct=None, cpu_cores=None, rss_mb=None,
                      sys_ram_pct=None, gpu_pct=None)]
    s = ResourceSampler(_seq_reader(snaps))
    s.sample()
    r = s.result()
    assert r["samples"] == 1
    assert r["cpu_pct_peak"] is None   # 유효값 0개 → None
    assert r["gpu_available"] is False


def test_disabled_sampler_returns_none():
    s = ResourceSampler(_seq_reader([]), enabled=False)
    assert s.sample() is None
    assert s.result()["samples"] == 0


def test_format_line_values_and_na():
    snap = Snapshot(cpu_pct=25.0, cpu_cores=1.0, rss_mb=210.0,
                    sys_ram_pct=41.0, gpu_pct=None)
    line = format_line(snap, ncpu=4)
    assert "CPU 25% (~1.0/4코어)" in line
    assert "RAM 210MB (sys 41%)" in line
    assert "GPU 해당없음" in line
    assert format_line(None, ncpu=4) == ""


# --- Task 2: CPU·RAM 측정 리더 (make_reader) --------------------------------

def test_cpu_formula_normalized_and_cores(monkeypatch):
    monkeypatch.setattr(R.os, "cpu_count", lambda: 4)
    wall = [0.0]; cput = [0.0]
    read = make_reader(now=lambda: wall[0], cpu_time=lambda: cput[0],
                       gpu_backend=None)
    assert read().cpu_pct is None            # baseline
    wall[0] = 1.0; cput[0] = 1.0             # 1s wall, 1.0s cpu = 한 코어
    snap = read()
    assert abs(snap.cpu_cores - 1.0) < 1e-9
    assert abs(snap.cpu_pct - 25.0) < 1e-9   # 100%/4코어


def test_cpu_ncpu_none_guard(monkeypatch):
    monkeypatch.setattr(R.os, "cpu_count", lambda: None)
    wall = [0.0]; cput = [0.0]
    read = make_reader(now=lambda: wall[0], cpu_time=lambda: cput[0], gpu_backend=None)
    read()
    wall[0] = 1.0; cput[0] = 0.5
    snap = read()                            # ncpu→1, 정규화=raw
    assert abs(snap.cpu_pct - 50.0) < 1e-9


@pytest.mark.skipif(not hasattr(os, "sysconf"),
                    reason="리눅스 전용 파서 — Windows에는 os.sysconf가 없음")
def test_ram_linux_parsers(monkeypatch, tmp_path):
    statm = tmp_path / "statm"; statm.write_text("1000 2500 100 1 0 200 0")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000 kB\nMemAvailable: 2000 kB\n")
    monkeypatch.setattr(R, "_PROC_STATM", str(statm))
    monkeypatch.setattr(R, "_PROC_MEMINFO", str(meminfo))
    monkeypatch.setattr(R.os, "sysconf", lambda k: 4096)
    monkeypatch.setattr(R.sys, "platform", "linux")
    assert abs(R._rss_mb_linux() - (2500 * 4096 / 1e6)) < 1e-6
    assert abs(R._sys_ram_pct_linux() - 75.0) < 1e-9   # (8000-2000)/8000


# --- Task 3: GPU best-effort (nvidia 탐지 + throttled one-shot) --------------

def test_gpu_parse_and_throttle(monkeypatch):
    clock = [0.0]
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1

        class R:
            stdout = "37, 512\n"

        return R()

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    read = R._make_gpu_read("nvidia", now=lambda: clock[0])
    assert read() == 37.0
    assert read() == 37.0            # throttle: 재호출 없이 캐시
    assert calls["n"] == 1
    clock[0] = 3.0                   # >2s 경과 → 재호출
    read(); assert calls["n"] == 2


def test_gpu_detect_absent(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda x: None)
    assert R._detect_gpu_backend() is None


def test_gpu_read_failure_degrades(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(R.subprocess, "run", boom)
    read = R._make_gpu_read("nvidia", now=lambda: 100.0)
    assert read() is None
