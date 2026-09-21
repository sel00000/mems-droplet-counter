"""CLI resource-monitor wiring: progress-line suffix + ``--resmon-interval``."""

from __future__ import annotations

from bubble_counter import cli
from bubble_counter.pipeline import ProgressEvent


def test_print_progress_appends_resources(capsys):
    ev = ProgressEvent(10, 100, 500.0, 3, 2.0,
                       cpu_pct=25.0, cpu_cores=1.0, rss_mb=210.0,
                       sys_ram_pct=41.0, gpu_pct=None)
    cli._print_progress(ev)
    err = capsys.readouterr().err
    assert "CPU 25% (~1.0/" in err and "RAM 210MB" in err and "GPU 해당없음" in err


def test_print_progress_without_resources(capsys):
    ev = ProgressEvent(10, 100, 500.0, 3, 2.0)   # 필드 None
    cli._print_progress(ev)
    err = capsys.readouterr().err
    assert "CPU" not in err                       # 접미 없음(하위호환)


def test_count_parser_has_resmon_interval():
    p = cli._build_parser()
    args = p.parse_args(["count", "v.mp4", "-o", "out", "--resmon-interval", "1.0"])
    assert args.resmon_interval == 1.0
