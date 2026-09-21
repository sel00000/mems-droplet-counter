"""bootstrap 단위 테스트 — tkinter 없이 로그 경로·기록·훅을 검증."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from bubble_counter import bootstrap


def test_default_log_path_dev_is_home(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert bootstrap.default_log_path() == Path.home() / ".bubble_counter_error.log"


def test_default_log_path_frozen_is_exe_parent(monkeypatch, tmp_path):
    exe = tmp_path / "dist" / "기포계수툴.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert bootstrap.default_log_path() == exe.parent / "error.log"


def test_write_exception_log_appends_traceback(tmp_path):
    log = tmp_path / "error.log"
    try:
        raise ValueError("boom-test")
    except ValueError as e:
        bootstrap.write_exception_log(log, e, where="unit")
    text = log.read_text(encoding="utf-8")
    assert "boom-test" in text
    assert "ValueError" in text
    assert "[unit]" in text


def test_write_exception_log_swallows_oserror(tmp_path):
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("x", encoding="utf-8")
    log = blocked / "error.log"
    bootstrap.write_exception_log(log, RuntimeError("x"), where="unit")


def test_install_excepthook_sets_sys_hook(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    log = tmp_path / "error.log"
    path = bootstrap.install_excepthook(log)
    assert path == log
    assert sys.excepthook is not sys.__excepthook__


def test_run_gui_returns_1_when_gui_main_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "default_log_path", lambda: tmp_path / "error.log")
    monkeypatch.setattr(bootstrap, "show_startup_error_dialog", lambda p=None: None)

    mod = types.ModuleType("bubble_counter.gui")
    mod.main = lambda: (_ for _ in ()).throw(RuntimeError("tk-fail"))
    monkeypatch.setitem(sys.modules, "bubble_counter.gui", mod)

    assert bootstrap.run_gui() == 1
    text = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "tk-fail" in text
    assert "run_gui" in text


def test_run_gui_returns_gui_main_code(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "default_log_path", lambda: tmp_path / "error.log")
    monkeypatch.setattr(bootstrap, "show_startup_error_dialog", lambda p=None: None)

    mod = types.ModuleType("bubble_counter.gui")
    mod.main = lambda: 0
    monkeypatch.setitem(sys.modules, "bubble_counter.gui", mod)

    assert bootstrap.run_gui() == 0


def test_run_gui_systemexit_none_is_0(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "default_log_path", lambda: tmp_path / "error.log")
    monkeypatch.setattr(bootstrap, "show_startup_error_dialog", lambda p=None: None)

    def _exit_none():
        raise SystemExit(None)

    mod = types.ModuleType("bubble_counter.gui")
    mod.main = _exit_none
    monkeypatch.setitem(sys.modules, "bubble_counter.gui", mod)

    assert bootstrap.run_gui() == 0
