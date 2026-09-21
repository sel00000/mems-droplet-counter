"""GUI 기동 부트스트랩 — portable/windowed 배포용 실패 로그.

tkinter·gui 는 함수 안에서만 import 한다 (WSL 등 tk 없는 환경에서 이 모듈
단위 테스트 가능). frozen(PyInstaller) 이면 exe 옆 error.log, 개발 실행이면
홈 디렉터리 .bubble_counter_error.log (packaging/implementation-notes D-P1).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_STARTUP_TITLE = "기포 계수 툴 — 시작 실패"
_STARTUP_BODY = (
    "프로그램을 시작하지 못했습니다.\n\n"
    "같은 폴더의 error.log 파일을\n"
    "툴 담당자에게 보내 주세요."
)
_LOG_NAME = "error.log"
_DEV_LOG_NAME = ".bubble_counter_error.log"


def default_log_path() -> Path:
    """배포(frozen)는 exe 옆, 개발 실행은 홈 디렉터리."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / _LOG_NAME
    return Path.home() / _DEV_LOG_NAME


def write_exception_log(
    log_path: Path,
    exc: BaseException | None = None,
    *,
    where: str = "uncaught",
) -> None:
    """traceback 을 log_path 에 append. 쓰기 실패는 삼킨다."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if exc is not None:
            body = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        else:
            body = "".join(traceback.format_exception(*sys.exc_info()))
        block = f"--- {stamp} [{where}] ---\n{body}\n"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(block)
    except OSError:
        pass


def show_startup_error_dialog(log_path: Path | None = None) -> None:
    """가능하면 tk 메시지박스. tk 없거나 실패하면 조용히 반환."""
    body = _STARTUP_BODY
    if log_path is not None and not getattr(sys, "frozen", False):
        body = (
            "프로그램을 시작하지 못했습니다.\n\n"
            f"로그 파일:\n{log_path}\n\n"
            "툴 담당자에게 보내 주세요."
        )
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror(_STARTUP_TITLE, body)
        finally:
            root.destroy()
    except Exception:
        pass


def install_excepthook(log_path: Path | None = None) -> Path:
    """sys.excepthook (+ threading.excepthook) 설치. 반환 = 사용 로그 경로."""
    path = log_path if log_path is not None else default_log_path()

    def _hook(exc_type, exc, tb) -> None:
        try:
            write_exception_log(path, exc, where="sys.excepthook")
        finally:
            show_startup_error_dialog(path)
            try:
                sys.__excepthook__(exc_type, exc, tb)
            except Exception:
                pass

    sys.excepthook = _hook

    try:
        import threading

        if hasattr(threading, "excepthook"):

            def _thread_hook(args) -> None:  # type: ignore[no-untyped-def]
                write_exception_log(path, args.exc_value, where="threading.excepthook")
                # 백그라운드 스레드는 시작 실패 다이얼로그를 띄우지 않는다
                # (메인 스레드 tk 와 충돌 위험). 로그만.

            threading.excepthook = _thread_hook  # type: ignore[assignment]
    except Exception:
        pass

    return path


def run_gui() -> int:
    """훅 설치 후 gui.main. 기동 실패 시 로그+다이얼로그, return 1."""
    log_path = install_excepthook()
    try:
        from bubble_counter.gui import main as gui_main

        return int(gui_main())
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except BaseException as e:
        write_exception_log(log_path, e, where="run_gui")
        show_startup_error_dialog(log_path)
        return 1
