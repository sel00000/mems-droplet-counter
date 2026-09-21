"""PyInstaller 엔트리 — GUI only (CLI 서브커맨드 비노출)."""

from __future__ import annotations

import sys


def main() -> int:
    from bubble_counter.bootstrap import run_gui

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
