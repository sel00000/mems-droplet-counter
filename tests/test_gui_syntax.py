"""Syntax-only smoke test for gui.py.

This dev machine has no tkinter, so gui.py can never be imported here (and
this test file must never import it, or tkinter). Instead we parse the
source as Python and check it mentions the pieces of the required threading
contract: cancel_event routed into run_count, and tk updates driven only via
root.after polling.
"""

import ast
from pathlib import Path

GUI_PATH = Path(__file__).resolve().parent.parent / "bubble_counter" / "gui.py"


def _read_source() -> str:
    return GUI_PATH.read_text(encoding="utf-8")


def test_gui_py_parses_as_valid_python():
    ast.parse(_read_source(), filename=str(GUI_PATH))


def test_gui_py_wires_run_count_cancel_event_and_after_polling():
    source = _read_source()
    assert "run_count" in source
    assert "cancel_event" in source
    assert "after(" in source


def test_gui_py_wires_multiway_notebook_tab():
    source = _read_source()
    assert "Notebook" in source
    assert "attach_multiway_tab" in source
    assert '"단일선"' in source or "'단일선'" in source
    assert "단일선 모드만" in source
