"""gui_review.py 스모크 — 이 dev 머신엔 tkinter가 없어 import 불가.

tests/test_gui_multiway_syntax.py와 같은 방식: 소스를 ast로 파싱하고 계약 배선만
소스 문자열/AST로 확인한다. 절대 import하지 않는다.

count error 탭(2026-07-17 사용자 지시 "다시보기")의 배선 회귀 락 포함.
"""

import ast
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parent.parent.parent / "bubble_counter" / "multiway" / "gui_review.py"


def _src() -> str:
    return GUI.read_text(encoding="utf-8")


def _class_node(name: str) -> ast.ClassDef:
    tree = ast.parse(_src(), filename=str(GUI))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} 정의를 찾지 못함")


def _method_src(cls: str, name: str) -> str:
    for node in _class_node(cls).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_src(), node) or ""
    raise AssertionError(f"{cls}.{name} 정의를 찾지 못함")


def test_parses_as_valid_python():
    ast.parse(_src(), filename=str(GUI))


def test_no_duplicate_methods_in_review_center():
    """과거 회귀: _toggle_* 3종 중복 정의 + _on_suspect_double 자기재귀 오버라이드."""
    for cls in ("ReviewCenter", "_FrameViewer"):
        names = [n.name for n in _class_node(cls).body if isinstance(n, ast.FunctionDef)]
        dups = [m for m, c in Counter(names).items() if c > 1]
        assert not dups, f"{cls} 중복 메서드 정의: {dups}"


def test_count_error_tab_wired():
    src = _src()
    assert '"count error"' in src, "count error 탭 미등록"
    assert "def _build_count_errors_tab(" in src
    assert "def _refresh_count_errors_tab(" in src
    # 로드 시 채워지는지
    populate = _method_src("ReviewCenter", "_populate_all")
    assert "_refresh_count_errors_tab" in populate
    # 탭 전환 상태 동기화
    tab_changed = _method_src("ReviewCenter", "_on_tab_changed")
    assert "COUNT_ERRORS" in tab_changed


def test_count_error_viewer_uses_targets():
    """더블클릭 → 프레임 뷰어에 targets(순회 목록) 전달."""
    open_sel = _method_src("ReviewCenter", "_open_selected_count_error")
    assert "violation_targets" in open_sel
    assert "targets=" in open_sel
    build = _method_src("ReviewCenter", "_build_count_errors_tab")
    assert "_on_count_error_double" in build
    assert "_clip_selected_count_error" in build


def test_frame_viewer_supports_target_navigation():
    node = _class_node("_FrameViewer")
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    arg_names = [a.arg for a in init.args.args] + [a.arg for a in init.args.kwonlyargs]
    assert "targets" in arg_names, "_FrameViewer가 targets 인자를 받지 않음"
    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    assert "_jump" in methods and "_set_center" in methods


def test_make_clip_wires_clipgen_overlay():
    """RI 배선 ②: _make_clip이 ClipOverlay를 구성해 render_clip을 호출."""
    make_clip = _method_src("ReviewCenter", "_make_clip")
    assert "ClipOverlay" in make_clip
    assert "render_clip" in make_clip
    assert "clip_path" in make_clip
    assert "startfile" in make_clip  # Windows 재생 + 실패 시 경로 안내
    assert "highlight_point" in make_clip


def test_state_uses_existing_chapter_view_fields():
    """과거 회귀: 존재하지 않는 view.filtered_out 참조로 검토 센터가 열리다 죽음."""
    state_src = (GUI.parent / "gui_review_state.py").read_text(encoding="utf-8")
    assert "view.filtered_out" not in state_src, "ChapterView에 없는 filtered_out 참조"
    assert "def violations_sorted(" in state_src
    assert "def violation_targets(" in state_src


def test_open_review_center_returns_instance():
    tree = ast.parse(_src(), filename=str(GUI))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "open_review_center")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "open_review_center에 return이 없음 — 창이 생성되지 않음"


def test_datetime_import_present_if_used():
    """리뷰 I-1 회귀: datetime 사용부(state→gui_review 이동)에 임포트가 따라오지 않아
    이물 태깅 클릭 시 NameError. ast.parse는 런타임 NameError를 못 잡는다."""
    src = _src()
    if "datetime." in src:
        assert "from datetime import datetime" in src or "import datetime" in src


def test_relative_imports_resolve_to_real_modules():
    """리뷰 후속 회귀: `from .video_io import ...`처럼 존재하지 않는 형제 모듈 임포트
    (clipgen 구초안·gui_review_audit에서 실재했던 결함 클래스). tkinter 부재로
    import 스모크가 불가능하므로 AST로 상대 임포트 대상 파일 실재를 검사한다."""
    pkg = GUI.parent            # bubble_counter/multiway
    parent_pkg = pkg.parent     # bubble_counter
    files = ["gui_review.py", "gui_review_state.py", "gui_review_audit.py",
             "clipgen.py", "loader.py"]
    for name in files:
        path = pkg / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level and node.module):
                continue
            base = pkg if node.level == 1 else parent_pkg
            top = node.module.split(".")[0]
            assert (base / f"{top}.py").exists() or (base / top).is_dir(), \
                f"{name}: 'from {'.' * node.level}{node.module} import …' 대상 모듈이 없음"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff 미설치 환경(Windows 등)")
def test_no_undefined_names_ruff_f821():
    """일반화 락: AST 파싱·문자열 계약이 못 잡는 정의되지 않은 이름(F821)을 정적 검사.
    tkinter 부재로 import 스모크가 불가능한 GUI 파일 전용 안전망."""
    targets = [GUI, GUI.parent / "gui_review_state.py", GUI.parent / "gui_review_audit.py"]
    res = subprocess.run(
        ["ruff", "check", "--select", "F821", *[str(t) for t in targets if t.exists()]],
        capture_output=True, text=True)
    assert res.returncode == 0, f"정의되지 않은 이름 존재:\n{res.stdout}{res.stderr}"
