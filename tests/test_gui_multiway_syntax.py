"""gui_multiway.py 스모크 — 이 dev 머신엔 tkinter가 없어 import 불가.

test_gui_syntax.py와 같은 방식: 소스를 ast로 파싱하고, 계약 배선(4개 순수 모듈
사용, 진입점 함수)만 소스 문자열로 확인한다. 절대 import하지 않는다.
"""

import ast
from pathlib import Path

GUI = Path(__file__).resolve().parent.parent / "bubble_counter" / "gui_multiway.py"


def _src() -> str:
    return GUI.read_text(encoding="utf-8")


def _func_src(name: str) -> str:
    """이름이 name인 메서드/함수의 소스만 추출(가드가 실제 그 함수 안에 있는지 확인용)."""
    tree = ast.parse(_src(), filename=str(GUI))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_src(), node) or ""
    raise AssertionError(f"{name} 정의를 찾지 못함")


def test_parses_as_valid_python():
    ast.parse(_src(), filename=str(GUI))


def test_defines_entry_point_and_tab():
    src = _src()
    assert "def attach_multiway_tab(" in src
    assert "class MultiwayTab(" in src


def test_imports_all_four_logic_modules():
    src = _src()
    for name in ("gui_view", "gui_roi_edit", "gui_multiway_state", "gui_graph"):
        assert name in src, f"{name} 미사용"


def test_roi_editor_wires_display_and_frame_nav():
    src = _src()
    assert "ppm_p6_bytes" in src
    assert "format='PPM'" in src or 'format="PPM"' in src   # 설계 §3 표시 기법
    assert "parse_frame_input" in src                       # 프레임 1차 좌표(시간 슬라이더 없음)
    assert "bubble_dims_from_drag" in src                   # 기포 크기 재기
    assert "(KeyError, IOError)" in src                     # L-2: 프로파일 예외 계약(무음 실패 금지)


def test_video_open_gate_is_wired():
    # B-12 리뷰(review-B-12-verdict.md)가 적발한 갭: video_open_enabled()가 어디서도
    # 호출되지 않음. [영상 열기] 버튼 활성/비활성이 이 게이트에 연결돼 있어야 한다.
    src = _src()
    assert "video_open_enabled" in src


def test_preflight_and_run_wired():
    src = _src()
    assert "PreflightDisplay" in src          # 차단/경고 표시 모델
    assert "_cancel_event" in src             # 진행 취소(기존 패턴 재사용)
    assert "after(" in src                    # 큐 폴링(기존 스레딩 계약)
    assert "self.phase_name" in src           # M-1: run_fn 3번째 인자로 페이즈명 전달


def test_run_active_guard_wired():
    # rev-B14 Critical: 실행 중 "이전" 클릭 → _show_step()이 _run_bar 등을 destroy →
    # 폴링 무음 사망 → 워커 고아화 → 재진입 시 이중 워커 동시 쓰기 위험 수정.
    assert "_run_active" in _func_src("_on_back")           # ① 이전: 실행 중 이탈 차단
    assert "_run_active" in _func_src("_start_run")         # ② 재클릭: 이중 워커 방지
    assert "_run_active = False" in _func_src("_poll_run")  # ③ 완료/오류: 플래그 해제


def test_compose_config_guard_wired():
    # D-13: Windows 실기 트레이스백(기포 크기 미지정·ROI 프레임 이탈) → 친절 배너 가드.
    guard = _func_src("_composed_config_or_warn")
    assert "기포 크기 재기" in guard                       # ① 크기 미지정 안내 문자열
    assert "프레임 안으로 이동" in guard                    # ② ROI 프레임 이탈(ValueError) 안내 문자열
    # 3개 소비처(프리플라이트 실행·프로파일 저장·측정 시작)가 가드를 경유해야 함
    for fn in ("_run_preflight", "_save_profile", "_start_run"):
        assert "_composed_config_or_warn" in _func_src(fn), f"{fn} 가드 미경유"


def test_recount_uses_stored_run_config_not_live_compose():
    # D-13후속 Critical(review-D13-verdict.md): 재계수 lambda가 tab._compose_config()를
    # 무가드로 재조립하면 "◀ 이전 → ROI를 프레임 밖으로 이동 → 다음 ▶"으로 RESULTS에
    # 돌아온 뒤 보정을 누를 때 stale ROI로 ValueError가 콘솔에 덤프됐다. 재계수는 그
    # 실행을 만든 cfg를 그대로 재사용해야 하고(재조립 금지), _correct()는 실행 이력
    # 없음/ValueError 양쪽 모두 기존 배너 패턴으로 방어해야 한다.
    src = _src()
    assert 'tab_box["tab"]._compose_config()' not in src   # 무가드 재조립 금지
    # cfg 대입은 _start_run이 아니라 워커 __done__ 성공분기(큐 소비, 메인스레드)에서
    # 일어난다 — 실패/취소된 재실행이 직전 성공 결과의 보정 cfg를 오염하지 않도록.
    assert '_last_run_cfg' in _func_src("_poll_run")
    correct_src = _func_src("_correct")
    assert "_last_run_cfg" in correct_src                  # 보관본 부재 시 중단
    assert "except ValueError" in correct_src              # 심층방어
    assert "프레임 안으로 이동" in correct_src              # 기존 D-13 배너 문구 재사용


def test_roi_edit_handlers_invalidate_preflight():
    # D-13후속: ROI 추가/이동/회전/삭제 후 4단계 재통과 없이 전진 불가(상태 위생).
    assert "invalidate_preflight" in _func_src("_finish_draw")   # 추가
    assert "invalidate_preflight" in _func_src("_on_release")    # 이동/크기변경/회전
    assert "invalidate_preflight" in _func_src("_on_delete")     # 삭제
    assert "invalidate_preflight" in _func_src("_finish_bubble")  # 기포 크기 변경
    # 프로파일 교체도 points/bubble/recording을 통째로 바꾸는 배치 변경이라 동일 취급.
    assert "invalidate_preflight" in _func_src("_load_profile")


def test_results_render_graph_and_review():
    src = _src()
    assert "error_bar_geometry" in src        # count error 그래프(§2.5)
    assert "bar_color" in src
    assert "counts_final" in src              # 최종 개수(보정 반영)
    assert "자기진단" in src                   # 편차>5% 배너(§2.4/결정6)
    assert "self_check_messages" in src       # L-1: 엔진 산출 자기진단 메시지도 표시
    assert "suspects_truncated" in src        # L-1: 의심 잘림 경고(조용한 잘림 금지)
