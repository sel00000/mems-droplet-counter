"""검토 센터 상태기계(tkinter-free) — 실제 loader 산출물 위에서 스모크 + count error 순회 로직.

이 테스트의 존재 이유: gui_review_state가 ChapterView에 없는 속성을 참조해
검토 센터가 열리다 죽는 회귀(view.filtered_out AttributeError)를 실데이터 형태로 잡는다.
"""

import json
from pathlib import Path

import pytest

from bubble_counter.multiway.gui_review_state import FilterMode, ReviewState, Tab
from bubble_counter.multiway.loader import load_chapter


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def result_folder(tmp_path: Path) -> Path:
    """신버전 결과 폴더 최소 재현 (count_errors.csv 포함)."""
    folder = tmp_path / "샘플_기포계수_멀티웨이"
    folder.mkdir()
    summary = {
        "video_path": str(tmp_path / "없는영상.avi"),
        "way": 2,
        "recording": {"width": 640, "height": 480, "fps": 300.0, "mode": "dma"},
        "sets_completed": 2,
        "violations": 2,
        "count_errors": 2,
        "order_gate": True,
    }
    _write(folder / "multiway_summary.json", json.dumps(summary, ensure_ascii=False))
    _write(folder / "events.csv",
           "point,frame,subframe,time_s,flags,position_px,size_px\n"
           "1,10,0.5000,0.033,,12.0,50.0\n"
           "2,14,0.2500,0.047,weak-bubble,13.0,48.0\n")
    _write(folder / "filtered_out.csv",
           "point,frame,size_px,reason\n"
           "1,20,8.0,undersize\n"
           "# 총 1건\n")
    _write(folder / "suspects.csv",
           "kind,point,frame,time_s,detail,snapshot,confidence\n"
           "merge,1,30,0.100,겹침 의심,,낮음\n")
    # 의도적으로 프레임 역순 기록 — 정렬 검증
    _write(folder / "count_errors.csv",
           "frame,time_s,expected_point,actual_point\n"
           "3058,10.193,1,2\n"
           "3048,10.160,2,1\n")
    _write(folder / "point_counts.csv",
           "point,count_auto,corrections,count_final,target,error\n"
           "1,2,0,2,500,-498\n"
           "2,2,0,2,500,-498\n")
    return folder


def _state(folder: Path) -> ReviewState:
    view = load_chapter(folder)
    return ReviewState(view=view, config=view.config, folder=folder)


def test_tab_enum_has_count_errors():
    assert hasattr(Tab, "COUNT_ERRORS")


def test_state_smoke_on_loaded_view(result_folder):
    """검토 센터 로드 경로 전체 — 어떤 접근자도 AttributeError 없이 동작해야 한다."""
    st = _state(result_folder)
    assert len(st.events_filtered()) == 2
    assert len(st.filtered_filtered()) == 1      # 회귀: view.filtered_out 참조였음
    assert len(st.suspects_filtered()) == 1
    assert st.foreign_filtered() == []
    assert st.counts_preview() == {1: 2, 2: 2}   # counts_auto 기반 (point_counts.csv)


def test_event_filter_still_works(result_folder):
    st = _state(result_folder)
    st.event_filters.add(FilterMode.WEAK)
    only_weak = st.events_filtered()
    assert len(only_weak) == 1
    assert only_weak[0][1].point == 2


def test_violations_sorted_by_frame(result_folder):
    st = _state(result_folder)
    vs = st.violations_sorted()
    assert [v.frame for v in vs] == [3048, 3058]
    assert (vs[0].expected_point, vs[0].actual_point) == (2, 1)


def test_violation_targets_align_with_sorted_and_label(result_folder):
    st = _state(result_folder)
    targets = st.violation_targets()
    assert len(targets) == 2
    frame, point, label = targets[0]
    assert (frame, point) == (3048, 1)           # 강조 = 실제(기각된 통과) Point
    assert label == "에러 1/2 — 기대 P2, 실제 P1"
    assert targets[1][2] == "에러 2/2 — 기대 P1, 실제 P2"


def test_violation_targets_empty_for_legacy_folder(result_folder):
    """구버전(위반 CSV 없음) 폴더도 예외 없이 빈 순회 목록."""
    (result_folder / "count_errors.csv").unlink()
    st = _state(result_folder)
    assert st.violations_sorted() == []
    assert st.violation_targets() == []
