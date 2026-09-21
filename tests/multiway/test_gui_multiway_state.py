"""gui_multiway_state 단위 테스트 — tkinter 없이 위저드 흐름/입력모델을 검증한다."""

import pytest

from bubble_counter.multiway.model import RecordingSpec
from bubble_counter.gui_multiway_state import WizardStep, WizardState
from bubble_counter.gui_multiway_state import Chapter, ChapterList
from bubble_counter.gui_multiway_state import (
    parse_frame_input, parse_target, parse_point_number,
    arrow_for, DIRECTION_ARROWS, PreflightItem, PreflightDisplay,
)


def _rec():
    return RecordingSpec(1280, 768, 300.0, "dma")


class TestWizardStep:
    def test_order_and_values(self):
        assert [s.value for s in WizardStep] == [1, 2, 3, 4, 5, 6]
        assert WizardStep.RECORDING < WizardStep.RESULTS


class TestWizardState:
    def test_video_open_needs_recording_and_chapter(self):
        s = WizardState()
        assert s.video_open_enabled() is False
        s.recording = _rec()
        assert s.video_open_enabled() is False   # 챕터 없으면 미리보기 불가
        s.chapter_count = 1
        assert s.video_open_enabled() is True

    def test_step1_blocks_without_recording(self):
        s = WizardState()
        ok, msg = s.can_advance()
        assert ok is False and "녹화 설정" in msg

    def test_full_advance_chain(self):
        s = WizardState()
        s.recording = _rec()
        assert s.can_advance()[0] is True
        s.advance(); assert s.step == WizardStep.CHAPTERS
        assert s.can_advance()[0] is False              # 챕터 0개
        s.chapter_count = 2
        s.advance(); assert s.step == WizardStep.ROI
        assert s.can_advance()[0] is False              # ROI 0개
        s.roi_count = 1
        s.advance(); assert s.step == WizardStep.PREFLIGHT
        assert s.can_advance()[0] is False              # 프리플라이트 미통과
        s.preflight_passed = True
        s.advance(); assert s.step == WizardStep.RUN
        assert s.can_advance()[0] is False              # 측정 미완료
        s.run_complete = True
        s.advance(); assert s.step == WizardStep.RESULTS

    def test_advance_blocked_is_noop(self):
        s = WizardState()                               # recording 없음
        s.advance()
        assert s.step == WizardStep.RECORDING           # 전진 안 함

    def test_back_never_below_first(self):
        s = WizardState()
        s.back()
        assert s.step == WizardStep.RECORDING

    def test_invalidate_preflight_blocks_advance_keeps_run_complete(self):
        # D-13후속(review-D13-verdict.md Critical-1): preflight_passed/run_complete는
        # back()으로도 리셋되지 않는 stale 플래그라, "◀ 이전 → ROI 편집 → 다음 ▶"으로
        # 되돌아오면 재검증 없이 전진할 수 있었다. ROI 편집 시 invalidate_preflight()를
        # 호출해 4단계 재통과를 강제해야 한다.
        s = WizardState()
        s.recording = _rec()
        s.chapter_count = 2
        s.roi_count = 1
        s.step = WizardStep.PREFLIGHT
        s.preflight_passed = True
        s.run_complete = True                    # RESULTS까지 갔다 되돌아온 상황 재현
        assert s.can_advance()[0] is True
        s.invalidate_preflight()
        assert s.can_advance() == (False, "프리플라이트 차단 항목을 해결하세요")
        assert s.run_complete is True             # 기존 결과 열람은 계속 허용(건드리지 않음)


class TestChapterList:
    def test_add_preserves_order(self):
        cl = ChapterList()
        cl.add("/a/6way_1_2.avi", 1280, 768, 12000)
        cl.add("/a/6way_1_1.avi", 1280, 768, 12000)
        assert [c.path for c in cl] == ["/a/6way_1_2.avi", "/a/6way_1_1.avi"]
        assert len(cl) == 2

    def test_phase_name_common_prefix(self):
        cl = ChapterList()
        cl.add("/x/6way_1_1.avi", 1280, 768, 100)
        cl.add("/x/6way_1_2.avi", 1280, 768, 100)
        assert cl.phase_name() == "6way_1"

    def test_phase_name_fallback_when_no_prefix(self):
        cl = ChapterList()
        cl.add("/x/abc.avi", 1, 1, 1)
        cl.add("/x/zzz.avi", 1, 1, 1)
        assert cl.phase_name() == "phase"

    def test_resolution_mismatch_indices(self):
        cl = ChapterList()
        cl.add("/x/a.avi", 1280, 768, 100)
        cl.add("/x/b.avi", 1920, 1080, 100)
        assert cl.resolution_mismatches(_rec()) == [1]

    def test_mode_contradiction_over_2gb(self):
        rec = RecordingSpec(1280, 768, 300.0, "memory")
        cl = ChapterList()
        cl.add("/x/a.avi", 1280, 768, 800)  # 1280*768*3*800 ≈ 2.36GB > 2GB
        cl.add("/x/b.avi", 1280, 768, 100)  # ≈ 0.29GB
        assert cl.mode_contradictions(rec) == [0]

    def test_remove(self):
        cl = ChapterList()
        cl.add("/x/a.avi", 1, 1, 1)
        cl.add("/x/b.avi", 1, 1, 1)
        cl.remove(0)
        assert [c.path for c in cl] == ["/x/b.avi"]


class TestInputParsers:
    def test_parse_frame_ok(self):
        assert parse_frame_input("100", total_frames=500) == 100

    def test_parse_frame_non_int(self):
        with pytest.raises(ValueError, match="프레임 번호"):
            parse_frame_input("x", 500)

    @pytest.mark.parametrize("text", ["500", "-1"])
    def test_parse_frame_out_of_range(self, text):
        with pytest.raises(ValueError, match="범위"):
            parse_frame_input(text, 500)          # 유효 = 0..499

    def test_parse_target(self):
        assert parse_target("0") == 0 and parse_target("500") == 500
        with pytest.raises(ValueError, match="목표"):
            parse_target("-1")
        with pytest.raises(ValueError, match="목표"):
            parse_target("abc")

    def test_parse_point_number(self):
        assert parse_point_number("3", existing=(1, 2)) == 3
        with pytest.raises(ValueError, match="1 이상"):
            parse_point_number("0", existing=())
        with pytest.raises(ValueError, match="이미 사용"):
            parse_point_number("2", existing=(1, 2))


class TestDirectionArrows:
    def test_map(self):
        assert DIRECTION_ARROWS == {"+x": "→", "-x": "←", "+y": "↓", "-y": "↑"}
        assert arrow_for("+y") == "↓"

    def test_bad_direction(self):
        with pytest.raises(ValueError, match="direction"):
            arrow_for("north")


class TestPreflightDisplay:
    def test_split_block_and_warn(self):
        d = PreflightDisplay.from_items([
            PreflightItem("block", "3번이 없습니다"),
            PreflightItem("warn", "기포가 뭉쳐 지나갑니다"),
            PreflightItem("block", "ROI가 겹칩니다"),
        ])
        assert d.blocking == ["3번이 없습니다", "ROI가 겹칩니다"]
        assert d.warnings == ["기포가 뭉쳐 지나갑니다"]
        assert d.has_blocking is True

    def test_empty_has_no_blocking(self):
        d = PreflightDisplay.from_items([])
        assert d.has_blocking is False
