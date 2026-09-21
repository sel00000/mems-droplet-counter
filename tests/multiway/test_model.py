"""multiway/model.py 타입·검증 테스트."""

import math
import pytest

from bubble_counter.multiway.model import (
    RecordingSpec, PointSpec, BubbleSizeSpec, MultiwayConfig, VALID_DIRECTIONS, VALID_MODES,
    BubbleEvent, ChapterResult, Correction, PhaseResult, SequenceViolation, SuspectMoment,
    ForeignObject,
)


class TestPointSpec:
    def test_valid_defaults(self):
        p = PointSpec(number=1, cx=0.5, cy=0.5, length=0.1, width=0.05, angle_deg=0.0)
        assert p.direction is None and p.target == 500

    @pytest.mark.parametrize("kw,msg", [
        (dict(number=0, cx=.5, cy=.5, length=.1, width=.05, angle_deg=0.0), "number"),
        (dict(number=1, cx=1.5, cy=.5, length=.1, width=.05, angle_deg=0.0), "cx"),
        (dict(number=1, cx=.5, cy=-.1, length=.1, width=.05, angle_deg=0.0), "cy"),
        (dict(number=1, cx=.5, cy=.5, length=0.0, width=.05, angle_deg=0.0), "length"),
        (dict(number=1, cx=.5, cy=.5, length=.1, width=1.5, angle_deg=0.0), "width"),
        (dict(number=1, cx=.5, cy=.5, length=.1, width=.05, angle_deg=float("nan")), "angle_deg"),
        (dict(number=1, cx=.5, cy=.5, length=.1, width=.05, angle_deg=0.0, direction="up"), "direction"),
        (dict(number=1, cx=.5, cy=.5, length=.1, width=.05, angle_deg=0.0, target=-1), "target"),
    ])
    def test_reject(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            PointSpec(**kw)

    def test_flow_unit_angle_convention(self):
        # +36도 = 화면에서 위로 → uy는 음수(+y가 아래이므로)
        p = PointSpec(1, .5, .5, .1, .05, angle_deg=36.0)
        ux, uy = p.flow_unit()
        assert ux == pytest.approx(math.cos(math.radians(36))) and uy < 0

    def test_corners_axis_aligned(self):
        # 1000x1000 프레임, 중심 (500,500), length 0.2(=200px), width 0.1(=100px), 회전 0
        p = PointSpec(1, .5, .5, .2, .1, angle_deg=0.0)
        cs = p.corners_px(1000, 1000)
        xs = sorted(c[0] for c in cs)
        ys = sorted(c[1] for c in cs)
        assert xs[0] == pytest.approx(400) and xs[-1] == pytest.approx(600)
        assert ys[0] == pytest.approx(450) and ys[-1] == pytest.approx(550)

    def test_corners_rotated_90(self):
        # 90도 회전 → length가 세로축으로: x범위=±width/2, y범위=±length/2
        p = PointSpec(1, .5, .5, .2, .1, angle_deg=90.0)
        cs = p.corners_px(1000, 1000)
        xs = sorted(c[0] for c in cs)
        ys = sorted(c[1] for c in cs)
        assert xs[0] == pytest.approx(450) and xs[-1] == pytest.approx(550)
        assert ys[0] == pytest.approx(400) and ys[-1] == pytest.approx(600)

    def test_corners_nonsquare_frame(self):
        p = PointSpec(1, .5, .5, .1, .1, angle_deg=0.0)
        cs = p.corners_px(1280, 768)
        xs = sorted(c[0] for c in cs); ys = sorted(c[1] for c in cs)
        assert xs[-1] - xs[0] == pytest.approx(128.0)   # length × frame_w
        assert ys[-1] - ys[0] == pytest.approx(76.8)    # width × frame_h
        assert isinstance(cs, tuple)


class TestRecordingSpec:
    def test_valid(self):
        r = RecordingSpec(width=1280, height=768, fps=300.0, mode="dma")
        assert r.width == 1280
        assert r.fps == 300.0

    def test_frozen(self):
        r = RecordingSpec(1280, 768, 300.0, "dma")
        with pytest.raises(Exception):
            r.fps = 15.0

    @pytest.mark.parametrize("kw,msg", [
        (dict(width=0, height=768, fps=300.0, mode="dma"), "width"),
        (dict(width=1280, height=-1, fps=300.0, mode="dma"), "height"),
        (dict(width=1280, height=768, fps=0.0, mode="dma"), "fps"),
        (dict(width=1280, height=768, fps=float("inf"), mode="dma"), "fps"),
        (dict(width=1280, height=768, fps=float("nan"), mode="dma"), "fps"),
        (dict(width=1280, height=768, fps=300.0, mode="ram"), "mode"),
    ])
    def test_reject(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            RecordingSpec(**kw)


class TestBubbleSizeSpec:
    def test_circle_valid(self):
        b = BubbleSizeSpec(shape="circle", major_px=50.0, minor_px=50.0)
        assert b.accept_low == 0.5 and b.accept_high == 1.8
        assert b.count_half_bubbles is False

    def test_min_accept_default(self):
        b = BubbleSizeSpec("circle", 50.0, 50.0)
        assert b.min_accept_px() == pytest.approx(25.0)      # 50 × 0.5

    def test_min_accept_half_toggle(self):
        b = BubbleSizeSpec("circle", 50.0, 50.0, count_half_bubbles=True)
        assert b.min_accept_px() == pytest.approx(12.5)      # 반쪽 인정: 하한 절반

    def test_merge_suspect(self):
        b = BubbleSizeSpec("ellipse", major_px=60.0, minor_px=45.0)
        assert b.min_accept_px() == pytest.approx(22.5)      # min(60,45) × 0.5
        assert b.merge_suspect_px() == pytest.approx(108.0)  # max(60,45) × 1.8

    @pytest.mark.parametrize("kw,msg", [
        (dict(shape="square", major_px=50, minor_px=50), "shape"),
        (dict(shape="circle", major_px=0, minor_px=0), "major_px"),
        (dict(shape="circle", major_px=float("nan"), minor_px=50), "major_px"),
        (dict(shape="ellipse", major_px=40, minor_px=50), "minor_px"),   # major < minor
        (dict(shape="circle", major_px=50, minor_px=40), "circle"),      # 원인데 비등축
        (dict(shape="circle", major_px=50, minor_px=50, accept_low=0.0), "accept_low"),
        (dict(shape="circle", major_px=50, minor_px=50, accept_low=1.0), "accept_low"),
        (dict(shape="circle", major_px=50, minor_px=50, accept_high=1.0), "accept_high"),
    ])
    def test_reject(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            BubbleSizeSpec(**kw)


def test_constants():
    assert VALID_DIRECTIONS == ("+x", "-x", "+y", "-y")
    assert VALID_MODES == ("dma", "memory")


def _rec():
    return RecordingSpec(1280, 768, 300.0, "dma")


def _bub():
    return BubbleSizeSpec("circle", 50.0, 50.0)


def _pt(n, cx=0.5, cy=0.5, **kw):
    return PointSpec(n, cx, cy, kw.pop("length", 0.08), kw.pop("width", 0.06),
                     kw.pop("angle_deg", 0.0), **kw)


def _chapter(counts, corrections=()):
    return ChapterResult(
        video_path="v.avi", recording=_rec(), counts_auto=dict(counts),
        corrections=list(corrections), events=[], violations=[], suspects=[],
        sets_completed=3, processed_frames=500,
    )


class TestMultiwayConfig:
    def test_valid_six_way(self):
        pts = tuple(_pt(i, cx=0.1 * i, cy=0.2) for i in range(1, 7))
        c = MultiwayConfig(recording=_rec(), points=pts, bubble=_bub())
        assert c.way == 6
        assert c.point_by_number(3).number == 3
        assert c.effective_direction(pts[0]) == "+x"

    def test_point_direction_override(self):
        p = _pt(1, direction="-y")
        c = MultiwayConfig(_rec(), (p,), _bub())
        assert c.effective_direction(p) == "-y"

    def test_reject_zero_points(self):
        with pytest.raises(ValueError, match="points"):
            MultiwayConfig(_rec(), (), _bub())

    def test_reject_ten_points(self):
        pts = tuple(_pt(i, cx=0.09 * i) for i in range(1, 11))
        with pytest.raises(ValueError, match="9"):
            MultiwayConfig(_rec(), pts, _bub())

    def test_reject_duplicate_number(self):
        with pytest.raises(ValueError, match="중복"):
            MultiwayConfig(_rec(), (_pt(1, cx=0.2), _pt(1, cx=0.6)), _bub())

    def test_reject_missing_number(self):
        # {1, 3} → "2번이 없습니다"
        with pytest.raises(ValueError, match="2"):
            MultiwayConfig(_rec(), (_pt(1, cx=0.2), _pt(3, cx=0.6)), _bub())

    def test_reject_out_of_frame_rotated(self):
        # 중심이 프레임 우상단 구석 + 회전 → 꼭짓점이 프레임 밖
        p = _pt(1, cx=0.99, cy=0.01, length=0.1, width=0.08, angle_deg=45.0)
        with pytest.raises(ValueError, match="프레임"):
            MultiwayConfig(_rec(), (p,), _bub())

    def test_reject_bad_global_direction(self):
        with pytest.raises(ValueError, match="global_direction"):
            MultiwayConfig(_rec(), (_pt(1),), _bub(), global_direction="north")

    def test_reject_negative_tolerance(self):
        with pytest.raises(ValueError, match="order_tolerance_frames"):
            MultiwayConfig(_rec(), (_pt(1),), _bub(), order_tolerance_frames=-1)

    def test_point_by_number_missing_raises(self):
        c = MultiwayConfig(_rec(), (_pt(1),), _bub())
        with pytest.raises(KeyError, match="7"):
            c.point_by_number(7)

    def test_frozen(self):
        c = MultiwayConfig(_rec(), (_pt(1),), _bub())
        with pytest.raises(Exception):
            c.global_direction = "-x"


class TestResults:
    def test_event_frozen_defaults(self):
        e = BubbleEvent(point=1, frame=32, subframe=0.25)
        assert e.flags == ()
        with pytest.raises(Exception):
            e.frame = 33

    def test_correction_delta_validation(self):
        with pytest.raises(ValueError, match="delta"):
            Correction(point=1, frame=10, delta=0, reason="x", corrected_at="t")
        with pytest.raises(ValueError, match="delta"):
            Correction(point=1, frame=10, delta=2, reason="x", corrected_at="t")

    def test_counts_final_applies_corrections(self):
        ch = _chapter({1: 10, 2: 11}, [
            Correction(1, 100, +1, "겹침 확인", "2026-07-10T00:00:00"),
            Correction(1, 200, -1, "파티클", "2026-07-10T00:00:00"),
            Correction(2, 300, +1, "누락", "2026-07-10T00:00:00"),
        ])
        assert ch.counts_final() == {1: 10, 2: 12}
        assert ch.counts_auto == {1: 10, 2: 11}   # 원본 불변

    def test_errors(self):
        ch = _chapter({1: 498, 2: 502})
        assert ch.errors({1: 500, 2: 500}) == {1: -2, 2: 2}

    def test_phase_sums(self):
        ph = PhaseResult("6way_1", [_chapter({1: 10}), _chapter({1: 5})])
        assert ph.sum_counts_final() == {1: 15}
        assert ph.sum_sets() == 6

    def test_trailing_default_fields(self):
        ch = _chapter({1: 1})
        assert ch.filtered_out == [] and ch.self_check_messages == []
        assert ch.suspects_truncated is False and ch.cancelled is False
        ch.filtered_out.append((1, 42, 18.0, "undersize"))      # (point, frame, size_px, reason)
        assert ch.filtered_out == [(1, 42, 18.0, "undersize")]

    def test_bubble_event_trailing_defaults(self):
        e = BubbleEvent(point=1, frame=10)
        assert e.position_px == -1.0 and e.size_px == -1.0        # 구버전 센티널
        e2 = BubbleEvent(point=1, frame=10, position_px=50.0, size_px=44.0)
        assert (e2.position_px, e2.size_px) == (50.0, 44.0)

    def test_correction_evidence_path_default(self):
        c = Correction(point=5, frame=6090, delta=-1, reason="foreign-pair",
                       corrected_at="2026-07-12T14:02:00")
        assert c.evidence_path == ""
        assert c.delta == -1                                       # 기존 __post_init__ 검증 유지

    def test_foreign_object_fields_and_defaults(self):
        fo = ForeignObject(id=0, x_px=1118.0, y_px=63.0, w_px=8.0, h_px=8.0,
                           area_px=50.0, near_points=(5,), source="preflight")
        assert fo.tag == "" and fo.tagged_at == "" and fo.note == ""
        assert fo.near_points == (5,) and fo.source == "preflight"
