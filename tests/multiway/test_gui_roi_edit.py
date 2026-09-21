"""gui_roi_edit 단위 테스트 — tkinter 없이 ROI 편집 기하를 검증한다.

모든 좌표는 A0 규약(+y 아래, angle_deg 양수=화면 위, corners 순서 [-u-n,-u+n,+u-n,+u+n])을 따른다.
1000×1000 프레임 기준: 중심(500,500), length 0.2→hl=100px, width 0.1→hw=50px, angle 0에서
corners = [0]=(400,450) [1]=(400,550) [2]=(600,450) [3]=(600,550).
"""

import math

import pytest

from bubble_counter.multiway.model import PointSpec
from bubble_counter.gui_roi_edit import handles_for, hit_test, Handles, HitTarget


def _pt(**kw):
    return PointSpec(
        kw.pop("number", 1), kw.pop("cx", 0.5), kw.pop("cy", 0.5),
        kw.pop("length", 0.2), kw.pop("width", 0.1), kw.pop("angle_deg", 0.0), **kw)


class TestHandles:
    def test_corners_match_pointspec(self):
        p = _pt()
        h = handles_for(p, 1000, 1000)
        assert h.corners == p.corners_px(1000, 1000)
        assert h.center == pytest.approx((500.0, 500.0))

    def test_rotate_handle_is_above_at_angle_zero(self):
        # angle 0: n=(0,1) → 회전핸들 = center - n*(hw+gap) = (500, 500-50-24) = (500,426)
        p = _pt(width=0.1)
        h = handles_for(p, 1000, 1000, rotate_gap_px=24.0)
        assert h.rotate == pytest.approx((500.0, 426.0))


class TestHitTest:
    def test_hit_corner_zero(self):
        hit = hit_test([_pt()], (400, 450), 1000, 1000, handle_radius_px=6)
        assert hit == HitTarget("corner", 1, 0)

    def test_hit_rotate(self):
        hit = hit_test([_pt(width=0.1)], (500, 426), 1000, 1000, handle_radius_px=6)
        assert hit.kind == "rotate" and hit.number == 1

    def test_hit_body(self):
        hit = hit_test([_pt()], (500, 500), 1000, 1000, handle_radius_px=6)
        assert hit.kind == "body" and hit.number == 1

    def test_hit_none_outside(self):
        hit = hit_test([_pt()], (10, 10), 1000, 1000, handle_radius_px=6)
        assert hit == HitTarget("none", -1)

    def test_topmost_wins(self):
        p1 = _pt(number=1)
        p2 = _pt(number=2)                 # 겹침, 나중에 그림 → 위
        hit = hit_test([p1, p2], (500, 500), 1000, 1000, handle_radius_px=6)
        assert hit.number == 2


from bubble_counter.gui_roi_edit import move_point, resize_point, rotate_point


class TestMove:
    def test_translates_center_ratio(self):
        q = move_point(_pt(cx=0.5, cy=0.5), dx_px=100, dy_px=-50, frame_w=1000, frame_h=1000)
        assert q.cx == pytest.approx(0.6) and q.cy == pytest.approx(0.45)

    def test_clamps_into_frame(self):
        q = move_point(_pt(cx=0.9, cy=0.9), dx_px=1000, dy_px=1000, frame_w=1000, frame_h=1000)
        assert q.cx == 1.0 and q.cy == 1.0


class TestResize:
    def test_axis_aligned_corner_drag(self):
        # corner0=(400,450)를 (300,400)으로 → 반대=corner3=(600,550) 고정
        # center=(450,475), 반폭u=150→length=0.3, 반폭n=75→width=0.15
        q = resize_point(_pt(cx=0.5, cy=0.5, length=0.2, width=0.1, angle_deg=0.0),
                         corner_index=0, new_corner_xy=(300, 400), frame_w=1000, frame_h=1000)
        assert q.length == pytest.approx(0.3)
        assert q.width == pytest.approx(0.15)
        assert q.cx == pytest.approx(0.45) and q.cy == pytest.approx(0.475)
        assert q.angle_deg == pytest.approx(0.0)


class TestRotate:
    def test_handle_straight_up_is_zero(self):
        q = rotate_point(_pt(cx=0.5, cy=0.5), handle_xy=(500, 100), frame_w=1000, frame_h=1000)
        assert q.angle_deg == pytest.approx(0.0)

    def test_handle_to_right_is_minus_90(self):
        # 위쪽 핸들을 오른쪽으로 = 화면상 시계방향 90도 = 음수(A0 규약)
        q = rotate_point(_pt(cx=0.5, cy=0.5), handle_xy=(900, 500), frame_w=1000, frame_h=1000)
        assert q.angle_deg == pytest.approx(-90.0)


from bubble_counter.gui_roi_edit import bubble_dims_from_drag
from bubble_counter.multiway.model import BubbleSizeSpec


class TestBubbleSizer:
    def test_circle_dims_equal(self):
        assert bubble_dims_from_drag("circle", 40, 60) == (50.0, 50.0)   # (40+60)/2

    def test_ellipse_major_ge_minor(self):
        assert bubble_dims_from_drag("ellipse", 40, 70) == (70.0, 40.0)

    def test_dims_construct_bubblesizespec_with_toggle(self):
        maj, mnr = bubble_dims_from_drag("ellipse", 62, 45)
        spec = BubbleSizeSpec("ellipse", maj, mnr, count_half_bubbles=True)
        assert spec.major_px == 62.0 and spec.minor_px == 45.0
        assert spec.count_half_bubbles is True                # A0 계약 정합 확인

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            bubble_dims_from_drag("square", 10, 10)
