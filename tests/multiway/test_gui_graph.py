"""gui_graph 단위 테스트 — tkinter 없이 count error 그래프 좌표/색을 검증한다(설계 §2.5)."""

import pytest

from bubble_counter.gui_graph import (
    error_bar_geometry, value_label, GraphLayout, Bar,
    bar_color, LIGHT_COLORS, DARK_COLORS,
)


class TestValueLabel:
    def test_signed_with_zero_plain(self):
        assert value_label(0) == "0"
        assert value_label(2) == "+2"
        assert value_label(-1) == "-1"


class TestGeometry:
    def test_baseline_centered(self):
        # 설계 §2.5 예: Point1..6 오차 0,0,0,0,-1,+1
        lay = error_bar_geometry({1: 0, 2: 0, 3: 0, 4: 0, 5: -1, 6: 1},
                                 canvas_w=600, canvas_h=200,
                                 top_margin=20, bottom_margin=20, side_margin=30)
        assert lay.baseline_y == pytest.approx(100.0)     # 20 + (200-40)/2

    def test_positive_above_negative_below(self):
        lay = error_bar_geometry({1: 1, 2: -1}, canvas_w=400, canvas_h=200,
                                 top_margin=20, bottom_margin=20, side_margin=20)
        assert lay.bar_for(1).y_top < lay.baseline_y      # +1 막대는 기준선 위(작은 y)
        assert lay.bar_for(2).y_bottom > lay.baseline_y   # -1 막대는 기준선 아래

    def test_symmetric_full_deflection(self):
        # 라벨 여유(half_plot = half-14)만큼 max 편향이 top/bottom에 안 닿음
        lay = error_bar_geometry({1: 2, 2: -2}, canvas_w=400, canvas_h=200,
                                 top_margin=20, bottom_margin=20, side_margin=20)
        assert lay.bar_for(1).y_top == pytest.approx(34.0)     # baseline - (half-14)
        assert lay.bar_for(2).y_bottom == pytest.approx(166.0)

    def test_value_label_not_past_axis_band(self):
        # 큰 |e|에서도 값 라벨이 하단 P축 영역(h-bottom_margin) 안에 머묾
        lay = error_bar_geometry({1: -194, 2: -194}, canvas_w=560, canvas_h=220,
                                 top_margin=30, bottom_margin=40, side_margin=32)
        for b in lay.bars:
            assert b.label == "-194"
            assert b.label_xy[1] <= 220 - 40 + 4

    def test_bars_sorted_by_number(self):
        lay = error_bar_geometry({2: 0, 1: 0}, canvas_w=400, canvas_h=200,
                                 top_margin=10, bottom_margin=10, side_margin=20)
        assert [b.number for b in lay.bars] == [1, 2]

    def test_label_text_matches_value(self):
        lay = error_bar_geometry({1: -1, 2: 1}, canvas_w=400, canvas_h=200,
                                 top_margin=20, bottom_margin=20, side_margin=20)
        assert lay.bar_for(1).label == "-1"
        assert lay.bar_for(2).label == "+1"


class TestColors:
    def test_light_diverging(self):
        assert bar_color(3, "light") == "#e34948"     # 초과 = 빨강
        assert bar_color(-3, "light") == "#2a78d6"    # 부족 = 파랑

    def test_dark_diverging(self):
        assert bar_color(3, "dark") == "#e66767"
        assert bar_color(-3, "dark") == "#3987e5"

    def test_zero_is_neutral(self):
        assert bar_color(0, "light") == LIGHT_COLORS["zero"]
        assert bar_color(0, "dark") == DARK_COLORS["zero"]

    def test_bad_theme(self):
        with pytest.raises(ValueError, match="theme"):
            bar_color(1, "blue")
