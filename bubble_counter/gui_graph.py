"""count error 그래프 좌표·색 — tkinter를 import하지 않는 순수 모듈(설계 §2.5).

세로축 e = 최종 개수 − 목표, 가로축 Point 번호. 0 기준선 강조, 초과=빨강/부족=파랑
(발산), 값은 막대 끝에 항상 병기(색맹·인쇄 판독). GUI는 이 좌표로 tkinter Canvas에
그리고, PNG는 트랙 C가 같은 §2.5 사양으로 OpenCV 렌더한다(별도 구현, 사양 공유).
"""

from __future__ import annotations

from dataclasses import dataclass


def value_label(error: int) -> str:
    return "0" if error == 0 else f"{error:+d}"


# 설계 §2.5: 라이트 파랑 #2a78d6 / 빨강 #e34948, 다크 파랑 #3987e5 / 빨강 #e66767.
# (dataviz validate_palette 4개 검사 전부 PASS한 값)
LIGHT_COLORS = {"pos": "#e34948", "neg": "#2a78d6", "zero": "#8a8f98"}
DARK_COLORS = {"pos": "#e66767", "neg": "#3987e5", "zero": "#9aa0aa"}


def bar_color(error: int, theme: str) -> str:
    if theme == "light":
        pal = LIGHT_COLORS
    elif theme == "dark":
        pal = DARK_COLORS
    else:
        raise ValueError(f"theme는 'light' 또는 'dark'여야 합니다: {theme!r}")
    if error > 0:
        return pal["pos"]
    if error < 0:
        return pal["neg"]
    return pal["zero"]


@dataclass(frozen=True)
class Bar:
    number: int
    x0: float
    x1: float
    y_top: float
    y_bottom: float
    value: int
    label_xy: tuple
    label: str


@dataclass(frozen=True)
class GraphLayout:
    baseline_y: float
    bars: tuple

    def bar_for(self, number: int) -> Bar:
        for b in self.bars:
            if b.number == number:
                return b
        raise KeyError(f"Point {number} 막대가 없습니다")


def error_bar_geometry(errors: dict, canvas_w: int, canvas_h: int, *,
                       top_margin: float = 28, bottom_margin: float = 36,
                       side_margin: float = 32, bar_frac: float = 0.55,
                       label_pad: float = 4) -> GraphLayout:
    """값 라벨이 축 라벨(P1…)과 겹치지 않도록 여백·클램프를 둔다."""
    nums = sorted(errors)
    n = len(nums)
    plot_h = canvas_h - top_margin - bottom_margin
    baseline_y = top_margin + plot_h / 2.0
    half = plot_h / 2.0
    # 막대 끝이 라벨 영역까지 차지하지 않게 스케일 여유
    half_plot = max(1.0, half - 14)
    max_abs = max(1, max((abs(errors[k]) for k in nums), default=1))
    usable = canvas_w - 2 * side_margin
    slot = usable / n if n else usable
    bars = []
    for i, num in enumerate(nums):
        e = errors[num]
        cx = side_margin + slot * (i + 0.5)
        bw = slot * bar_frac
        x0, x1 = cx - bw / 2, cx + bw / 2
        y_e = baseline_y - (e / max_abs) * half_plot
        y_top, y_bottom = min(baseline_y, y_e), max(baseline_y, y_e)
        if e >= 0:
            ly = max(top_margin + 2, y_top - label_pad)
            label_xy = (cx, ly)
        else:
            # 하단 P1… 라벨(h-4)과 겹치지 않게 상한
            ly = min(canvas_h - bottom_margin + 2, y_bottom + label_pad)
            label_xy = (cx, ly)
        bars.append(Bar(num, x0, x1, y_top, y_bottom, e, label_xy, value_label(e)))
    return GraphLayout(baseline_y, tuple(bars))
