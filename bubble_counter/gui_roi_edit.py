"""멀티웨이 ROI 편집 기하 — tkinter를 import하지 않는 순수 모듈.

캔버스 이벤트(마우스)에서 tkinter가 좌표를 화면→이미지(px)로 바꿔(ViewTransform) 넘기면,
이 모듈이 A0 PointSpec 규약에 맞춰 핸들 배치·히트테스트·이동/크기/회전·기포 크기 재기를
순수 계산한다. 반환은 항상 새 PointSpec(frozen)이라 A0의 __post_init__ 검증을 그대로 탄다.

각도·좌표 규약은 A0와 동일: +y 아래, u=(cos a,-sin a), n=(sin a, cos a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from bubble_counter.multiway.model import PointSpec

ROTATE_HANDLE_GAP_PX = 24.0
MIN_SIZE_RATIO = 0.005          # length/width 하한(0 방지, A0 (0,1] 검증 통과용)


@dataclass(frozen=True)
class Handles:
    corners: tuple            # 4점(px), PointSpec.corners_px 순서와 동일
    rotate: tuple             # 회전 핸들(px)
    center: tuple             # 중심(px)


@dataclass(frozen=True)
class HitTarget:
    kind: str                 # "corner" | "rotate" | "body" | "none"
    number: int               # PointSpec.number, 없으면 -1
    corner: int = -1          # kind=="corner"일 때 0..3


def handles_for(point: PointSpec, frame_w: int, frame_h: int,
                rotate_gap_px: float = ROTATE_HANDLE_GAP_PX) -> Handles:
    a = math.radians(point.angle_deg)
    nx, ny = math.sin(a), math.cos(a)                 # 폭축(A0 규약)
    cx, cy = point.cx * frame_w, point.cy * frame_h
    hw = point.width * frame_h / 2.0
    rot = (cx - nx * (hw + rotate_gap_px), cy - ny * (hw + rotate_gap_px))
    return Handles(corners=point.corners_px(frame_w, frame_h), rotate=rot, center=(cx, cy))


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _inside_rect(pt: tuple, p: PointSpec, frame_w: int, frame_h: int) -> bool:
    a = math.radians(p.angle_deg)
    ux, uy = math.cos(a), -math.sin(a)
    nx, ny = math.sin(a), math.cos(a)
    cx, cy = p.cx * frame_w, p.cy * frame_h
    dx, dy = pt[0] - cx, pt[1] - cy
    pu = dx * ux + dy * uy
    pn = dx * nx + dy * ny
    return abs(pu) <= p.length * frame_w / 2.0 and abs(pn) <= p.width * frame_h / 2.0


def hit_test(points, click_xy: tuple, frame_w: int, frame_h: int,
             handle_radius_px: float,
             rotate_gap_px: float = ROTATE_HANDLE_GAP_PX) -> HitTarget:
    # 핸들 먼저(작은 표적), 위에 그려진 것부터. 그다음 몸통.
    for p in reversed(points):
        h = handles_for(p, frame_w, frame_h, rotate_gap_px)
        if _dist(click_xy, h.rotate) <= handle_radius_px:
            return HitTarget("rotate", p.number)
        for i, c in enumerate(h.corners):
            if _dist(click_xy, c) <= handle_radius_px:
                return HitTarget("corner", p.number, i)
    for p in reversed(points):
        if _inside_rect(click_xy, p, frame_w, frame_h):
            return HitTarget("body", p.number)
    return HitTarget("none", -1)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def move_point(point: PointSpec, dx_px: float, dy_px: float,
               frame_w: int, frame_h: int) -> PointSpec:
    cx = _clamp((point.cx * frame_w + dx_px) / frame_w, 0.0, 1.0)
    cy = _clamp((point.cy * frame_h + dy_px) / frame_h, 0.0, 1.0)
    return replace(point, cx=cx, cy=cy)


def resize_point(point: PointSpec, corner_index: int, new_corner_xy: tuple,
                 frame_w: int, frame_h: int) -> PointSpec:
    a = math.radians(point.angle_deg)
    ux, uy = math.cos(a), -math.sin(a)
    nx, ny = math.sin(a), math.cos(a)
    opp = point.corners_px(frame_w, frame_h)[3 - corner_index]   # 대각 모서리 고정
    px, py = new_corner_xy
    center = ((px + opp[0]) / 2.0, (py + opp[1]) / 2.0)
    dx, dy = px - center[0], py - center[1]
    half_len = abs(dx * ux + dy * uy)
    half_wid = abs(dx * nx + dy * ny)
    length = _clamp(2 * half_len / frame_w, MIN_SIZE_RATIO, 1.0)
    width = _clamp(2 * half_wid / frame_h, MIN_SIZE_RATIO, 1.0)
    cx = _clamp(center[0] / frame_w, 0.0, 1.0)
    cy = _clamp(center[1] / frame_h, 0.0, 1.0)
    return replace(point, cx=cx, cy=cy, length=length, width=width)


def rotate_point(point: PointSpec, handle_xy: tuple,
                 frame_w: int, frame_h: int) -> PointSpec:
    cx, cy = point.cx * frame_w, point.cy * frame_h
    hx, hy = handle_xy[0] - cx, handle_xy[1] - cy
    # 회전핸들은 -n 방향(=(-sin a,-cos a)). 관측 벡터로 a 역산:
    angle = math.degrees(math.atan2(-hx, -hy))
    return replace(point, angle_deg=angle)


def bubble_dims_from_drag(shape: str, dx_px: float, dy_px: float) -> tuple[float, float]:
    """기포 크기 재기 드래그 → (major_px, minor_px). A0 BubbleSizeSpec 제약을 충족."""
    w, h = abs(dx_px), abs(dy_px)
    if shape == "circle":
        d = round((w + h) / 2.0, 1)
        return (d, d)
    if shape == "ellipse":
        return (round(max(w, h), 1), round(min(w, h), 1))
    raise ValueError(f"shape는 'circle' 또는 'ellipse'여야 합니다: {shape!r}")
