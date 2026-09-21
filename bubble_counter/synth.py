"""Synthetic bubble video generator with analytically known ground truth.

Renders an MJPG .avi test video containing circular "bubbles" that spawn
just below the visible frame and move straight upward (with a small
horizontal sinusoidal drift), plus an optional ground-truth JSON file
recording exactly when each bubble crosses a horizontal counting line.

The ground truth is derived purely from the analytic motion model
(``crossing_frame`` below), independent of rendering, so it is exact and
does not depend on any downstream vision processing.

Standalone module: only numpy/cv2/stdlib, plus (for the multiway section
below) ``multiway.model.PointSpec`` — itself pure stdlib, so no circular
or heavyweight bubble_counter dependency is introduced.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .multiway.model import PointSpec


@dataclass
class SynthConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    duration_s: float = 20.0
    rate_per_s: float = 2.5        # bubbles spawned per second after lead-in
    speed_min: float = 2.0         # px/frame upward (full-frame scale)
    speed_max: float = 60.0
    radius_min: int = 5
    radius_max: int = 10
    brightness: float = 80.0       # additive intensity delta (may be negative)
    alpha: float = 0.85            # bubble opacity
    noise_sigma: float = 2.0       # per-frame gaussian noise
    lead_in_s: float = 2.0         # bubble-free lead-in (aligns with MOG2 warmup)
    line_pos: float = 0.5          # counting line as fraction of full-frame HEIGHT
    drift_px: float = 1.0          # horizontal sinusoidal drift amplitude
    seed: int = 42


def crossing_frame(f0: int, y0: float, speed: float, line_y: float) -> int | None:
    """Minimal integer f >= f0 with y0 - speed * (f - f0) <= line_y.

    y0 is the bubble's y position at frame f0 (spawn), speed is the
    constant upward px/frame velocity. Returns None if the bubble never
    reaches the line (speed <= 0 and y0 is still above the line).
    """
    delta = y0 - line_y
    if delta <= 0:
        return f0
    if speed <= 0:
        return None
    return f0 + math.ceil(delta / speed)


def _line_y(cfg: SynthConfig) -> int:
    """Counting-line pixel row, matching geometry.line_center_px's formula
    (round(pos * (roi_len - 1)) clipped to [0, roi_len - 1]) with roi_len =
    the full frame height, so an E2E run over the full frame lines up
    exactly with this GT.
    """
    if cfg.height <= 1:
        return 0
    y = round(cfg.line_pos * (cfg.height - 1))
    return min(max(y, 0), cfg.height - 1)


def _n_frames(cfg: SynthConfig) -> int:
    return max(0, int(round(cfg.duration_s * cfg.fps)))


def _lead_in_frames(cfg: SynthConfig) -> int:
    return max(0, int(round(cfg.lead_in_s * cfg.fps)))


def _make_background(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """Static vertical gradient (60 -> 120, top to bottom) plus a seed-fixed
    blurred noise texture. Generated once and reused every frame.
    """
    gradient = np.linspace(60.0, 120.0, cfg.height, dtype=np.float32)
    gradient = np.tile(gradient.reshape(-1, 1), (1, cfg.width))
    texture = rng.normal(0.0, 6.0, size=(cfg.height, cfg.width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=3.0)
    return gradient + texture


@dataclass
class _Bubble:
    f0: int
    y0: float
    speed: float
    radius: int
    x0: float
    phase: float


def _spawn_bubbles(cfg: SynthConfig, rng: np.random.Generator,
                    n_frames: int, lead_in_frames: int) -> list[_Bubble]:
    if lead_in_frames >= n_frames:
        return []
    n_bubbles = max(0, round(cfg.rate_per_s * (cfg.duration_s - cfg.lead_in_s)))
    if n_bubbles == 0:
        return []

    f0s = rng.integers(lead_in_frames, n_frames, size=n_bubbles)
    speeds = rng.uniform(cfg.speed_min, cfg.speed_max, size=n_bubbles)
    radii = rng.integers(cfg.radius_min, cfg.radius_max + 1, size=n_bubbles)
    x0s = rng.uniform(0.0, cfg.width, size=n_bubbles)
    phases = rng.uniform(0.0, 2 * math.pi, size=n_bubbles)

    bubbles = []
    for i in range(n_bubbles):
        radius = int(radii[i])
        bubbles.append(_Bubble(
            f0=int(f0s[i]),
            y0=float(cfg.height + radius),
            speed=float(speeds[i]),
            radius=radius,
            x0=float(x0s[i]),
            phase=float(phases[i]),
        ))
    return bubbles


def _bubble_xy(b: _Bubble, f: int, drift_px: float) -> tuple[float, float]:
    df = f - b.f0
    y = b.y0 - b.speed * df
    x = b.x0 + drift_px * math.sin(2 * math.pi * df / 60.0 + b.phase)
    return x, y


def _active_bubbles(bubbles: list[_Bubble], f: int, cfg: SynthConfig) -> list[_Bubble]:
    """Bubbles spawned by frame f whose circle still intersects the frame."""
    active = []
    for b in bubbles:
        if f < b.f0:
            continue
        x, y = _bubble_xy(b, f, cfg.drift_px)
        if y + b.radius <= 0 or y - b.radius >= cfg.height:
            continue
        if x + b.radius <= 0 or x - b.radius >= cfg.width:
            continue
        active.append(b)
    return active


def _compute_crossings(bubbles: list[_Bubble], cfg: SynthConfig,
                        line_y: int, n_frames: int) -> list[dict]:
    """GT crossings, computed analytically from the motion model only."""
    crossings = []
    for b in bubbles:
        cf = crossing_frame(b.f0, b.y0, b.speed, line_y)
        if cf is None or cf >= n_frames:
            continue
        x, _y = _bubble_xy(b, cf, cfg.drift_px)
        crossings.append({"frame": int(cf), "time_s": cf / cfg.fps, "x": float(x)})
    crossings.sort(key=lambda c: (c["frame"], c["x"]))
    return crossings


def generate(cfg: SynthConfig, out_video: str | Path,
             gt_json: str | Path | None = None) -> dict:
    """Write MJPG .avi + optional GT json. Return GT dict:
    {"total_crossings": int, "line_y": int, "n_frames": int,
     "crossings": [{"frame": int, "time_s": float, "x": float}, ...],
     "config": {...asdict...}}

    Bubble model: spawn frame f0 (integer, uniformly among frames in
    [lead_in_s*fps, n_frames)), y(f) = (height + radius) - speed * (f - f0),
    x(f) = x0 + drift_px * sin(2*pi*(f - f0)/60 + phase); bubble is drawn
    while it intersects the frame. Crossing frame = min integer f with
    y(f) <= line_y. GT counts crossings with f < n_frames only.
    Deterministic per seed (np.random.default_rng(cfg.seed)).
    """
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    n_frames = _n_frames(cfg)
    lead_in_frames = _lead_in_frames(cfg)
    line_y = _line_y(cfg)

    background = _make_background(cfg, rng)
    bubbles = _spawn_bubbles(cfg, rng, n_frames, lead_in_frames)
    crossings = _compute_crossings(bubbles, cfg, line_y, n_frames)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_video), fourcc, cfg.fps,
                              (cfg.width, cfg.height), isColor=True)
    if not writer.isOpened():
        raise IOError(f"영상 파일을 생성할 수 없습니다: {out_video}")

    layer = np.zeros((cfg.height, cfg.width), dtype=np.float32)
    try:
        for f in range(n_frames):
            noise = rng.normal(0.0, cfg.noise_sigma,
                                size=(cfg.height, cfg.width)).astype(np.float32)
            frame_f32 = background + noise

            active = _active_bubbles(bubbles, f, cfg)
            if active:
                layer.fill(0.0)
                for b in active:
                    x, y = _bubble_xy(b, f, cfg.drift_px)
                    cv2.circle(layer, (int(round(x)), int(round(y))), b.radius, 1.0, -1)
                layer_blurred = cv2.GaussianBlur(layer, (0, 0), sigmaX=1.0)
                frame_f32 += cfg.brightness * cfg.alpha * layer_blurred

            frame_u8 = np.clip(frame_f32, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    gt = {
        "total_crossings": len(crossings),
        "line_y": line_y,
        "n_frames": n_frames,
        "crossings": crossings,
        "config": asdict(cfg),
    }

    if gt_json is not None:
        gt_path = Path(gt_json)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    return gt


# ==========================================================================
# 멀티웨이 시나리오 생성기 (설계 §4.2) — 해석적 GT, MJPG(cv2 안전)
# ==========================================================================


@dataclass
class MultiwaySynthConfig:
    width: int = 1280
    height: int = 768
    fps: float = 300.0
    mode: str = "dma"
    n_frames: int = 200
    radius: int = 20            # 기포 반지름 px(~40px 지름, 실측 근사)
    brightness: float = 80.0
    alpha: float = 0.85
    noise_sigma: float = 2.0
    seed: int = 42


@dataclass
class MwBubble:
    point: int
    radius: int
    waypoints: list          # [(frame:int, x:float, y:float), ...] frame 오름차순


def _pos_at(b: MwBubble, f: int) -> tuple[float, float]:
    wps = b.waypoints
    if f <= wps[0][0]:
        return wps[0][1], wps[0][2]
    if f >= wps[-1][0]:
        return wps[-1][1], wps[-1][2]
    for (f1, x1, y1), (f2, x2, y2) in zip(wps, wps[1:]):
        if f1 <= f <= f2:
            t = (f - f1) / (f2 - f1) if f2 > f1 else 0.0
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    return wps[-1][1], wps[-1][2]


def straight_bubble(point_num: int, p: PointSpec, cfg: MultiwaySynthConfig,
                    f0: int, frames: int, dist: float = 120.0) -> MwBubble:
    """Point 중심을 흐름축 따라 상류→하류로 관통하는 직선 기포."""
    cx, cy = p.cx * cfg.width, p.cy * cfg.height
    ux, uy = p.flow_unit()
    start = (cx - dist * ux, cy - dist * uy)
    end = (cx + dist * ux, cy + dist * uy)
    return MwBubble(point_num, cfg.radius,
                    [(f0, start[0], start[1]), (f0 + frames, end[0], end[1])])


def _point_by_number(points, n: int) -> PointSpec:
    for p in points:
        if p.number == n:
            return p
    raise KeyError(f"Point {n} 없음")


def _compute_events(points, bubbles, cfg: MultiwaySynthConfig) -> list:
    """해석적 GT — 각 기포의 계수 평면 하류 순통과 이벤트(설계 §3 결정3 정답).

    전제: 기포는 첫 waypoint에서 상류(s<0)여야 상류→하류 전이가 관측되어
    이벤트가 기록된다. 첫 waypoint부터 이미 하류(s>=0)로 시작하는 기포는
    "관측된 교차"가 없어 이벤트가 무음으로 누락된다(C11 코드리뷰 인계 사항,
    C12에서 재검토 — `straight_bubble`/현재 정의된 시나리오는 전부 첫
    waypoint가 상류(s=-dist<0 또는 명시적 상류 좌표)에서 시작해 해당 없음).
    새 시나리오 추가 시 순 통과로 셀 기포는 첫 waypoint를 상류에 두어야 한다.
    """
    events = []
    for b in bubbles:
        p = _point_by_number(points, b.point)
        cx, cy = p.cx * cfg.width, p.cy * cfg.height
        ux, uy = p.flow_unit()
        prev_s = None
        cross = None
        for f in range(b.waypoints[0][0], b.waypoints[-1][0] + 1):
            x, y = _pos_at(b, f)
            s = (x - cx) * ux + (y - cy) * uy
            if prev_s is not None and prev_s < 0 <= s and cross is None:
                cross = (f, x, y)
            prev_s = s
        if cross is not None and prev_s is not None and prev_s >= 0:   # 순 하류 통과
            events.append({"point": b.point, "frame": int(cross[0]),
                           "x": float(cross[1]), "y": float(cross[2])})
    events.sort(key=lambda e: (e["frame"], e["point"]))
    return events


def _render_multiway(cfg: MultiwaySynthConfig, bubbles, out_video) -> None:
    rng = np.random.default_rng(cfg.seed)
    W, H = cfg.width, cfg.height
    gradient = np.tile(np.linspace(60.0, 120.0, H, dtype=np.float32).reshape(-1, 1), (1, W))
    texture = cv2.GaussianBlur(rng.normal(0.0, 6.0, (H, W)).astype(np.float32), (0, 0), 3.0)
    background = gradient + texture

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_video), fourcc, cfg.fps, (W, H), isColor=True)
    if not writer.isOpened():
        raise IOError(f"영상 파일을 생성할 수 없습니다: {out_video}")
    layer = np.zeros((H, W), dtype=np.float32)
    try:
        for f in range(cfg.n_frames):
            frame = background + rng.normal(0.0, cfg.noise_sigma, (H, W)).astype(np.float32)
            active = [b for b in bubbles
                      if b.waypoints[0][0] <= f <= b.waypoints[-1][0]]
            if active:
                layer.fill(0.0)
                for b in active:
                    x, y = _pos_at(b, f)
                    cv2.circle(layer, (int(round(x)), int(round(y))), b.radius, 1.0, -1)
                blurred = cv2.GaussianBlur(layer, (0, 0), 1.0)
                frame += cfg.brightness * cfg.alpha * blurred
            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def generate_multiway(cfg: MultiwaySynthConfig, points, bubbles, out_video,
                      gt_json=None) -> dict:
    """MJPG 클립 + 해석적 GT dict. GT 스키마는 트랙 C 플랜 Interfaces 참조."""
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    _render_multiway(cfg, bubbles, out_video)
    events = _compute_events(points, bubbles, cfg)
    counts: dict = {}
    for e in events:
        counts[e["point"]] = counts.get(e["point"], 0) + 1
    gt = {
        "recording": {"width": cfg.width, "height": cfg.height,
                      "fps": cfg.fps, "mode": cfg.mode},
        "points": [asdict(p) for p in points],
        "expected_events": events,
        "expected_counts": counts,
        "n_frames": cfg.n_frames,
        "notes": {},
    }
    if gt_json is not None:
        gt_path = Path(gt_json)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(json.dumps(gt, ensure_ascii=False, allow_nan=False, indent=2),
                            encoding="utf-8")
    return gt


def _finalize_gt(gt: dict, scenario: str, notes: dict, gt_json) -> dict:
    """GT dict에 scenario/notes 병합 후(gt_json 지정 시) 재저장."""
    gt["scenario"] = scenario
    gt["notes"] = notes
    if gt_json is not None:
        Path(gt_json).write_text(
            json.dumps(gt, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8")
    return gt


def rotated_channel_clip(out_video, *, gt_json=None, seed=42) -> dict:
    """회전 채널(angle 30°) 1-Point에 직선 기포 5개 → 5 이벤트."""
    cfg = MultiwaySynthConfig(seed=seed)
    p = PointSpec(1, 0.5, 0.5, 0.20, 0.08, angle_deg=30.0)
    points = (p,)
    bubbles = [straight_bubble(1, p, cfg, f0=10 + i * 30, frames=24) for i in range(5)]
    gt = generate_multiway(cfg, points, bubbles, out_video, gt_json)
    return _finalize_gt(gt, "rotated_channel", {"angle_deg": 30.0}, gt_json)


def merge_clip(out_video, *, gt_json=None, seed=42) -> dict:
    """두 기포가 평면 근처에서 겹쳐 통과 → 정답 2개(merge 의심 대상)."""
    cfg = MultiwaySynthConfig(seed=seed)
    p = PointSpec(1, 0.5, 0.5, 0.20, 0.08, angle_deg=0.0)
    points = (p,)
    cx, cy = p.cx * cfg.width, p.cy * cfg.height
    a = straight_bubble(1, p, cfg, f0=20, frames=24)                 # 중심 관통
    b = MwBubble(1, cfg.radius,                                      # +10px 측면 오프셋(겹침)
                 [(20, cx - 120, cy + 10), (44, cx + 120, cy + 10)])
    gt = generate_multiway(cfg, points, [a, b], out_video, gt_json)
    merge_frame = gt["expected_events"][0]["frame"] if gt["expected_events"] else 32
    return _finalize_gt(gt, "merge", {"merge_frame": int(merge_frame)}, gt_json)


def reflux_clip(out_video, *, gt_json=None, seed=42) -> dict:
    """하류→상류→하류로 끝나는 역류 기포 → 순 통과 1개."""
    cfg = MultiwaySynthConfig(seed=seed)
    p = PointSpec(1, 0.5, 0.5, 0.20, 0.08, angle_deg=0.0)
    points = (p,)
    cy = p.cy * cfg.height
    reflux = MwBubble(1, cfg.radius, [
        (10, 520, cy),   # 상류
        (34, 720, cy),   # 하류(첫 통과)
        (58, 560, cy),   # 되돌아 상류(역류)
        (82, 760, cy),   # 다시 하류(최종 하류 → 순 통과 1)
    ])
    gt = generate_multiway(cfg, points, [reflux], out_video, gt_json)
    return _finalize_gt(gt, "reflux", {"reflux_bubble_point": 1}, gt_json)


def tight_column_clip(out_video, *, gt_json=None, seed=42) -> dict:
    """6개 기포가 촘촘히 한 줄로 통과 → 정답 6개(밀집 분리 검증)."""
    cfg = MultiwaySynthConfig(seed=seed)
    p = PointSpec(1, 0.5, 0.5, 0.20, 0.08, angle_deg=0.0)
    points = (p,)
    bubbles = [straight_bubble(1, p, cfg, f0=10 + i * 4, frames=40) for i in range(6)]
    gt = generate_multiway(cfg, points, bubbles, out_video, gt_json)
    return _finalize_gt(gt, "tight_column", {"n_bubbles": 6}, gt_json)
