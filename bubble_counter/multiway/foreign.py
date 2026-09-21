"""정지 이물 스캔 (스펙 ④, R0 §3.1, D-17 실측 판별).

판별: 정지 이물 = 활성 채널 경로를 가로막는 어두운 정지 blob.
  ① 어두움: 국소 배경(medianBlur 31) − 중앙값 > DARK_THR (이물은 흡광 — 어두운 쪽만)
  ② 채널 교차: std > ACT_THR 활성 픽셀의 NEAR_DILATE_PX 팽창 마스크와 교집합
     (칩 벽 텍스처·표면 얼룩은 활성 경로 밖 → 원천 배제)
  ③ 형태: MIN_DIAM~MAX_DIAM, area≥MIN_AREA, aspect≤MAX_ASPECT, fill≥MIN_FILL (벽선 배제)
  ④ 정지성: blob 내부 std 평균 < 인접 채널 std 최대 × STILL_RATIO
근거 실측(1_2 f5850-6349): 파티클(1118,63) diff 25~66·blob std ~4.6, 채널 std 14~18,
벽/얼룩 std 2~3, 표면 플리커 std 6~9. 파라미터는 Task 11 캘리브 게이트가 회귀 락.
"""

from __future__ import annotations

import numpy as np
import cv2

from .model import BubbleEvent, ForeignObject, MultiwayConfig, PointSpec
from .warp import patch_affine, patch_dims

BAND_MARGIN_PX = 25.0   # 계수선 ROI 밴드 주변 마진 — near_points 판정
DARK_THR = 25           # 국소 배경 대비 어두움(그레이)
ACT_THR = 12.0          # 활성 채널 std 하한
NEAR_DILATE_PX = 6      # 채널 마스크 팽창 반경("경로를 가로막음" 판정)
MIN_DIAM = 8
MAX_DIAM = 125
MIN_AREA = 40
MAX_ASPECT = 4.0        # 벽선(길쭉) 배제
MIN_FILL = 0.25
STILL_RATIO = 0.6       # blob std < 인접 채널 max std × 이 값 → 정지물


def frame_stats(grays) -> "tuple[np.ndarray, np.ndarray]":
    """균등 샘플 프레임들의 (픽셀 중앙값 uint8, 픽셀 std float32)."""
    if not grays:
        raise ValueError("프레임 샘플이 비어 있습니다")
    stack = np.stack(grays, axis=0)
    med = np.median(stack, axis=0).astype(np.uint8)
    std = stack.astype(np.float32).std(axis=0)
    return med, std


def _map_to_patch(x: float, y: float, point: PointSpec, frame_w: int, frame_h: int,
                  downstream: str) -> "tuple[float, float]":
    """프레임 좌표 → 패치 좌표 (patch_affine은 패치→프레임 방향이므로 역변환)."""
    M = patch_affine(point, frame_w, frame_h, downstream)
    inv = cv2.invertAffineTransform(M)
    x_p = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
    y_p = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
    return float(x_p), float(y_p)


def point_band_x(fo: ForeignObject, point: PointSpec, frame_w: int, frame_h: int,
                 downstream: str) -> "tuple[bool, float]":
    """이물 중심을 point 패치 좌표로 사상 → (밴드±마진 안?, 패치 폭축 x_p).

    x_p는 BubbleEvent.position_px와 동일 좌표계 — 이물↔이벤트 x-겹침 비교의 기준.
    """
    x_p, y_p = _map_to_patch(fo.x_px, fo.y_px, point, frame_w, frame_h, downstream)
    w_p, h_p = patch_dims(point, frame_w, frame_h)
    m = BAND_MARGIN_PX
    inside = (-m <= x_p <= w_p - 1 + m) and (-m <= y_p <= h_p - 1 + m)
    return inside, x_p


def scan_stats(median_gray: np.ndarray, std_gray: np.ndarray, cfg: MultiwayConfig, *,
               source: str, band_margin_px: float = BAND_MARGIN_PX) -> "list[ForeignObject]":
    """중앙값+std 통계에서 정지 이물 검출(모듈 docstring의 ①~④)."""
    bg = cv2.medianBlur(median_gray, 31)
    dark = (bg.astype(np.int16) - median_gray.astype(np.int16)) > DARK_THR
    active = (std_gray > ACT_THR).astype(np.uint8)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (NEAR_DILATE_PX * 2 + 1, NEAR_DILATE_PX * 2 + 1))
    near_channel = cv2.dilate(active, k) > 0
    mask = (dark & near_channel).astype(np.uint8)
        # morph open removed - relies on size/shape filters instead
        # (2x2 open was too aggressive for sparse signals like parked bubbles)

    n, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    fh, fw = median_gray.shape[:2]
    out: list[ForeignObject] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        diam = max(w, h)
        aspect = diam / max(1, min(w, h))
        fill = area / float(w * h)
        if not (MIN_DIAM <= diam <= MAX_DIAM and area >= MIN_AREA
                and aspect <= MAX_ASPECT and fill >= MIN_FILL):
            continue
        blob_std = float(std_gray[y:y + h, x:x + w].mean())
        y0, y1 = max(0, y - NEAR_DILATE_PX), y + h + NEAR_DILATE_PX
        x0, x1 = max(0, x - NEAR_DILATE_PX), x + w + NEAR_DILATE_PX
        chan_std = float(std_gray[y0:y1, x0:x1].max())
        if blob_std >= chan_std * STILL_RATIO:
            continue                              # 채널 자체(움직임) — 정지물 아님
        cx, cy = float(cents[i][0]), float(cents[i][1])
        probe = ForeignObject(id=len(out), x_px=cx, y_px=cy, w_px=float(w), h_px=float(h),
                              area_px=float(area), near_points=(), source=source)
        near = tuple(p.number for p in cfg.points
                     if point_band_x(probe, p, fw, fh, cfg.effective_direction(p))[0])
        out.append(ForeignObject(id=len(out), x_px=cx, y_px=cy, w_px=float(w),
                                 h_px=float(h), area_px=float(area),
                                 near_points=near, source=source))
    return out


def scan_frames(grays, cfg, *, source: str,
                band_margin_px: float = BAND_MARGIN_PX) -> "list[ForeignObject]":
    """frame_stats → scan_stats 위임 (preflight가 세그먼트 샘플 재사용 시 진입점)."""
    med, std = frame_stats(grays)
    return scan_stats(med, std, cfg, source=source, band_margin_px=band_margin_px)


def scan_video(video_path, cfg, *, n_samples: int = 60,
               source: str = "review-scan") -> "tuple[list[ForeignObject], np.ndarray]":
    """VideoSource.sample_gray 균등 n_samples → scan_frames. 반환 median_gray는 표시/스냅샷용."""
    from bubble_counter.video_io import VideoSource
    source_obj = VideoSource(video_path)
    try:
        idxs = np.linspace(0, source_obj.info.frame_count - 1, n_samples, dtype=int)
        grays = source_obj.sample_gray(idxs.tolist())
        return scan_frames(grays, cfg, source=source), np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    finally:
        source_obj.close()


def overlapping_events(fo, events, point: PointSpec, frame_w, frame_h, downstream,
                       *, margin_px: float = 5.0) -> "list[BubbleEvent]":
    """x-겹침: |event.position_px − fo_x_p| ≤ (event.size_px + fo.w_px)/2 + margin.

    position_px < 0(구버전 미기록) 이벤트는 판정 불가 → 제외(재분석 유도).
    """
    inside, fo_x_p = point_band_x(fo, point, frame_w, frame_h, downstream)
    if not inside:
        return []
    res = []
    for e in events:
        if e.point != point.number:
            continue
        if e.position_px < 0 or e.size_px < 0:
            continue                    # 구버전 미기록 — 판정 불가
        half_span = (e.size_px + fo.w_px) / 2.0 + margin_px
        if abs(e.position_px - fo_x_p) <= half_span:
            res.append(e)
    return res
