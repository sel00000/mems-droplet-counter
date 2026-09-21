"""multiway/foreign.py — 정지 이물 스캔 (R0 §3.1, D-17). 합성 픽스처 검증."""

import numpy as np
import cv2
import pytest

from bubble_counter.multiway import foreign
from bubble_counter.multiway.model import (
    BubbleSizeSpec, ForeignObject, MultiwayConfig, PointSpec, RecordingSpec)


def _synthetic(with_channel=True, with_blob=True, with_wall=False, n=40):
    """(200,300) 합성: 배경 120, 채널=y≈50 행을 좌→우로 지나는 밝은 원(프레임별 이동),
    blob=(150,65) 정지 어두운 원(채널 경로 바로 아래), wall=x40 세로 어두운 선(채널 밖)."""
    rng = np.random.default_rng(7)
    frames = []
    for k in range(n):
        f = np.full((200, 300), 120, np.uint8)
        if with_wall:
            f[:, 40:44] = 70
        if with_blob:
            cv2.circle(f, (150, 65), 6, 60, -1)
        if with_channel:
            cv2.circle(f, (10 + (k * 23) % 280, 50), 12, 200, -1)
        noise = rng.integers(0, 3, f.shape, np.uint8)
        frames.append(cv2.add(f, noise))
    return frames


def _cfg_one_point():
    # ROI가 합성 채널(y≈50)과 blob(y≈65) 모두 덮도록: 중심 (150,57.5), angle=0, 폭축=width×frame_h
    return MultiwayConfig(
        RecordingSpec(300, 200, 300.0, "dma"),
        (PointSpec(1, 150 / 300, 57.5 / 200, 60 / 300, 90 / 200, 0.0),),
        BubbleSizeSpec("circle", 24.0, 24.0))


def test_frame_stats_median_and_std():
    frames = _synthetic()
    med, std = foreign.frame_stats(frames)
    assert med.dtype == np.uint8 and std.dtype == np.float32
    assert med[70, 150] < 80                  # 정지 blob(y=70)은 중앙값에 잔존(어두움)
    assert std[50, 150] > foreign.ACT_THR     # 채널 경로(y=50)는 고활성
    assert std[20, 20] < 4.0                   # 배경은 정적
    with pytest.raises(ValueError):
        foreign.frame_stats([])


def test_scan_detects_blob_on_channel():
    med, std = foreign.frame_stats(_synthetic())
    out = foreign.scan_stats(med, std, _cfg_one_point(), source="review-scan")
    assert len(out) == 1
    fo = out[0]
    assert abs(fo.x_px - 150) <= 4 and abs(fo.y_px - 64) <= 4  # blob at y≈65, detected at y≈63.6
    assert fo.source == "review-scan" and fo.tag == ""
    assert fo.near_points == (1,)              # ROI(중심 150,57.5) 밴드±마진 안
    assert fo.id == 0


def test_scan_ignores_blob_without_channel():
    med, std = foreign.frame_stats(_synthetic(with_channel=False))
    assert foreign.scan_stats(med, std, _cfg_one_point(), source="review-scan") == []


def test_scan_ignores_wall_line():
    med, std = foreign.frame_stats(_synthetic(with_wall=True))
    out = foreign.scan_stats(med, std, _cfg_one_point(), source="review-scan")
    assert all(abs(fo.x_px - 42) > 10 for fo in out)   # 벽선(x=40~44) 미검출


def test_point_band_x_identity_mapping():
    # angle=0·+x에서 ROI 중심의 프레임 좌표는 패치 중심으로 사상된다(warp 중심 규약)
    cfg = _cfg_one_point()
    p = cfg.points[0]
    fo = ForeignObject(0, 150.0, 57.5, 8.0, 8.0, 50.0, (), "preflight")
    inside, x_p = foreign.point_band_x(fo, p, 300, 200, "+x")
    assert inside is True
    w_p = max(1, round(p.width * 200))          # warp.patch_dims와 동일 정의
    assert x_p == pytest.approx((w_p - 1) / 2.0, abs=1.0)


def test_point_band_x_outside():
    cfg = _cfg_one_point()
    fo = ForeignObject(0, 20.0, 20.0, 8.0, 8.0, 50.0, (), "preflight")
    inside, _ = foreign.point_band_x(fo, cfg.points[0], 300, 200, "+x")
    assert inside is False