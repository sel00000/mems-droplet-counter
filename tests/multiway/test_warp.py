"""multiway/warp.py — 회전 패치 어파인·추출·각도 정규화."""

import math

import cv2
import numpy as np
import pytest

from bubble_counter.multiway.model import PointSpec
from bubble_counter.multiway.warp import (
    extract_patch, fast_slice_patch, normalize_rect_angle, patch_affine, patch_dims,
)


def _unique_gray(h=100, w=100):
    return (np.arange(h * w).reshape(h, w) % 251).astype(np.uint8)


class TestPatchDims:
    def test_dims_use_frame_h_for_width_and_frame_w_for_length(self):
        # width 0.07×frame_h(100)=7(폭축 W_p), length 0.11×frame_w(100)=11(흐름축 H_p)
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=0.0)
        assert patch_dims(p, 100, 100) == (7, 11)


class TestPixelExactAngleZero:
    """픽셀 정확도 (2): angle=0 패치 == 순수 슬라이싱 (INTER_NEAREST 비트 일치)."""

    def test_angle0_plus_x_equals_transpose_slice(self):
        gray = _unique_gray()
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=0.0)
        W_p, H_p = patch_dims(p, 100, 100)          # (7, 11)
        M = patch_affine(p, 100, 100, "+x")
        patch = extract_patch(gray, M, W_p, H_p, interp=cv2.INTER_NEAREST)
        # 유도: row0=cy-(W_p-1)/2=47, col0=cx-(H_p-1)/2=45, patch=gray[47:54,45:56].T
        expect = gray[47:47 + W_p, 45:45 + H_p].T
        assert patch.shape == (H_p, W_p)
        assert np.array_equal(patch, expect)

    def test_fast_slice_matches_warp_bit_for_bit(self):
        gray = _unique_gray()
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=0.0)
        W_p, H_p = patch_dims(p, 100, 100)
        M = patch_affine(p, 100, 100, "+x")
        warp = extract_patch(gray, M, W_p, H_p, interp=cv2.INTER_NEAREST)
        fast = fast_slice_patch(gray, p, 100, 100, "+x")
        assert fast is not None
        assert np.array_equal(fast, warp)

    def test_fast_slice_minus_x_flips_flow_axis(self):
        gray = _unique_gray()
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=0.0)
        M = patch_affine(p, 100, 100, "-x")
        W_p, H_p = patch_dims(p, 100, 100)
        warp = extract_patch(gray, M, W_p, H_p, interp=cv2.INTER_NEAREST)
        fast = fast_slice_patch(gray, p, 100, 100, "-x")
        assert np.array_equal(fast, warp)

    def test_fast_slice_returns_none_for_rotated(self):
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=30.0)
        assert fast_slice_patch(_unique_gray(), p, 100, 100, "+x") is None


class TestPixelExactAngle90:
    """픽셀 정확도 (1): angle=90 패치 == 슬라이스 상하반전 (INTER_NEAREST 비트 일치)."""

    def test_angle90_minus_y_equals_flipud_slice(self):
        gray = _unique_gray()
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=90.0)
        W_p, H_p = patch_dims(p, 100, 100)          # (7, 11)
        M = patch_affine(p, 100, 100, "-y")
        patch = extract_patch(gray, M, W_p, H_p, interp=cv2.INTER_NEAREST)
        # 유도: col_src=x_p+47, row_src=55-y_p → gray[45:56,47:54] 상하반전
        expect = np.flipud(gray[45:45 + H_p, 47:47 + W_p])
        assert patch.shape == (H_p, W_p)
        assert np.array_equal(patch, expect)


class TestDownstreamFlip:
    def test_plus_y_patch_row_is_downstream(self):
        # 흐름축 아래로: 패치 마지막 행(y_p=H_p-1)이 프레임에서 더 하류(+y=아래)여야
        gray = _unique_gray()
        p = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=0.0, direction="+y")
        # angle 0에서 흐름벡터 u=(1,0); direction +y=(0,1) → 내적 0 (모호) — 아래 방향 케이스는 angle로 검증
        p2 = PointSpec(1, 0.5, 0.5, length=0.11, width=0.07, angle_deg=-90.0)  # u=(0,1) 아래
        M_dn = patch_affine(p2, 100, 100, "+y")   # 하류 +y, u와 동일 → 뒤집기 없음
        M_up = patch_affine(p2, 100, 100, "-y")   # 하류 -y, u와 반대 → 뒤집기
        # 두 행렬의 흐름축(2열)이 부호 반대여야 한다
        assert np.sign(M_dn[1, 1]) == -np.sign(M_up[1, 1])


class TestNormalizeRectAngle:
    def test_long_horizontal_is_zero(self):
        rect = ((50, 50), (80, 20), 0.0)
        assert normalize_rect_angle(rect) == pytest.approx(0.0, abs=1e-6)

    def test_long_vertical_is_minus_90(self):
        rect = ((50, 50), (20, 80), 0.0)
        assert normalize_rect_angle(rect) == pytest.approx(-90.0, abs=1e-6)

    def test_opencv_positive_angle_maps_to_screen_up_negative(self):
        # OpenCV의 +30(화면 시계방향)은 우리 규약(+x기준 위=양수)에서 -30
        assert normalize_rect_angle(((50, 50), (80, 20), 30.0)) == pytest.approx(-30.0, abs=1e-4)
        assert normalize_rect_angle(((50, 50), (80, 20), -30.0)) == pytest.approx(30.0, abs=1e-4)

    def test_square_45(self):
        assert normalize_rect_angle(((50, 50), (40, 40), 45.0)) == pytest.approx(45.0, abs=1e-4)

    def test_roundtrip_our_plus_30(self):
        # 우리 규약 +30(위로 30°)으로 만든 사각형 → 정규화가 +30 복원
        a = math.radians(30.0)
        u = np.array([math.cos(a), -math.sin(a)])
        n = np.array([math.sin(a), math.cos(a)])
        c = np.array([50.0, 50.0])
        hl, hw = 40.0, 10.0
        corners = np.array(
            [c - hl * u - hw * n, c - hl * u + hw * n, c + hl * u + hw * n, c + hl * u - hw * n],
            dtype=np.float32,
        )
        rect = cv2.minAreaRect(corners)
        assert normalize_rect_angle(rect) == pytest.approx(30.0, abs=1e-3)
