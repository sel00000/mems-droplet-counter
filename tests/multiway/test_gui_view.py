"""gui_view 단위 테스트 — tkinter 없이 뷰 변환/표시 계층을 검증한다."""

import numpy as np
import pytest

from bubble_counter.gui_view import ViewTransform, ppm_p6_bytes


class TestViewTransformCore:
    def test_fit_centers_and_scales(self):
        # scale = min(400/1000, 400/500) = 0.4 ; 세로 여백을 위/아래로 나눔
        vt = ViewTransform.fit(image_w=1000, image_h=500, canvas_w=400, canvas_h=400)
        assert vt.scale == pytest.approx(0.4)
        assert vt.tx == pytest.approx(0.0)                 # (400 - 0.4*1000)/2 = 0
        assert vt.ty == pytest.approx(100.0)               # (400 - 0.4*500)/2 = 100

    def test_screen_image_roundtrip(self):
        vt = ViewTransform(scale=2.0, tx=10.0, ty=20.0)
        assert vt.image_to_screen(5.0, 7.0) == pytest.approx((20.0, 34.0))
        assert vt.screen_to_image(20.0, 34.0) == pytest.approx((5.0, 7.0))

    def test_matrix_is_3x3_affine(self):
        vt = ViewTransform(scale=2.0, tx=10.0, ty=20.0)
        assert vt.matrix == (
            (2.0, 0.0, 10.0),
            (0.0, 2.0, 20.0),
            (0.0, 0.0, 1.0),
        )


class TestViewTransformZoomPan:
    def test_zoom_at_keeps_cursor_point_fixed(self):
        vt = ViewTransform(scale=1.0, tx=0.0, ty=0.0, min_scale=0.1, max_scale=10.0)
        before = vt.screen_to_image(300, 200)
        vt.zoom_at(300, 200, 2.0)
        assert vt.scale == pytest.approx(2.0)
        assert vt.screen_to_image(300, 200) == pytest.approx(before)  # 커서 아래 점 불변

    def test_zoom_at_clamps_scale(self):
        vt = ViewTransform(scale=1.0, tx=0.0, ty=0.0, min_scale=0.5, max_scale=4.0)
        vt.zoom_at(0, 0, 100.0)
        assert vt.scale == pytest.approx(4.0)
        vt.zoom_at(0, 0, 0.0001)
        assert vt.scale == pytest.approx(0.5)

    def test_pan_translates(self):
        vt = ViewTransform(scale=1.0, tx=0.0, ty=0.0)
        vt.pan(15.0, -5.0)
        assert (vt.tx, vt.ty) == pytest.approx((15.0, -5.0))

    def test_clamp_pan_keeps_image_partly_visible(self):
        vt = ViewTransform(scale=1.0, tx=5000.0, ty=-9000.0)
        vt.clamp_pan(image_w=1000, image_h=800, canvas_w=400, canvas_h=400, margin=50)
        # tx <= canvas_w - margin = 350 ; tx >= margin - scale*image_w = 50 - 1000 = -950
        assert -950 - 1e-6 <= vt.tx <= 350 + 1e-6
        assert -750 - 1e-6 <= vt.ty <= 350 + 1e-6   # margin - scale*image_h = 50 - 800 = -750


class TestPpmBytes:
    def test_header_and_body(self):
        # 1행 2열: 빨강, 초록
        rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        data = ppm_p6_bytes(rgb)
        assert data.startswith(b"P6\n2 1\n255\n")        # width=2 height=1
        body = data[len(b"P6\n2 1\n255\n"):]
        assert body == bytes([255, 0, 0, 0, 255, 0])     # 행 우선 RGB

    def test_rejects_non_uint8(self):
        with pytest.raises(ValueError, match="uint8"):
            ppm_p6_bytes(np.zeros((2, 2, 3), dtype=np.float32))

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="RGB"):
            ppm_p6_bytes(np.zeros((2, 2, 4), dtype=np.uint8))
