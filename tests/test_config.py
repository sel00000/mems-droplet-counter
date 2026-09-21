import pytest

from bubble_counter.config import CounterConfig, LineSpec, RoiSpec


class TestRoiSpecValidation:
    def test_x_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(x=1.2)

    def test_y_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(y=1.0)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(w=0)

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(h=0)

    def test_w_over_one_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(w=1.1)

    def test_x_plus_w_over_one_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(x=0.6, w=0.5)

    def test_y_plus_h_over_one_raises(self):
        with pytest.raises(ValueError):
            RoiSpec(y=0.6, h=0.5)

    def test_boundary_sum_exactly_one_is_valid(self):
        roi = RoiSpec(x=0.5, y=0.5, w=0.5, h=0.5)
        assert roi.x == 0.5 and roi.w == 0.5

    def test_roi_x_error_message_is_korean(self):
        with pytest.raises(ValueError, match="범위여야 합니다"):
            RoiSpec(x=1.5)


class TestRoiSpecToPixels:
    def test_default_full_frame(self):
        assert RoiSpec().to_pixels(640, 480) == (0, 0, 640, 480)

    def test_quarter_offset_half_size(self):
        assert RoiSpec(0.25, 0.25, 0.5, 0.5).to_pixels(640, 480) == (160, 120, 320, 240)

    def test_tiny_roi_guarantees_minimum_one_pixel(self):
        x0, y0, w_px, h_px = RoiSpec(x=0.0, y=0.0, w=0.001, h=0.001).to_pixels(100, 100)
        assert w_px >= 1
        assert h_px >= 1

    def test_tiny_roi_stays_within_frame_bounds(self):
        x0, y0, w_px, h_px = RoiSpec(x=0.999, y=0.999, w=0.001, h=0.001).to_pixels(640, 480)
        assert x0 + w_px <= 640
        assert y0 + h_px <= 480


class TestLineSpecValidation:
    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            LineSpec(axis="x")

    def test_pos_below_zero_raises(self):
        with pytest.raises(ValueError):
            LineSpec(pos=-0.1)

    def test_pos_above_one_raises(self):
        with pytest.raises(ValueError):
            LineSpec(pos=1.5)

    def test_band_px_zero_raises(self):
        with pytest.raises(ValueError):
            LineSpec(band_px=0)

    def test_valid_vertical_axis_accepted(self):
        line = LineSpec(axis="v")
        assert line.axis == "v"

    def test_band_px_error_message_is_korean(self):
        with pytest.raises(ValueError, match="1 이상이어야 합니다"):
            LineSpec(band_px=0)


class TestCounterConfigDefaults:
    def test_defaults_smoke(self):
        cfg = CounterConfig()
        assert cfg.roi == RoiSpec()
        assert cfg.line == LineSpec()
        assert cfg.scale == 1.0
        assert cfg.history == 300
        assert cfg.var_threshold == 16.0
        assert cfg.morph_kernel == 3
        assert cfg.row_close_px == 3
        assert cfg.merge_gap_frames == 1
        assert cfg.min_width_px == 2
        assert cfg.min_area_px == 4
        assert cfg.warmup_frames == 60
        assert cfg.bucket_seconds == 1.0
        assert cfg.fps_override is None
        assert cfg.annotate is False
        assert cfg.save_rhythm is False
        assert cfg.max_frames is None


class TestCounterConfigValidation:
    def test_default_config_is_valid(self):
        # 기본값(bucket=1.0, fps=None)은 __post_init__ 검증을 통과해야 한다.
        assert CounterConfig().bucket_seconds == 1.0

    @pytest.mark.parametrize("bad", [0, -1, float("nan")])
    def test_bucket_seconds_non_positive_or_nan_raises(self, bad):
        with pytest.raises(ValueError, match="bucket_seconds"):
            CounterConfig(bucket_seconds=bad)

    def test_bucket_seconds_message_matches_reporting_guard(self):
        with pytest.raises(ValueError, match="0보다 커야 합니다"):
            CounterConfig(bucket_seconds=0)

    def test_bucket_seconds_positive_passes(self):
        assert CounterConfig(bucket_seconds=2.5).bucket_seconds == 2.5

    @pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0, -1])
    def test_fps_override_non_finite_or_non_positive_raises(self, bad):
        with pytest.raises(ValueError, match="fps_override"):
            CounterConfig(fps_override=bad)

    def test_fps_override_none_passes(self):
        assert CounterConfig(fps_override=None).fps_override is None

    def test_fps_override_positive_finite_passes(self):
        assert CounterConfig(fps_override=29.97).fps_override == 29.97
