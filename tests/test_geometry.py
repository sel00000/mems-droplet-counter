import numpy as np
import pytest

from bubble_counter.geometry import extract_band_row, line_center_px, strip_bounds


class TestLineCenterPx:
    def test_midline_rounds_half_to_even(self):
        # round(0.5 * 479) == round(239.5) == 240
        assert line_center_px(480, 0.5) == 240

    def test_pos_zero_gives_first_index(self):
        assert line_center_px(10, 0.0) == 0

    def test_pos_one_gives_last_index(self):
        assert line_center_px(10, 1.0) == 9

    def test_single_pixel_line_stays_in_bounds(self):
        assert line_center_px(1, 0.5) == 0


class TestStripBounds:
    def test_hand_calculated_interior_case(self):
        # band = [50 - 5//2, +5) = [48, 53); strip = band +-5 = [43, 58)
        assert strip_bounds(100, 50, 5, 5) == (43, 58, 5, 10)

    def test_low_edge_clips_band_and_strip(self):
        # band_lo_abs = 1 - 5//2 = -1 -> clipped 0; band_hi_abs = -1+5 = 4
        # band = [0, 4); strip = [0-5, 4+5) clipped = [0, 9)
        # band relative to strip_lo(0): [0, 4)
        assert strip_bounds(100, 1, 5, 5) == (0, 9, 0, 4)

    def test_center_at_zero_band_not_empty(self):
        # band_lo_abs = 0-2=-2 -> clip 0; band_hi_abs = -2+5=3
        # band = [0,3); strip = [0-5,3+5) clipped = [0,8); band rel = [0,3)
        result = strip_bounds(100, 0, 5, 5)
        assert result == (0, 8, 0, 3)
        strip_lo, strip_hi, band_lo, band_hi = result
        assert band_hi > band_lo

    def test_center_at_last_index_band_not_empty(self):
        # roi_len=100, center=99 (roi_len-1)
        # band_lo_abs=99-2=97; band_hi_abs=97+5=102 -> clip 100; band=[97,100)
        # strip_lo_abs=97-5=92; strip_hi_abs=100+5=105 -> clip 100; strip=[92,100)
        # band rel to strip_lo(92): [5,8)
        result = strip_bounds(100, 99, 5, 5)
        assert result == (92, 100, 5, 8)
        strip_lo, strip_hi, band_lo, band_hi = result
        assert band_hi > band_lo

    def test_guarantees_hold_for_interior_case(self):
        roi_len = 100
        strip_lo, strip_hi, band_lo, band_hi = strip_bounds(roi_len, 50, 5, 5)
        assert 0 <= strip_lo < strip_hi <= roi_len
        assert 0 <= band_lo < band_hi <= strip_hi - strip_lo

    def test_band_entirely_outside_raises_value_error(self):
        # center far negative -> both band bounds clip to 0 -> empty band
        with pytest.raises(ValueError):
            strip_bounds(100, -1000, 5, 5)

    def test_band_entirely_beyond_roi_raises_value_error(self):
        # center far beyond roi_len -> both band bounds clip to roi_len -> empty band
        with pytest.raises(ValueError):
            strip_bounds(100, 1000, 5, 5)


class TestExtractBandRow:
    def test_bool_dtype_input_returns_bool_column_max(self):
        strip = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        row = extract_band_row(strip, 0, 2)  # rows 0,1 only
        assert row.dtype == np.bool_
        np.testing.assert_array_equal(row, np.array([True, False, True]))

    def test_uint8_0_255_input_supported(self):
        strip = np.array(
            [
                [0, 255, 0],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        row = extract_band_row(strip, 0, 2)
        assert row.dtype == np.bool_
        np.testing.assert_array_equal(row, np.array([False, True, True]))

    def test_band_subset_of_taller_strip(self):
        # only rows [1,3) participate; row 0 and row 3 must be ignored
        strip = np.array(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )
        row = extract_band_row(strip, 1, 3)
        np.testing.assert_array_equal(row, np.array([False, True]))
