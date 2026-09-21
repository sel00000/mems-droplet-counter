"""gui_params 단위 테스트 — tkinter 없이 GUI 입력 정책을 검증한다."""

from __future__ import annotations

import pytest

from bubble_counter.gui_params import validate_params


def _ok(**overrides):
    base = dict(band_px=5, bucket_seconds=1.0, var_threshold=16.0,
                merge_gap_frames=1, fps_text="")
    base.update(overrides)
    return validate_params(**base)


class TestValidValues:
    def test_defaults_pass_and_blank_fps_is_none(self):
        assert _ok() is None

    def test_fps_text_parses_to_float(self):
        assert _ok(fps_text="24") == 24.0

    def test_fps_text_strips_whitespace(self):
        assert _ok(fps_text=" 29.97 ") == 29.97

    def test_fps_text_positive_integer_string_passes(self):
        assert _ok(fps_text="30") == 30.0

    @pytest.mark.parametrize("field,value", [
        ("band_px", 1), ("band_px", 512),
        ("bucket_seconds", 0.1), ("bucket_seconds", 3600.0),
        ("var_threshold", 1.0), ("var_threshold", 500.0),
        ("merge_gap_frames", 0), ("merge_gap_frames", 1000),
    ])
    def test_boundary_values_accepted(self, field, value):
        _ok(**{field: value})


class TestRejectedValues:
    @pytest.mark.parametrize("field,value,fragment", [
        ("band_px", 0, "밴드 폭"),
        ("band_px", 513, "밴드 폭"),
        ("bucket_seconds", 0.0, "시간 버킷"),
        ("bucket_seconds", 3601.0, "시간 버킷"),
        ("var_threshold", 0.5, "var-threshold"),
        ("var_threshold", 501.0, "var-threshold"),
        ("merge_gap_frames", -1, "merge-gap"),
        ("merge_gap_frames", 1001, "merge-gap"),
    ])
    def test_out_of_range_raises_korean_message_naming_the_field(
            self, field, value, fragment):
        with pytest.raises(ValueError, match=fragment):
            _ok(**{field: value})

    def test_fps_text_not_a_number_raises(self):
        with pytest.raises(ValueError, match="fps 강제"):
            _ok(fps_text="abc")

    def test_fps_text_nonpositive_raises(self):
        with pytest.raises(ValueError, match="fps 강제"):
            _ok(fps_text="-5")

    @pytest.mark.parametrize("text", ["inf", "-inf", "nan", "Infinity"])
    def test_fps_text_non_finite_raises(self, text):
        with pytest.raises(ValueError, match="유한"):
            _ok(fps_text=text)
