"""gui_settings 단위 테스트 — 영속화 라운드트립·sanitize·비율 환산 (tkinter 불필요)."""

from __future__ import annotations

import json

import pytest

from bubble_counter.gui_settings import (
    band_px_for,
    frac_for,
    load_settings,
    save_settings,
)


def _full_settings() -> dict:
    return {
        "axis": "h", "line_pos": 0.95, "band_px": 64, "band_frac": 64 / 480,
        "band_auto": True, "scale": 1.0, "bucket_seconds": 1.0,
        "var_threshold": 16.0, "merge_gap_frames": 1,
        "annotate": False, "save_rhythm": True, "last_video_dir": "C:/실험영상",
    }


class TestPersistence:
    def test_round_trip_restores_every_field(self, tmp_path):
        path = tmp_path / "s.json"
        assert save_settings(_full_settings(), path) is True
        assert load_settings(path) == _full_settings()

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_settings(tmp_path / "none.json") == {}

    def test_corrupt_json_returns_empty(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{잘림", encoding="utf-8")
        assert load_settings(path) == {}

    def test_non_dict_json_returns_empty(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("[1, 2]", encoding="utf-8")
        assert load_settings(path) == {}

    def test_save_failure_returns_false_not_raise(self, tmp_path):
        assert save_settings({}, tmp_path / "없는폴더" / "s.json") is False

    def test_version_key_is_written(self, tmp_path):
        path = tmp_path / "s.json"
        save_settings({"axis": "h"}, path)
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

    def test_load_strips_utf8_bom(self, tmp_path):
        # 메모장이 붙이는 UTF-8 BOM이 있어도 로드가 {}로 초기화되면 안 된다
        path = tmp_path / "s.json"
        body = json.dumps({"version": 1, "axis": "h", "band_px": 64})
        path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
        assert load_settings(path) == {"axis": "h", "band_px": 64}

    def test_save_writes_no_bom(self, tmp_path):
        # 저장은 무BOM utf-8 유지 — BOM을 새로 만들어내지 않는다
        path = tmp_path / "s.json"
        save_settings({"axis": "h"}, path)
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_save_leaves_no_temp_file(self, tmp_path):
        # 원자적 저장: 성공 후 임시(.tmp) 파일이 남지 않아야 한다
        path = tmp_path / "s.json"
        assert save_settings(_full_settings(), path) is True
        assert list(tmp_path.glob("*.tmp")) == []
        assert load_settings(path) == _full_settings()

    def test_failed_save_preserves_existing_file(self, tmp_path, monkeypatch):
        # 원자적 저장: 교체(os.replace) 중 실패해도 기존 파일이 손상되지 않고
        # 임시 파일도 정리되어야 한다
        path = tmp_path / "s.json"
        good = {"axis": "h", "band_px": 64}
        assert save_settings(good, path) is True

        def boom(*args, **kwargs):
            raise OSError("replace failed")
        monkeypatch.setattr("os.replace", boom)

        assert save_settings({"axis": "v", "band_px": 128}, path) is False
        assert load_settings(path) == good          # 기존 내용 보존
        assert list(tmp_path.glob("*.tmp")) == []   # 실패 시 임시 파일 정리


class TestSanitize:
    def test_unknown_keys_dropped_including_fps_override(self, tmp_path):
        path = tmp_path / "s.json"
        save_settings({"axis": "h", "fps_override": 30.0, "정체불명": 1}, path)
        assert load_settings(path) == {"axis": "h"}

    @pytest.mark.parametrize("key,bad", [
        ("axis", "x"), ("line_pos", 1.5), ("line_pos", "0.5"),
        ("band_px", 0), ("band_px", 513), ("band_px", 64.5), ("band_px", True),
        ("scale", 0.05), ("bucket_seconds", 0), ("var_threshold", 501),
        ("merge_gap_frames", -1), ("band_frac", 0), ("band_frac", 1.5),
        ("band_auto", 1), ("last_video_dir", 123),
    ])
    def test_invalid_value_dropped_not_clamped(self, tmp_path, key, bad):
        path = tmp_path / "s.json"
        save_settings({key: bad}, path)
        assert key not in load_settings(path)

    def test_band_frac_null_is_preserved(self, tmp_path):
        # null = "아직 비율 미채택" 상태 (스펙 §3.2) — 키 자체는 살아남아야 한다
        path = tmp_path / "s.json"
        save_settings({"band_frac": None}, path)
        assert load_settings(path) == {"band_frac": None}

    def test_integral_float_accepted_for_int_keys(self, tmp_path):
        # 손으로 고친 JSON의 64.0 같은 값은 받아준다 (64.5는 위에서 거부 확인)
        path = tmp_path / "s.json"
        save_settings({"band_px": 64.0, "merge_gap_frames": 2.0}, path)
        assert load_settings(path) == {"band_px": 64, "merge_gap_frames": 2}


class TestBandConversion:
    def test_demo_ratio_round_trips_at_same_resolution(self):
        frac = frac_for(64, 480, 1.0)
        assert band_px_for(frac, 480, 1.0) == 64

    def test_ratio_scales_linearly_between_resolutions(self):
        frac = frac_for(64, 480, 1.0)          # 480p에서 64px였다면
        assert band_px_for(frac, 720, 1.0) == 96    # 720p → 96
        assert band_px_for(frac, 1080, 1.0) == 144  # 1080p → 144

    def test_scale_factor_shrinks_processing_length(self):
        # band_px는 처리 스케일 픽셀이므로 scale이 곱해진 길이 기준 (스펙 §4)
        frac = frac_for(64, 480, 1.0)
        assert band_px_for(frac, 480, 0.5) == 32

    def test_result_clamped_to_widget_range(self):
        assert band_px_for(1.0, 4000, 1.0) == 512
        assert band_px_for(0.0001, 480, 1.0) == 1

    def test_tiny_axis_processing_length_floors_at_one(self):
        # VideoSource.out_size와 동일한 max(1, round(len*scale)) 규칙
        assert frac_for(1, 3, 0.1) == 1.0

    def test_frac_clamped_when_band_exceeds_processing_length(self):
        # 병리 케이스: 480p·scale0.1 → 처리길이 48인데 band 64 → frac은 1.0으로 클램프
        assert frac_for(64, 480, 0.1) == 1.0

    def test_frac_unchanged_for_normal_band(self):
        # 정상 케이스(band ≤ 처리길이)는 클램프 영향 없이 라운드트립 보존
        frac = frac_for(64, 480, 1.0)
        assert frac == pytest.approx(64 / 480)
        assert band_px_for(frac, 480, 1.0) == 64

    def test_clamped_frac_survives_sanitize_round_trip(self, tmp_path):
        # 클램프된 frac은 (0,1]이라 sanitize가 드롭하지 않는다 (병리 입력의 실제 버그)
        frac = frac_for(64, 480, 0.1)
        path = tmp_path / "s.json"
        save_settings({"band_frac": frac}, path)
        assert load_settings(path) == {"band_frac": frac}
