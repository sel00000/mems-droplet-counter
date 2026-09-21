"""profiles 순수 직렬화 테스트 — config↔dict 라운드트립 (설계 §1.4)."""

from __future__ import annotations

import json

import pytest

from bubble_counter.config import CounterConfig
from bubble_counter.multiway.model import (
    BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec,
)
from bubble_counter.multiway.profiles import config_to_profile, profile_to_config


def _bubble():
    return BubbleSizeSpec("circle", 50.0, 50.0)


def _config(**over):
    pts = over.pop("points", (
        PointSpec(1, 0.742, 0.918, 0.060, 0.035, -32.5),
        PointSpec(2, 0.300, 0.500, 0.060, 0.035, 0.0, direction="-y", target=400),
    ))
    return MultiwayConfig(
        recording=RecordingSpec(1280, 768, 300.0, "dma"),
        points=pts, bubble=_bubble(),
        global_direction=over.pop("global_direction", "+x"),
        order_tolerance_frames=over.pop("order_tolerance_frames", 0),
    )


def test_config_to_profile_shape():
    d = config_to_profile(_config())
    assert set(d) == {"recording", "global_direction", "order_tolerance_frames",
                      "order_gate", "points", "bubble"}
    assert "base" not in d and "version" not in d
    assert d["recording"] == {"width": 1280, "height": 768, "fps": 300.0, "mode": "dma"}
    assert set(d["bubble"]) == {"shape", "major_px", "minor_px", "accept_low",
                                "accept_high", "count_half_bubbles"}
    assert d["bubble"]["shape"] == "circle" and d["bubble"]["major_px"] == 50.0
    p0 = d["points"][0]
    assert set(p0) == {"number", "cx", "cy", "length", "width", "angle_deg",
                       "direction", "target"}
    assert p0["number"] == 1 and p0["angle_deg"] == -32.5 and p0["direction"] is None
    assert d["points"][1]["direction"] == "-y" and d["points"][1]["target"] == 400


def test_round_trip_identity():
    cfg = _config()
    assert profile_to_config(config_to_profile(cfg), cfg.base) == cfg   # bubble도 왕복


def test_bubble_none_round_trip():
    # D-13: 기포 크기 미지정(None) 프로파일 저장→로드 왕복 허용(null 직렬화).
    import dataclasses
    cfg = dataclasses.replace(_config(), bubble=None)
    d = config_to_profile(cfg)
    assert d["bubble"] is None
    assert json.loads(json.dumps(d))["bubble"] is None      # JSON 직렬화 안전
    assert profile_to_config(d, cfg.base).bubble is None     # 로드 왕복도 None


def test_round_trip_survives_json():
    # 실제 저장은 JSON을 거치므로 float/None이 왕복해도 동일해야 한다.
    cfg = _config()
    layout = json.loads(json.dumps(config_to_profile(cfg)))
    assert profile_to_config(layout, cfg.base) == cfg


def test_bubble_round_trips_ellipse_and_toggle():
    cfg = _config()
    cfg = MultiwayConfig(cfg.recording, cfg.points,
                         BubbleSizeSpec("ellipse", 70.0, 55.0, count_half_bubbles=True))
    back = profile_to_config(config_to_profile(cfg))
    assert back.bubble == cfg.bubble


def test_profile_to_config_default_base_when_none():
    cfg = _config()
    back = profile_to_config(config_to_profile(cfg))   # base=None
    assert back.base == CounterConfig()


def test_profile_to_config_rejects_bad_point():
    layout = config_to_profile(_config())
    layout["points"][0]["cx"] = 1.5           # 범위 밖
    with pytest.raises(ValueError, match="cx"):
        profile_to_config(layout)


def test_profile_to_config_rejects_number_gap():
    # {1, 3} → MultiwayConfig 중앙 검증에서 차단
    layout = config_to_profile(_config())
    layout["points"][1]["number"] = 3
    with pytest.raises(ValueError):
        profile_to_config(layout)


from bubble_counter.multiway.profiles import ProfileStore


def _store(tmp_path):
    return ProfileStore(tmp_path / "p.json", now=lambda: "2026-07-09T18:00:00")


class TestProfileStore:
    def test_save_then_load_round_trips(self, tmp_path):
        store = _store(tmp_path)
        cfg = _config()
        assert store.save("6way 칩 A", cfg) is None
        assert store.load("6way 칩 A") == cfg          # bubble 포함 완전 복원
        disk = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
        assert disk["profiles"]["6way 칩 A"]["saved_at"] == "2026-07-09T18:00:00"

    def test_names_sorted(self, tmp_path):
        store = _store(tmp_path)
        store.save("B칩", _config())
        store.save("A칩", _config())
        assert store.names() == ["A칩", "B칩"]

    def test_save_preserves_other_profiles(self, tmp_path):
        store = _store(tmp_path)
        store.save("A칩", _config())
        store.save("B칩", _config())
        assert set(store.names()) == {"A칩", "B칩"}   # 두 번째 저장이 첫째를 지우지 않음

    def test_last_used_tracks_latest_save(self, tmp_path):
        store = _store(tmp_path)
        store.save("A칩", _config())
        store.save("B칩", _config())
        assert store.last_used() == "B칩"

    def test_load_unknown_raises_keyerror(self, tmp_path):
        with pytest.raises(KeyError, match="없음"):
            _store(tmp_path).load("없음")

    def test_missing_file_is_empty(self, tmp_path):
        store = ProfileStore(tmp_path / "none.json")
        assert store.names() == [] and store.last_used() is None
        with pytest.raises(KeyError):
            store.load("x")

    def test_corrupt_file_falls_back_empty(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{잘림", encoding="utf-8")
        store = ProfileStore(path)
        assert store.names() == []
        with pytest.raises(KeyError):
            store.load("x")

    def test_load_strips_utf8_bom(self, tmp_path):
        # 메모장 BOM이 있어도 로드가 빈 상태로 초기화되면 안 된다
        path = tmp_path / "p.json"
        store = ProfileStore(path, now=lambda: "2026-07-09T18:00:00")
        store.save("A칩", _config())
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        assert store.names() == ["A칩"]

    def test_save_leaves_no_temp_file(self, tmp_path):
        store = _store(tmp_path)
        assert store.save("A칩", _config()) is None
        assert list(tmp_path.glob("*.tmp")) == []

    def test_save_failure_raises_ioerror(self, tmp_path):
        store = ProfileStore(tmp_path / "없는폴더" / "p.json")
        with pytest.raises((IOError, OSError)):
            store.save("A칩", _config())
