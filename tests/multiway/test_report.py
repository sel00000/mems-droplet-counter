"""report CSV 테스트 — 컬럼·time_s 파생·인용·상한 (설계 §1.5)."""

from __future__ import annotations

import csv
import json

import pytest

from bubble_counter.multiway.model import (
    BubbleEvent, BubbleSizeSpec, ChapterResult, Correction, MultiwayConfig,
    PhaseResult, PointSpec, RecordingSpec, SequenceViolation, SuspectMoment,
)
from bubble_counter.multiway import report


def _config():
    return MultiwayConfig(
        recording=RecordingSpec(1280, 768, 300.0, "dma"),
        points=(PointSpec(1, 0.3, 0.5, 0.05, 0.05, 0.0),
                PointSpec(2, 0.6, 0.5, 0.05, 0.05, 0.0)),
        bubble=BubbleSizeSpec("circle", 50.0, 50.0),
    )


def _chapter(**over):
    base = dict(
        video_path="6way_1_1.avi", recording=RecordingSpec(1280, 768, 300.0, "dma"),
        counts_auto={1: 436, 2: 500},
        corrections=[Correction(1, 8432, +1, "겹침 확인", "2026-07-10T00:00:00")],
        events=[BubbleEvent(1, 300, 0.25, ("merge-suspect",)), BubbleEvent(2, 150, 0.0)],
        violations=[SequenceViolation(8432, 4, 5)],
        suspects=[SuspectMoment("merge", 5, 8432, "기포 2개 겹침, 확인 요망",
                                "suspects/s0001_p5_f008432.png")],
        sets_completed=431, processed_frames=12339,
    )
    base.update(over)
    return ChapterResult(**base)


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_time_s_derived_from_recording_fps():
    assert report.time_s(300, 300.0) == pytest.approx(1.0)
    assert report.time_s(8432, 300.0) == pytest.approx(28.106667, abs=1e-6)


def test_point_counts_csv(tmp_path):
    path = tmp_path / "point_counts.csv"
    report.write_point_counts_csv(path, _chapter(), _config())
    rows = _rows(path)
    assert rows[0] == ["point", "count_auto", "corrections", "count_final", "target", "error"]
    # p1: auto 436, +1 보정 → final 437, target 500, error -63, corrections 1
    assert rows[1] == ["1", "436", "1", "437", "500", "-63"]
    assert rows[2] == ["2", "500", "0", "500", "500", "0"]


def test_events_csv(tmp_path):
    path = tmp_path / "events.csv"
    ch = _chapter(events=[
        BubbleEvent(1, 300, 0.25, ("merge-suspect",), position_px=37.5, size_px=48.0),
        BubbleEvent(2, 150, 0.0, position_px=52.0, size_px=45.0),
    ])
    report.write_events_csv(path, ch)
    rows = _rows(path)
    assert rows[0] == ["point", "frame", "subframe", "time_s", "flags", "position_px", "size_px"]
    # (frame, subframe, point) 정렬 → frame 150 먼저
    assert rows[1] == ["2", "150", "0.0000", "0.500", "", "52.0", "45.0"]
    assert rows[2] == ["1", "300", "0.2500", "1.000", "merge-suspect", "37.5", "48.0"]


def test_events_csv_legacy_sentinel(tmp_path):
    ch = _chapter(events=[BubbleEvent(1, 10)])          # position/size 미지정 → -1.0
    report.write_events_csv(tmp_path / "e.csv", ch)
    assert _rows(tmp_path / "e.csv")[1][5:] == ["-1.0", "-1.0"]


def test_violations_csv(tmp_path):
    path = tmp_path / "violations.csv"
    report.write_violations_csv(path, _chapter())
    rows = _rows(path)
    assert rows[0] == ["frame", "time_s", "expected_point", "actual_point"]
    assert rows[1] == ["8432", "28.107", "4", "5"]


def test_suspects_csv_quotes_free_text(tmp_path):
    path = tmp_path / "suspects.csv"
    report.write_suspects_csv(path, _chapter())
    rows = _rows(path)
    assert rows[0] == ["kind", "point", "frame", "time_s", "detail", "snapshot", "confidence"]
    # detail의 쉼표가 인용되어 필드가 쪼개지지 않아야 한다
    assert rows[1] == ["merge", "5", "8432", "28.107", "기포 2개 겹침, 확인 요망",
                       "suspects/s0001_p5_f008432.png", ""]   # D-15: merge는 확신 등급 대상 아님("")


def test_suspects_csv_none_snapshot_is_empty(tmp_path):
    ch = _chapter(suspects=[SuspectMoment("slow", 2, 10, "정체", None)])
    path = tmp_path / "suspects.csv"
    report.write_suspects_csv(path, ch)
    assert _rows(path)[1] == ["slow", "2", "10", "0.033", "정체", "", ""]


def test_suspects_csv_confidence_column_for_graded_reflux(tmp_path):
    # D-15: reflux 확신 등급("높음"/"낮음")이 마지막 열로 그대로 실려야 한다(기존 열 순서 유지).
    ch = _chapter(suspects=[SuspectMoment("reflux", 3, 900, "역류 감지", None, confidence="높음")])
    path = tmp_path / "suspects.csv"
    report.write_suspects_csv(path, ch)
    assert _rows(path)[0] == ["kind", "point", "frame", "time_s", "detail", "snapshot", "confidence"]
    assert _rows(path)[1] == ["reflux", "3", "900", "3.000", "역류 감지", "", "높음"]


def test_filtered_out_csv_with_summary_row(tmp_path):
    path = tmp_path / "filtered_out.csv"
    blobs = [(5, 100, 12.3, "undersize"), (3, 200, 8.0, "duplicate")]   # (point, frame, size_px, reason) = ch.filtered_out 형식
    report.write_filtered_out_csv(path, blobs)
    rows = _rows(path)
    assert rows[0] == ["point", "frame", "size_px", "reason"]
    assert rows[1] == ["5", "100", "12.3", "undersize"]      # (frame, point) 정렬
    assert rows[2] == ["3", "200", "8.0", "duplicate"]
    assert rows[3][0].startswith("# 총 2건")     # 요약행


def test_filtered_out_csv_caps_rows(tmp_path):
    path = tmp_path / "filtered_out.csv"
    blobs = [(1, f, 5.0, "undersize") for f in range(5)]
    report.write_filtered_out_csv(path, blobs, cap=2)
    rows = _rows(path)
    assert rows[0] == ["point", "frame", "size_px", "reason"]
    assert len(rows) == 1 + 2 + 1                 # 헤더 + 2행 + 요약행
    assert rows[3][0].startswith("# 총 5건 중 2건")


def test_filtered_out_csv_empty(tmp_path):
    path = tmp_path / "filtered_out.csv"
    report.write_filtered_out_csv(path, [])
    rows = _rows(path)
    assert rows[0] == ["point", "frame", "size_px", "reason"]
    assert rows[1][0].startswith("# 총 0건")


def test_build_multiway_summary_schema():
    s = report.build_multiway_summary(_chapter(), _config())
    assert s["video_path"] == "6way_1_1.avi"
    assert s["way"] == 2
    assert s["recording"] == {"width": 1280, "height": 768, "fps": 300.0, "mode": "dma"}
    assert s["points"][0] == {"number": 1, "target": 500, "count_auto": 436,
                              "count_final": 437, "error": -63}
    assert s["points"][1] == {"number": 2, "target": 500, "count_auto": 500,
                              "count_final": 500, "error": 0}
    assert s["sets_completed"] == 431
    assert s["violations"] == 1 and s["suspects"] == 1
    assert s["suspects_high"] == 0   # D-15: 기본 fixture의 "merge" 의심은 확신 등급 대상이 아님(reflux 전용)
    assert s["corrections"] == [{"point": 1, "frame": 8432, "delta": 1,
                                  "reason": "겹침 확인", "corrected_at": "2026-07-10T00:00:00",
                                  "evidence_path": ""}]
    assert s["processed_frames"] == 12339
    assert s["cancelled"] is False and s["suspects_truncated"] is False
    assert s["self_check_messages"] == []
    assert "300fps" in s["time_base"] and "컨테이너 fps 불사용" in s["time_base"]


def test_summary_includes_new_chapter_fields():
    ch = _chapter(cancelled=True, suspects_truncated=True,
                  self_check_messages=["Point 3이 다른 Point 대비 낮습니다"])
    s = report.build_multiway_summary(ch, _config())
    assert s["cancelled"] is True and s["suspects_truncated"] is True
    assert s["self_check_messages"] == ["Point 3이 다른 Point 대비 낮습니다"]


def test_write_multiway_summary_json_korean_no_ascii_escape(tmp_path):
    path = tmp_path / "multiway_summary.json"
    report.write_multiway_summary_json(path, _chapter(), _config())
    raw = path.read_text(encoding="utf-8")
    assert "겹침 확인" in raw and "\\u" not in raw          # ensure_ascii=False
    assert json.loads(raw)["way"] == 2


def test_write_json_rejects_nan(tmp_path):
    # allow_nan=False 계약: 비유한 값은 조용히 쓰이지 않고 즉시 ValueError.
    with pytest.raises(ValueError):
        report._write_json(tmp_path / "x.json", {"v": float("nan")})


def test_write_multiway_summary_json_preserves_existing_file_on_failure(tmp_path):
    # 원자적 쓰기 계약: 실패한 재작성이 기존에 유효했던 파일을 손상시키면 안 된다.
    path = tmp_path / "multiway_summary.json"
    report.write_multiway_summary_json(path, _chapter(), _config())
    original = path.read_text(encoding="utf-8")

    bad = _chapter(counts_auto={1: float("nan"), 2: 500})
    with pytest.raises(ValueError):
        report.write_multiway_summary_json(path, bad, _config())

    assert path.read_text(encoding="utf-8") == original


import numpy as np

from bubble_counter.imgio import imread_unicode


def test_render_count_error_png_colors(tmp_path):
    path = tmp_path / "count_error.png"
    assert report.render_count_error_png(path, {1: 2, 2: -3, 3: 0}, theme="light") is True
    img = imread_unicode(path)
    assert img is not None
    # 초과(+2) 빨강 · 부족(−3) 파랑 픽셀이 실제로 존재(막대는 AA 없이 단색 채움)
    red = np.array(report._hex_to_bgr("#e34948"))
    blue = np.array(report._hex_to_bgr("#2a78d6"))
    assert (img == red).all(axis=2).any(), "초과 막대 빨강이 없음"
    assert (img == blue).all(axis=2).any(), "부족 막대 파랑이 없음"


def test_render_count_error_png_size_scales_with_points(tmp_path):
    path = tmp_path / "c.png"
    report.render_count_error_png(path, {1: 1, 2: 1, 3: 1})
    img = imread_unicode(path)
    assert img.shape == (54 + 300 + 56, 64 + 30 + 3 * 90, 3)   # top+plot+bottom, left+right+n·col


def test_render_count_error_png_dark_theme_bg(tmp_path):
    path = tmp_path / "d.png"
    report.render_count_error_png(path, {1: 2}, theme="dark")
    img = imread_unicode(path)
    assert tuple(int(v) for v in img[0, 0]) == report._hex_to_bgr("#181a1f")


def test_render_count_error_png_bad_theme_raises(tmp_path):
    with pytest.raises(ValueError, match="theme"):
        report.render_count_error_png(tmp_path / "x.png", {1: 1}, theme="sepia")


def test_render_count_error_png_empty_ok(tmp_path):
    path = tmp_path / "e.png"
    assert report.render_count_error_png(path, {}) is True
    assert imread_unicode(path) is not None


# D-4: e==0 라벨 표기를 B트랙 GUI 그래프 플랜과 통일(무부호 "0" + zero 색).
# 라벨은 LINE_AA로 그려져 렌더 결과의 픽셀 정확 일치 검증이 불가능함을 실측
# 확인(실측: "0" 글자에 zero 색 정확 일치 픽셀 0개) — 문자열·색 선택 로직을
# 분리한 순수 함수(_error_label/_label_color)를 렌더링 없이 직접 단위검증한다.


def test_error_label_zero_is_unsigned():
    assert report._error_label(0) == "0"
    assert "+0" not in report._error_label(0)


def test_error_label_nonzero_keeps_sign():
    assert report._error_label(2) == "+2"
    assert report._error_label(-3) == "-3"


def test_label_color_zero_uses_zero_palette():
    pal = {"zero": (1, 2, 3), "text": (4, 5, 6)}
    assert report._label_color(0, pal) == (1, 2, 3)
    assert report._label_color(2, pal) == (4, 5, 6)
    assert report._label_color(-3, pal) == (4, 5, 6)


def test_render_count_error_png_zero_palette_present_in_both_themes(tmp_path):
    # zero 팔레트 키가 실제로 render 경로에 배선돼 있는지(KeyError 없이 통과) 확인.
    assert report.render_count_error_png(tmp_path / "z1.png", {1: 0}, theme="light") is True
    assert report.render_count_error_png(tmp_path / "z2.png", {1: 0}, theme="dark") is True


def test_dir_name_helpers():
    assert report.chapter_dir_name("C:/x/6way_1_1.avi") == "6way_1_1_기포계수_멀티웨이"
    assert report.phase_dir_name("6way_1") == "6way_1_페이즈결과"


def test_write_chapter_outputs_creates_all_files(tmp_path):
    out = tmp_path / "6way_1_1_기포계수_멀티웨이"
    report.write_chapter_outputs(out, _chapter(), _config())
    for name in ("multiway_summary.json", "point_counts.csv", "events.csv",
                 "violations.csv", "suspects.csv", "filtered_out.csv", "count_error.png"):
        assert (out / name).exists(), f"{name} 누락"
    assert json.loads((out / "multiway_summary.json").read_text(encoding="utf-8"))["way"] == 2
    assert imread_unicode(str(out / "count_error.png")) is not None


def test_write_chapter_outputs_writes_chapter_filtered_out(tmp_path):
    out = tmp_path / "ch"
    ch = _chapter(filtered_out=[(5, 100, 12.3, "undersize")])   # A0 ChapterResult.filtered_out
    report.write_chapter_outputs(out, ch, _config())
    rows = _rows(out / "filtered_out.csv")
    assert rows[1] == ["5", "100", "12.3", "undersize"]


def _phase():
    ch1 = _chapter(video_path="a.avi", counts_auto={1: 10, 2: 11}, corrections=[],
                   sets_completed=10)
    ch2 = _chapter(video_path="b.avi", counts_auto={1: 5, 2: 9}, corrections=[],
                   sets_completed=8)
    return PhaseResult("6way_1", [ch1, ch2])


def test_build_phase_summary_aggregates(tmp_path):
    s = report.build_phase_summary(_phase(), _config())
    assert s["name"] == "6way_1"
    assert [c["video_path"] for c in s["chapters"]] == ["a.avi", "b.avi"]
    assert s["sum_sets"] == 18
    # 합산 target = 500 × 2챕터 = 1000; count_final(1) = 10+5 = 15 → error -985
    assert s["points"][0] == {"number": 1, "target_phase": 1000, "count_final": 15,
                              "error": -985}


def test_write_phase_outputs_creates_files(tmp_path):
    out = tmp_path / "6way_1_페이즈결과"
    report.write_phase_outputs(out, _phase(), _config())
    assert (out / "phase_summary.json").exists()
    assert (out / "phase_count_error.png").exists()
    assert imread_unicode(str(out / "phase_count_error.png")) is not None
