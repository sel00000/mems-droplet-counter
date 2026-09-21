"""multiway/loader.py 테스트."""

import json
import tempfile
from pathlib import Path

import pytest

from bubble_counter.multiway.loader import (
    ChapterView, find_video, is_review_folder, load_chapter,
)
from bubble_counter.multiway.model import (
    BubbleEvent, BubbleSizeSpec, ChapterResult, Correction, MultiwayConfig,
    PointSpec, RecordingSpec, SequenceViolation, SuspectMoment, StabilizationStats,
    ForeignObject,
)


def _cfg():
    return MultiwayConfig(
        RecordingSpec(1280, 768, 300.0, "dma"),
        tuple(PointSpec(i, 0.1 * i + 0.1, 0.5, 0.05, 0.05, 0.0) for i in range(1, 7)),
        BubbleSizeSpec("circle", 44.0, 44.0),
    )


def _chapter():
    return ChapterResult(
        video_path="test.avi", recording=_cfg().recording,
        counts_auto={1: 10, 2: 11}, corrections=[],
        events=[BubbleEvent(1, 100, 0.25, ("merge-suspect",), position_px=50.0, size_px=44.0)],
        violations=[SequenceViolation(150, 3, 4)], suspects=[SuspectMoment("merge", 1, 200, "test")],
        sets_completed=3, processed_frames=1000,
    )


def test_find_video_from_summary(tmp_path):
    summary = {"video_path": "/absolute/path/test.avi"}
    folder = tmp_path / "test_folder"
    folder.mkdir()
    (folder / "multiway_summary.json").write_text(json.dumps(summary))
    # 절대 경로 파일이 없으면 None
    assert find_video(summary, folder) is None


def test_is_review_folder(tmp_path):
    folder = tmp_path / "review"
    folder.mkdir()
    assert not is_review_folder(folder)
    (folder / "multiway_summary.json").write_text("{}")
    assert is_review_folder(folder)


def test_load_chapter_with_all_files(tmp_path):
    folder = tmp_path / "test.avi_기포계수_멀티웨이"
    folder.mkdir()

    # 최소 필수 파일들
    ch = _chapter()
    import csv
    with open(folder / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "frame", "subframe", "time_s", "flags", "position_px", "size_px"])
        w.writerow(["1", "100", "0.25", "0.333", "merge-suspect", "50.0", "44.0"])

    with open(folder / "filtered_out.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "frame", "size_px", "reason"])
        w.writerow(["1", "50", "20.0", "undersize"])

    with open(folder / "suspects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "point", "frame", "time_s", "detail", "snapshot", "confidence"])
        w.writerow(["merge", "1", "200", "0.667", "test", "", ""])

    with open(folder / "multiway_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump({
            "video_path": "test.avi",
            "way": 6,
            "counts_auto": {"1": 10},
            "corrections": [],
            "violations": 0,
            "suspects": 1,
            "sets_completed": 3,
            "processed_frames": 1000,
            "cancelled": False,
            "config": {"recording": {"width": 1280, "height": 768, "fps": 300.0, "mode": "dma"},
                      "points": [{"number": 1, "cx": 0.1, "cy": 0.1, "length": 0.05, "width": 0.05, "angle_deg": 0.0}],
                      "bubble": {"shape": "circle", "major_px": 44.0, "minor_px": 44.0}},
        }, f, ensure_ascii=False)

    (folder / "corrections_log.csv").write_text("corrected_at,point,frame,delta,reason,evidence_path\n")
    (folder / "foreign_objects.json").write_text("{}")
    (folder / "audit.json").write_text("{}")

    view = load_chapter(folder)
    assert isinstance(view, ChapterView)
    assert len(view.events) == 1
    assert view.events[0].position_px == 50.0
    assert view.events[0].size_px == 44.0
    assert len(view.filtered) == 1
    assert view.filtered[0][3] == "undersize"
    assert len(view.suspects) == 1
    assert view.config is not None
    assert view.video_path is None  # 원본 영상 없음


def test_load_chapter_legacy_csv(tmp_path):
    """구버전 CSV(컬럼 적음) 로드 시 기본값 채워짐."""
    folder = tmp_path / "legacy"
    folder.mkdir()

    with open(folder / "events.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["point", "frame", "subframe", "time_s", "flags"])
        w.writerow(["1", "100", "0.25", "0.333", ""])

    with open(folder / "filtered_out.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["point", "frame", "size_px"])
        w.writerow(["1", "50", "20.0"])

    with open(folder / "suspects.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["kind", "point", "frame", "time_s", "detail", "snapshot"])
        w.writerow(["merge", "1", "200", "0.667", "test", ""])

    with open(folder / "multiway_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"video_path": "test.avi", "way": 1, "counts_auto": {"1": 10}}, f)

    (folder / "corrections_log.csv").write_text("corrected_at,point,frame,delta,reason,evidence_path\n")

    view = load_chapter(folder)
    assert view.events[0].position_px == -1.0
    assert view.events[0].size_px == -1.0
    assert view.filtered[0][3] == ""  # reason 빈 문자열
    assert view.suspects[0].confidence == ""


def test_load_chapter_count_errors_counts_recording(tmp_path):
    """신버전 폴더: violations·counts_auto·recording 세 필드 모두 로드."""
    folder = tmp_path / "sample"
    folder.mkdir()

    (folder / "multiway_summary.json").write_text(json.dumps({
        "video_path": "test.avi",
        "recording": {"width": 1280, "height": 768, "fps": 300.0, "mode": "dma"},
    }), encoding="utf-8")
    # count_errors.csv (헤더: frame,time_s,expected_point,actual_point) — time_s는 파생값이라 버림
    (folder / "count_errors.csv").write_text(
        "frame,time_s,expected_point,actual_point\n"
        "3048,10.160,5,6\n"
        "3058,10.193,1,2\n",
        encoding="utf-8",
    )
    # point_counts.csv (헤더 기반 파싱 — 여분 컬럼은 무시)
    (folder / "point_counts.csv").write_text(
        "point,count_auto,corrections,count_final,target,error\n"
        "1,308,0,308,500,-192\n"
        "2,310,0,310,500,-190\n",
        encoding="utf-8",
    )

    view = load_chapter(folder)
    # violations: (frame, expected_point, actual_point), time_s는 버림
    assert len(view.violations) == 2
    assert (view.violations[0].frame, view.violations[0].expected_point,
            view.violations[0].actual_point) == (3048, 5, 6)
    # counts_auto: point → count_auto
    assert view.counts_auto == {1: 308, 2: 310}
    # recording: summary["recording"] 복원
    assert view.recording is not None
    assert view.recording.fps == 300.0


def test_load_chapter_violations_csv_fallback(tmp_path):
    """count_errors.csv 없음 + violations.csv만 존재 → 폴백으로 동일 로드."""
    folder = tmp_path / "fallback"
    folder.mkdir()
    (folder / "multiway_summary.json").write_text(
        json.dumps({"video_path": "test.avi"}), encoding="utf-8")
    (folder / "violations.csv").write_text(
        "frame,time_s,expected_point,actual_point\n"
        "3048,10.160,2,1\n",
        encoding="utf-8",
    )

    view = load_chapter(folder)
    assert len(view.violations) == 1
    assert (view.violations[0].frame, view.violations[0].expected_point,
            view.violations[0].actual_point) == (3048, 2, 1)


def test_load_chapter_legacy_no_count_files(tmp_path):
    """구버전 폴더(위반 CSV 둘 다 없음·point_counts.csv 없음·recording 키 없음): [] / {} / None, 예외 없음."""
    folder = tmp_path / "legacy_count"
    folder.mkdir()
    (folder / "multiway_summary.json").write_text(
        json.dumps({"video_path": "test.avi"}), encoding="utf-8")

    view = load_chapter(folder)
    assert view.violations == []
    assert view.counts_auto == {}
    assert view.recording is None


def test_load_chapter_recording_invalid_fps(tmp_path):
    """recording 비정상 값(fps=0 → RecordingSpec 검증 실패) → None 폴백(예외 전파 금지)."""
    folder = tmp_path / "bad_recording"
    folder.mkdir()
    (folder / "multiway_summary.json").write_text(json.dumps({
        "video_path": "test.avi",
        "recording": {"width": 1280, "height": 768, "fps": 0, "mode": "dma"},
    }), encoding="utf-8")

    view = load_chapter(folder)
    assert view.recording is None