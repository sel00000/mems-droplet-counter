"""멀티웨이 synth 시나리오 테스트 — 해석적 GT·결정론·렌더 유효성 (설계 §4.2)."""

from __future__ import annotations

import cv2

from bubble_counter.synth import (
    generate_multiway,
    merge_clip,
    reflux_clip,
    rotated_channel_clip,
    tight_column_clip,
)


def test_rotated_channel_gt_counts(tmp_path):
    gt = rotated_channel_clip(tmp_path / "rot.avi")
    assert gt["scenario"] == "rotated_channel"
    assert gt["expected_counts"] == {1: 5}
    assert len(gt["expected_events"]) == 5
    assert all(e["point"] == 1 for e in gt["expected_events"])
    frames = [e["frame"] for e in gt["expected_events"]]
    assert frames == sorted(frames)                       # (frame, point) 정렬
    assert gt["points"][0]["angle_deg"] == 30.0


def test_rotated_channel_events_cross_near_center(tmp_path):
    gt = rotated_channel_clip(tmp_path / "rot.avi")
    for e in gt["expected_events"]:
        assert 0 <= e["frame"] < gt["n_frames"]
        assert abs(e["x"] - 640) < 15 and abs(e["y"] - 384) < 15   # 평면=중심 근처 교차


def test_rotated_channel_is_deterministic(tmp_path):
    gt1 = rotated_channel_clip(tmp_path / "a.avi", seed=7)
    gt2 = rotated_channel_clip(tmp_path / "b.avi", seed=7)
    assert gt1["expected_events"] == gt2["expected_events"]
    assert (tmp_path / "a.avi").read_bytes() == (tmp_path / "b.avi").read_bytes()


def test_rotated_channel_video_valid(tmp_path):
    gt = rotated_channel_clip(tmp_path / "rot.avi")
    cap = cv2.VideoCapture(str(tmp_path / "rot.avi"))
    assert cap.isOpened()
    n = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        last = frame.shape
    cap.release()
    assert n == gt["n_frames"] and last == (768, 1280, 3)


def test_gt_json_written(tmp_path):
    import json
    gt = rotated_channel_clip(tmp_path / "rot.avi", gt_json=tmp_path / "rot.gt.json")
    disk = json.loads((tmp_path / "rot.gt.json").read_text(encoding="utf-8"))
    assert disk["expected_counts"] == {"1": 5}       # JSON은 int 키를 문자열화


def test_merge_two_bubbles_counted_as_two(tmp_path):
    gt = merge_clip(tmp_path / "merge.avi")
    assert gt["scenario"] == "merge"
    assert gt["expected_counts"] == {1: 2}          # 겹쳐도 정답은 2개
    assert len(gt["expected_events"]) == 2
    assert isinstance(gt["notes"]["merge_frame"], int)


def test_reflux_net_passage_is_one(tmp_path):
    gt = reflux_clip(tmp_path / "reflux.avi")
    assert gt["scenario"] == "reflux"
    assert gt["expected_counts"] == {1: 1}          # 되돌아와도 순 통과 1개
    assert len(gt["expected_events"]) == 1
    assert gt["notes"]["reflux_bubble_point"] == 1


def test_reflux_is_deterministic(tmp_path):
    a = reflux_clip(tmp_path / "a.avi", seed=3)
    b = reflux_clip(tmp_path / "b.avi", seed=3)
    assert a["expected_events"] == b["expected_events"]
    assert (tmp_path / "a.avi").read_bytes() == (tmp_path / "b.avi").read_bytes()


def test_tight_column_all_counted(tmp_path):
    gt = tight_column_clip(tmp_path / "col.avi")
    assert gt["scenario"] == "tight_column"
    assert gt["expected_counts"] == {1: 6}
    frames = [e["frame"] for e in gt["expected_events"]]
    assert frames == sorted(frames) and len(set(frames)) == 6   # 6개 서로 다른 프레임
