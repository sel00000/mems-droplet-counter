"""clipgen(P34-1 골격 + P34-2 오버레이) — FFV1 소형 픽스처로 렌더 계약 검증.

플랜 정본: docs/superpowers/plans/2026-07-11-review-center-P34-clip-audit.md
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_counter.multiway import clipgen
from bubble_counter.multiway.model import PointSpec

FW, FH = 160, 120


def write_ffv1(path, n_frames: int = 60):
    """움직이는 사각형 패턴 — 프레임마다 픽셀이 달라 마커/결정론 검증에 적합."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 300.0, (FW, FH))
    assert w.isOpened(), "이 환경에서 FFV1 라이터를 열 수 없습니다"
    for i in range(n_frames):
        frame = np.full((FH, FW, 3), 30, np.uint8)
        x = (i * 2) % (FW - 20)
        cv2.rectangle(frame, (x, 40), (x + 20, 60), (200, 200, 200), -1)
        w.write(frame)
    w.release()


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _overlay(event_frame=30):
    pts = (PointSpec(1, 0.5, 0.5, 0.1, 0.05, 0.0),)
    return clipgen.ClipOverlay(points=pts, frame_w=FW, frame_h=FH,
                               global_direction="+x", highlight_point=1,
                               event_frame=event_frame)


class TestClipPath:
    def test_rule_no_padding(self):
        # D6 예시 P5_f6084.avi가 정본
        assert clipgen.clip_path(Path("결과"), 5, 6084) == Path("결과") / "clips" / "P5_f6084.avi"


class TestSkeleton:
    def test_frame_count_and_fps(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        out = tmp_path / "clips" / "P1_f30.avi"
        got = clipgen.render_clip(str(src), out, 30)  # hw=25 → [5, 55] = 51프레임
        assert got == out and out.exists()
        cap = cv2.VideoCapture(str(out))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 51
        assert abs(cap.get(cv2.CAP_PROP_FPS) - clipgen.DEFAULT_FPS_OUT) < 0.1
        cap.release()

    def test_clamp_at_start(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        out = tmp_path / "c.avi"
        clipgen.render_clip(str(src), out, 3)  # [0, 28] = 29프레임
        assert len(read_frames(out)) == 29

    def test_clamp_at_end(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        out = tmp_path / "c.avi"
        clipgen.render_clip(str(src), out, 58)  # [33, 59] = 27프레임
        assert len(read_frames(out)) == 27

    def test_center_outside_video_raises(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        with pytest.raises(ValueError):
            clipgen.render_clip(str(src), tmp_path / "c.avi", 500)

    def test_overwrite_rerender(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        out = tmp_path / "c.avi"
        clipgen.render_clip(str(src), out, 30)
        clipgen.render_clip(str(src), out, 30)  # 두 번째도 성공(선삭제 후 이동)
        assert len(read_frames(out)) == 51

    def test_progress_cb_monotonic_to_total(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        calls = []
        clipgen.render_clip(str(src), tmp_path / "c.avi", 30,
                            progress_cb=lambda w, t: calls.append((w, t)))
        assert calls[0] == (1, 51) and calls[-1] == (51, 51)
        assert [w for w, _ in calls] == list(range(1, 52))

    def test_korean_output_path(self, tmp_path):
        """한글 최종 경로 — ASCII 임시에 쓰고 move하는 우회가 동작해야 한다."""
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        out = tmp_path / "한글 결과" / "clips" / "P1_f30.avi"
        clipgen.render_clip(str(src), out, 30)
        assert len(read_frames(out)) == 51


class TestClipOverlay:
    def test_overlay_changes_pixels(self, tmp_path):
        src = tmp_path / "s_ffv1.avi"
        write_ffv1(src, 60)
        plain = tmp_path / "plain.avi"
        over = tmp_path / "over.avi"
        clipgen.render_clip(str(src), plain, 30, half_window=5)
        clipgen.render_clip(str(src), over, 30, half_window=5, overlay=_overlay())
        fp = read_frames(plain)[0]
        fo = read_frames(over)[0]
        assert fp.shape == fo.shape
        assert not np.array_equal(fp, fo)  # ROI·계수선·프레임번호가 그려져 달라짐

    def test_event_frame_marker_only_on_event(self, tmp_path):
        """event_frame 지정/미지정 렌더는 정확히 그 프레임(로컬 5)에서만 달라야 한다."""
        src = tmp_path / "s2_ffv1.avi"
        write_ffv1(src, 60)
        a = tmp_path / "ev.avi"
        b = tmp_path / "noev.avi"
        clipgen.render_clip(str(src), a, 30, half_window=5, overlay=_overlay(event_frame=30))
        clipgen.render_clip(str(src), b, 30, half_window=5, overlay=_overlay(event_frame=None))
        fa, fb = read_frames(a), read_frames(b)
        assert len(fa) == len(fb) == 11
        diff_idx = [i for i, (x, y) in enumerate(zip(fa, fb)) if not np.array_equal(x, y)]
        assert diff_idx == [5]  # center=30 → lo=25 → 로컬 인덱스 5

    def test_deterministic_rerender(self, tmp_path):
        src = tmp_path / "s3_ffv1.avi"
        write_ffv1(src, 60)
        a = tmp_path / "a.avi"
        b = tmp_path / "b.avi"
        clipgen.render_clip(str(src), a, 30, half_window=5, overlay=_overlay())
        clipgen.render_clip(str(src), b, 30, half_window=5, overlay=_overlay())
        for x, y in zip(read_frames(a), read_frames(b)):
            assert np.array_equal(x, y)
