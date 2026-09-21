"""live.py — 인코드·제어·L4(관찰 무영향) 헬퍼."""

import queue
import time

import numpy as np
import pytest

from bubble_counter.live import (
    LiveControl,
    LiveEmitter,
    LiveSnapshot,
    encode_preview,
    queue_put_drop,
)


class TestEncodePreview:
    def test_gray_to_jpeg(self):
        img = np.full((100, 200), 40, np.uint8)
        data, w, h = encode_preview(img, max_edge=640)
        assert data[:2] == b"\xff\xd8"
        assert (w, h) == (200, 100)

    def test_downscales_long_edge(self):
        img = np.zeros((1200, 800), np.uint8)
        data, w, h = encode_preview(img, max_edge=320)
        assert max(w, h) <= 320
        assert len(data) > 0


class TestLiveControl:
    def test_speed_and_pause(self):
        c = LiveControl(1.0)
        assert c.get_speed() == 1.0
        assert c.is_display_paused() is False
        c.set_speed(0.1)
        c.set_display_paused(True)
        assert c.get_speed() == pytest.approx(0.1)
        assert c.is_display_paused() is True

    def test_bad_speed(self):
        with pytest.raises(ValueError):
            LiveControl(0)


class TestQueuePutDrop:
    def test_keeps_latest(self):
        q = queue.Queue(maxsize=1)
        assert queue_put_drop(q, "a")
        assert queue_put_drop(q, "b")
        assert q.get_nowait() == "b"


class TestLiveEmitter:
    def test_disabled_without_cb(self):
        em = LiveEmitter(None)
        assert em.enabled is False
        em.pace()
        em.emit(0, 10, np.zeros((8, 8), np.uint8), {"total": 0}, "", 0.0)

    def test_throttles_and_delivers(self):
        got = []
        em = LiveEmitter(got.append, max_ui_fps=1000.0)
        img = np.full((40, 60), 30, np.uint8)
        em.emit(1, 100, img, {"total": 2}, "ch", 50.0)
        em.emit(2, 100, img, {"total": 3}, "ch", 50.0)
        assert len(got) >= 1
        assert isinstance(got[0], LiveSnapshot)
        assert got[0].counts["total"] == 2 or got[-1].counts["total"] in (2, 3)

    def test_cb_exception_swallowed(self):
        def bad(_):
            raise RuntimeError("ui dead")

        em = LiveEmitter(bad, max_ui_fps=1000.0)
        em.emit(0, 1, np.zeros((4, 4), np.uint8), {}, "", 0.0)

    def test_pace_slows_when_speed_low(self):
        ctrl = LiveControl(0.05)
        em = LiveEmitter(lambda s: None, ctrl, max_ui_fps=1000.0)
        t0 = time.perf_counter()
        for _ in range(3):
            em.pace()
        elapsed = time.perf_counter() - t0
        # 0.05 * 30fps → 1.5fps → ~0.67s/frame → 3 paces ≳ 1s
        assert elapsed >= 0.8


class TestLiveL4Pipeline:
    """L4: live on/off/드롭이 단일선 계수 결과를 바꾸지 않는다."""

    def test_run_count_live_invariant(self, tmp_path):
        from bubble_counter.config import CounterConfig, LineSpec
        from bubble_counter.pipeline import run_count
        from bubble_counter.synth import SynthConfig, generate

        video = tmp_path / "live_l4.avi"
        sc = SynthConfig(duration_s=2.0, fps=30.0, seed=7)
        generate(sc, video)
        cfg = CounterConfig(line=LineSpec(axis="h", pos=0.5, band_px=64),
                            warmup_frames=5, max_frames=60)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        out_c = tmp_path / "c"
        out_a.mkdir()
        out_b.mkdir()
        out_c.mkdir()

        r0 = run_count(video, out_a, cfg, resmon_interval=0)
        snaps = []
        r1 = run_count(video, out_b, cfg, resmon_interval=0,
                       live_cb=snaps.append, live_label="t")
        ctrl = LiveControl(0.5)
        r2 = run_count(video, out_c, cfg, resmon_interval=0,
                       live_cb=lambda s: None, live_control=ctrl, live_label="t")
        assert r0.total == r1.total == r2.total
        assert [(m.first_frame, m.last_frame, m.area_px) for m in r0.marks] == [
            (m.first_frame, m.last_frame, m.area_px) for m in r1.marks]
        assert len(snaps) >= 1


class TestLiveL4Multiway:
    def test_chapter_live_invariant(self, tmp_path):
        import cv2
        from bubble_counter.multiway import engine
        from bubble_counter.multiway.model import (
            BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec,
        )

        FW, FH = 200, 120

        def bg():
            return np.full((FH, FW), 40, np.uint8)

        def crossing(y0, y1, step):
            frames = []
            for y in range(y0, y1, step):
                f = bg()
                cv2.circle(f, (FW // 2, y), 12, 220, -1)
                frames.append(f)
            return frames

        frames = [bg() for _ in range(20)] + crossing(55, 150, 12)
        clip = tmp_path / "mw_live_ffv1.avi"
        w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"FFV1"), 300.0, (FW, FH))
        for g in frames:
            w.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        w.release()

        p = PointSpec(1, cx=0.5, cy=0.5, length=0.5, width=0.5, angle_deg=0.0)
        rec = RecordingSpec(width=FW, height=FH, fps=300.0, mode="dma")
        cfg = MultiwayConfig(recording=rec, points=(p,),
                             bubble=BubbleSizeSpec("circle", 44.0, 44.0),
                             global_direction="+x")

        ch0 = engine.run_multiway_chapter(str(clip), cfg, tmp_path / "o0")
        snaps = []
        ch1 = engine.run_multiway_chapter(
            str(clip), cfg, tmp_path / "o1", live_cb=snaps.append, live_label="c1")
        ctrl = LiveControl(1.0)
        ctrl.set_display_paused(True)  # 엔진은 무시해야 함
        ch2 = engine.run_multiway_chapter(
            str(clip), cfg, tmp_path / "o2",
            live_cb=lambda s: None, live_control=ctrl, live_label="c2")

        assert ch0.counts_auto == ch1.counts_auto == ch2.counts_auto
        assert [(e.point, e.frame) for e in ch0.events] == [
            (e.point, e.frame) for e in ch1.events]
        assert len(snaps) >= 1
