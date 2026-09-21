from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_counter.config import RoiSpec
from bubble_counter.video_io import VideoInfo, VideoSource, probe_video, resolve_fps

N_FRAMES = 20
FPS = 8.0
WIDTH = 64
HEIGHT = 48


def _write_test_video(path: Path) -> None:
    """MJPG .avi with N_FRAMES frames, each frame filled entirely with its own index."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.avi"
    _write_test_video(path)
    return path


class TestVideoSourceReading:
    def test_frame_count_and_consecutive_indices(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            indices = [idx for idx, gray, bgr in vs.frames()]
        assert indices == list(range(N_FRAMES))

    def test_gray_pixel_values_track_frame_index_within_mjpg_tolerance(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            for idx, gray, bgr in vs.frames():
                assert gray.dtype == np.uint8
                assert abs(float(gray.mean()) - idx) <= 3

    def test_gray_frame_has_no_color_channel(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            idx, gray, bgr = next(vs.frames())
        assert gray.ndim == 2

    def test_no_color_by_default(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            idx, gray, bgr = next(vs.frames())
        assert bgr is None

    def test_with_color_yields_bgr_matching_out_size(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            out_w, out_h = vs.out_size
            idx, gray, bgr = next(vs.frames(with_color=True))
        assert bgr.shape == (out_h, out_w, 3)
        assert bgr.dtype == np.uint8


class TestVideoSourceRoiAndScale:
    def test_default_roi_and_scale_out_size_matches_frame(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            assert vs.info.width == WIDTH
            assert vs.info.height == HEIGHT
            assert vs.out_size == (WIDTH, HEIGHT)

    def test_roi_crop_halves_out_size(self, sample_video):
        roi = RoiSpec(0.5, 0.5, 0.5, 0.5)
        with VideoSource(str(sample_video), roi=roi) as vs:
            assert vs.out_size == (32, 24)

    def test_scale_half_rounds_out_size(self, sample_video):
        with VideoSource(str(sample_video), scale=0.5) as vs:
            assert vs.out_size == (32, 24)

    def test_scale_tiny_guarantees_minimum_one_pixel(self, sample_video):
        with VideoSource(str(sample_video), scale=0.01) as vs:
            assert vs.out_size == (1, 1)


class TestVideoSourceErrors:
    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            VideoSource("no_such_file_xyz123.avi")

    def test_garbage_file_raises_io_error(self, tmp_path):
        junk = tmp_path / "junk.avi"
        junk.write_bytes(b"this is not a video")
        with pytest.raises(IOError, match="영상을 열 수 없습니다"):
            VideoSource(str(junk))

    def test_scale_below_zero_raises_value_error(self, sample_video):
        with pytest.raises(ValueError, match="scale"):
            VideoSource(str(sample_video), scale=-1.0)

    def test_scale_above_one_raises_value_error(self, sample_video):
        with pytest.raises(ValueError, match="scale"):
            VideoSource(str(sample_video), scale=2.0)

    def test_scale_validated_before_file_existence(self):
        # The scale guard runs before the existence check, so a nonexistent
        # path with a bad scale surfaces ValueError, not FileNotFoundError.
        with pytest.raises(ValueError, match="scale"):
            VideoSource("no_such_file_xyz123.avi", scale=2.0)


class TestResolveFps:
    def test_override_wins_over_info(self):
        info = VideoInfo(fps=0.0, frame_count=100, width=64, height=48)
        assert resolve_fps(info, 15.0) == 15.0

    def test_uses_info_fps_when_no_override(self):
        info = VideoInfo(fps=30.0, frame_count=100, width=64, height=48)
        assert resolve_fps(info, None) == 30.0

    def test_zero_fps_and_no_override_raises_with_flag_hint(self):
        info = VideoInfo(fps=0.0, frame_count=100, width=64, height=48)
        with pytest.raises(ValueError, match="--fps"):
            resolve_fps(info, None)

    def test_nan_fps_and_no_override_raises_with_flag_hint(self):
        info = VideoInfo(fps=float("nan"), frame_count=100, width=64, height=48)
        with pytest.raises(ValueError, match="--fps"):
            resolve_fps(info, None)

    def test_non_positive_override_falls_back_to_info_fps(self):
        info = VideoInfo(fps=25.0, frame_count=100, width=64, height=48)
        assert resolve_fps(info, 0.0) == 25.0

    def test_fps_error_names_both_cli_flag_and_gui_field(self):
        info = VideoInfo(fps=0.0, frame_count=10, width=64, height=64)
        with pytest.raises(ValueError, match="fps 강제"):
            resolve_fps(info, None)


class TestProbeVideo:
    def test_returns_metadata_matching_fixture(self, sample_video):
        info = probe_video(sample_video)
        assert (info.width, info.height) == (WIDTH, HEIGHT)
        assert info.fps == pytest.approx(FPS)
        assert info.frame_count == N_FRAMES

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            probe_video("없는영상.avi")

    def test_garbage_file_raises_io_error(self, tmp_path):
        junk = tmp_path / "junk.avi"
        junk.write_bytes(b"not a video")
        with pytest.raises(IOError, match="영상을 열 수 없습니다"):
            probe_video(junk)


class TestSeekAndSampleGray:
    def test_sample_gray_returns_requested_frames_in_order(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            grays = vs.sample_gray([0, 5, 10, 19])
        assert len(grays) == 4
        assert all(g.ndim == 2 and g.dtype == np.uint8 for g in grays)
        # 각 프레임은 자기 인덱스 값으로 채워짐 (MJPG 허용오차 ±3)
        for g, idx in zip(grays, [0, 5, 10, 19]):
            assert abs(float(g.mean()) - idx) <= 3

    def test_sample_gray_rewinds_to_zero_for_full_pass(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            vs.sample_gray([7, 12])
            # 샘플 후 frames()는 프레임 0부터 전량을 순차 산출해야 한다
            idxs = [i for i, gray, bgr in vs.frames()]
        assert idxs == list(range(N_FRAMES))

    def test_sample_gray_matches_frames_geometry_under_roi_scale(self, sample_video):
        roi = RoiSpec(0.25, 0.25, 0.5, 0.5)
        with VideoSource(str(sample_video), roi=roi, scale=0.5) as vs:
            out_w, out_h = vs.out_size
            grays = vs.sample_gray([3])
        assert grays[0].shape == (out_h, out_w)

    def test_sample_gray_skips_out_of_range_index(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            grays = vs.sample_gray([0, 999])   # 999는 범위 밖 → 건너뜀
        assert len(grays) == 1

    def test_seek_negative_raises(self, sample_video):
        with VideoSource(str(sample_video)) as vs:
            with pytest.raises(ValueError, match="frame_idx"):
                vs.seek(-1)
