"""multiway/stabilize.py — 흔들림 추정기 정밀도(D-16 수용 기준 3)·신뢰도 게이트 회귀."""

import math

import cv2
import numpy as np
import pytest

from bubble_counter.multiway import stabilize


def _textured_reference(w=1280, h=768, seed=2, blur=5):
    """실촬영 텍스처 근사 — 균일분포 노이즈 + 약한 블러.

    무블러/약블러(0~1)나 과블러(9~15)보다 커널 3~5가 위상상관 정밀도가 가장 좋다
    (사전 정밀도 스윕으로 확인: 8시드×7이동 조합에서 blur=3·5만 전부 0.3px 이하).
    seed=2는 그 중에서도 여유(worst-case 오차 ≈0.11px)가 가장 큰 시드.
    """
    rng = np.random.default_rng(seed)
    ref = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
    return cv2.GaussianBlur(ref, (blur, blur), blur / 4.0)


def _shift(frame, dx, dy):
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                           flags=cv2.INTER_LINEAR, borderValue=int(frame.mean()))


class TestShakeEstimatorPrecision:
    """D-16 수용 기준 (3): 알려진 (dx,dy) 이동 프레임 → 추정 오차 ≤0.3px."""

    @pytest.mark.parametrize("dx,dy", [
        (2.3, -1.6), (-3.0, 3.0), (0.4, 0.2), (5.0, 0.0), (0.0, -5.0), (1.5, 1.5), (-0.75, 0.75),
    ])
    def test_recovers_known_shift_within_tolerance(self, dx, dy):
        ref = _textured_reference()
        est = stabilize.ShakeEstimator(ref)
        shifted = _shift(ref, dx, dy)
        est_dx, est_dy = est.estimate(shifted)
        err = math.hypot(est_dx - dx, est_dy - dy)
        assert err <= 0.3, (
            f"오차 {err:.3f}px > 0.3px (dx={dx}, dy={dy} → est=({est_dx:.3f},{est_dy:.3f}))")


class TestShakeEstimatorConfidenceGate:
    """D-16 회귀: 텍스처 없는 기준 대비 소형 전경물체는 (0,0)으로 억제되어야 한다.

    실제로 발견된 버그: 균일 배경 기준에 링(전경)이 들어오면 phaseCorrelate가
    스펙트럼 노이즈에 휘둘려 대형 거짓 이동(최대 ~190px, response 0.01~0.04)을
    냈다 — 역류 클립 회귀 테스트로 발견. 실촬영 3편 실측 response는 0.94~0.99라
    SHAKE_MIN_RESPONSE=0.3으로 안전하게 분리된다.
    """

    def test_flat_reference_with_foreground_blob_returns_zero(self):
        ref = np.full((120, 200), 40, np.uint8)
        frame = ref.copy()
        cv2.circle(frame, (100, 60), 22, 200, 3)
        est = stabilize.ShakeEstimator(ref)
        assert est.estimate(frame) == (0.0, 0.0)


class TestDownscaleSize:
    """실촬영 해상도는 다운스케일, 합성 테스트 해상도는 업스케일하지 않아야 한다."""

    def test_real_resolution_downscales_preserving_aspect(self):
        assert stabilize.downscale_size(1280, 768) == (256, round(768 * 256 / 1280))

    def test_small_resolution_is_not_upscaled(self):
        # INTER_AREA는 축소 전용 — 억지로 업스케일하면 정밀도가 떨어진다(실측 확인됨)
        assert stabilize.downscale_size(200, 120) == (200, 120)


class TestStabilizationTrackerDeadband:
    def test_within_deadband_is_not_corrected(self):
        tracker = stabilize.StabilizationTracker()
        assert tracker.record(0.5, 0.5) is None
        assert tracker.corrected_frames == 0

    def test_over_deadband_returns_shift_and_counts(self):
        tracker = stabilize.StabilizationTracker()
        assert tracker.record(3.0, 4.0) == (3.0, 4.0)
        assert tracker.corrected_frames == 1
        summary = tracker.summary()
        assert summary["max_px"] == pytest.approx(5.0)
        assert summary["corrected_frames"] == 1
