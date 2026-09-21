"""D-16: 전역 카메라 흔들림 자동 감지 + 패치 좌표 보정.

평행이동만 다룬다(회전·줌 보정은 범위 밖 — summary.json의 stabilization 블록과
GUI 자기진단 배너로 이 한계를 사용자에게 알린다). SHAKE_STRIDE 프레임 간격으로
위상상관(cv2.phaseCorrelate)으로 전역 (dx,dy)를 추정하고(성능 실측 결과 매프레임
측정은 비용 초과 — 아래 SHAKE_STRIDE 주석 참고), 그 사이 프레임은 직전 추정치를
재사용한다. 불감대(SHAKE_DEADBAND_PX) 이하는 항등 취급해 저흔들림 영상의 결과를
완전히 보존한다. 초과분만 각 Point 패치의 샘플링 좌표(X+dx, Y+dy)로 보정한다 —
전체 프레임 워프는 하지 않는다(패치별 warpAffine 1회로 충분해 비용이 더 싸다).

기준(reference)은 프라이밍 샘플의 전체프레임 중앙값(engine.estimate_background와
동일 산식을 패치가 아닌 전체 프레임에 적용) — 위상상관용 다운스케일 + 한닝 창
곱을 ShakeEstimator가 1회 계산해 챕터 동안 재사용한다(비용 최소화).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SHAKE_DEADBAND_PX = 0.75   # 이하 = 무보정(항등) — 현재 12개 검증 영상 전부 이 아래(실측 σ≈0.17px·최대 0.62px)
SHAKE_BANNER_PX = 3.0      # GUI 자기진단 배너 임계 — max_px가 이보다 클 때만 표시
_DS_MAX = 256              # 위상상관 다운스케일 장축 목표 픽셀(≈256×154 @ 1280×768, 비용 절감)
SHAKE_MIN_RESPONSE = 0.3   # phaseCorrelate 신뢰도(response) 게이트 — 실제 영상 3편 실측
                           # response 0.94~0.99, 텍스처 없는 기준 대비 무늬 없는 프레임에서
                           # 나오는 가비지 추정치는 0.01~0.04(음수 포함) — 양쪽에 6배 이상
                           # 여유를 두고 중간에 배치. 미만이면 (dx,dy)를 버리고 (0,0) 반환
                           # (판단 불가 상태를 "흔들림 없음"으로 간주 — 오보정보다 무보정이 안전).
SHAKE_STRIDE = 2           # 흔들림 측정 프레임 간격 — 실측 결과 매 프레임 측정(stride=1) 시
                           # 12339프레임·6포인트 실촬영 챕터에서 +24.4%(28.26s→35.15s) 비용
                           # 증가로 목표(+15% 이내)를 초과, stride=2로 재측정해 +19.8%로 완화
                           # (28.26s→33.86s, engine.run_multiway_chapter 참고 — 3회 측정 중
                           # 캐시 워밍 후 안정된 2회 평균, 여전히 +15% 목표는 초과하나 사양상
                           # stride=2가 명시적으로 허용된 완화책이라 이 수치로 채택). 미측정
                           # 프레임은 직전 측정값을 그대로 재사용(보정은 매 프레임 계속 적용,
                           # 위상상관만 격프레임).


def downscale_size(w: int, h: int) -> tuple[int, int]:
    """장축을 _DS_MAX에 맞춘 종횡비 보존 다운스케일 목표 (W, H). 1280×768 → (256, 154).

    이미 장축이 _DS_MAX 이하면 원본 그대로 반환(업스케일 금지 — INTER_AREA는 축소 전용이라
    작은 해상도에 억지로 늘리면 오히려 정밀도가 떨어진다. 실촬영 1280×768 기준 설계지만
    합성 테스트처럼 더 작은 프레임에도 안전하게 적용되도록 가드).
    """
    if max(w, h) <= _DS_MAX:
        return w, h
    if w >= h:
        return _DS_MAX, max(1, round(h * _DS_MAX / w))
    return max(1, round(w * _DS_MAX / h)), _DS_MAX


def reference_background(samples: "list[np.ndarray]") -> np.ndarray:
    """프라이밍 샘플의 전체프레임 중앙값 — engine.estimate_background와 동일 산식(전체 프레임판)."""
    if not samples:
        raise ValueError("기준 프레임 샘플이 비어 있습니다")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


class ShakeEstimator:
    """챕터 1개 동안 재사용하는 위상상관 흔들림 추정기 — 기준·한닝 창을 1회 계산해 캐시."""

    def __init__(self, reference_gray: np.ndarray):
        h, w = reference_gray.shape
        ds_w, ds_h = downscale_size(w, h)
        self._scale_x = w / ds_w
        self._scale_y = h / ds_h
        self._size = (ds_w, ds_h)
        self._hann = cv2.createHanningWindow(self._size, cv2.CV_32F)
        ref_small = cv2.resize(reference_gray, self._size, interpolation=cv2.INTER_AREA)
        self._ref = ref_small.astype(np.float32) * self._hann

    def estimate(self, frame_gray: np.ndarray) -> tuple[float, float]:
        """frame_gray(원본 해상도, 기준과 동일 크기) → 기준 대비 전역 (dx,dy), 원본 픽셀 단위.

        cv2.phaseCorrelate(ref, cur) 규약: frame이 기준보다 (dx,dy)만큼 밀렸다면(즉
        frame(x,y)=ref(x-dx,y-dy)) 반환값은 (dx,dy) — 이 값을 그대로 패치 샘플링
        좌표에 더하면(X+dx, Y+dy) 흔들리기 전 물리 위치를 다시 가리킨다(부호는
        TestShakeEstimator로 실측 검증됨).

        response(신뢰도)가 SHAKE_MIN_RESPONSE 미만이면 (0,0)을 반환한다 — 텍스처가
        거의 없는 기준/프레임 조합(예: 균일 배경)에서 phaseCorrelate가 스펙트럼
        노이즈에 휘둘려 내는 거짓 대형 이동을 걸러낸다(오보정보다 무보정이 안전).
        """
        small = cv2.resize(frame_gray, self._size, interpolation=cv2.INTER_AREA)
        cur = small.astype(np.float32) * self._hann
        (dx, dy), response = cv2.phaseCorrelate(self._ref, cur)
        if response < SHAKE_MIN_RESPONSE:
            return 0.0, 0.0
        return dx * self._scale_x, dy * self._scale_y


class StabilizationTracker:
    """챕터 진행 중 프레임별 |shift| 통계 누적 → mean_px/max_px/corrected_frames.

    |shift|는 벡터 크기 math.hypot(dx,dy)로 정의(각 축 개별 초과가 아니라 전체
    이동량 기준 — 물리적 "흔들림 크기"에 대응하는 자연스러운 해석).
    """

    def __init__(self):
        self._mags: "list[float]" = []
        self.corrected_frames = 0

    def record(self, dx: float, dy: float) -> "tuple[float, float] | None":
        """(dx,dy) 1건 기록. 불감대 초과면 보정에 쓸 (dx,dy) 반환, 이하면 None(무보정)."""
        mag = math.hypot(dx, dy)
        self._mags.append(mag)
        if mag <= SHAKE_DEADBAND_PX:
            return None
        self.corrected_frames += 1
        return dx, dy

    def summary(self) -> dict:
        mean_px = float(np.mean(self._mags)) if self._mags else 0.0
        max_px = float(np.max(self._mags)) if self._mags else 0.0
        return {"mean_px": mean_px, "max_px": max_px, "corrected_frames": self.corrected_frames}
