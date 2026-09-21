"""회전 사각형 ROI → 축 정렬 패치 warp (설계 결정 1).

패치 좌표계: 행 y_p = 흐름축(+y_p = 하류), 열 x_p = 폭축.
W_p = round(width×frame_h), H_p = round(length×frame_w).  warpAffine 중심 규약
(cols−1)/2 = cols/2−0.5 (opencv #11784) 준수.  변환 행렬은 측정 시작 시 1회
계산·고정(보간 양자화 #26361).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .model import PointSpec

# direction 문자열 → 화면 벡터 (x 오른쪽, y 아래)
_DIR_VEC = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def patch_dims(point: PointSpec, frame_w: int, frame_h: int) -> tuple[int, int]:
    """(W_p 폭축, H_p 흐름축) 패치 픽셀 치수."""
    w_p = max(1, round(point.width * frame_h))
    h_p = max(1, round(point.length * frame_w))
    return w_p, h_p


def _flow_and_normal(point: PointSpec, downstream: str) -> tuple[float, float, float, float]:
    """(ux, uy, nx, ny): +y_p가 하류가 되도록 뒤집힌 흐름벡터 u와 폭벡터 n."""
    a = math.radians(point.angle_deg)
    ux, uy = math.cos(a), -math.sin(a)      # 흐름축 (자연 방향)
    nx, ny = math.sin(a), math.cos(a)       # 폭축
    dx, dy = _DIR_VEC[downstream]
    if ux * dx + uy * dy < 0:               # 하류와 반대 → 흐름축 뒤집기
        ux, uy = -ux, -uy
    return ux, uy, nx, ny


def patch_affine(point: PointSpec, frame_w: int, frame_h: int, downstream: str) -> np.ndarray:
    """패치(dst) 픽셀 [x_p, y_p, 1] → 프레임(src) [col, row] 2×3 어파인 (dst→src)."""
    w_p, h_p = patch_dims(point, frame_w, frame_h)
    ux, uy, nx, ny = _flow_and_normal(point, downstream)
    cx, cy = point.cx * frame_w, point.cy * frame_h
    hw = (w_p - 1) / 2.0        # 폭축 반폭 (cols/2−0.5 규약)
    hh = (h_p - 1) / 2.0        # 흐름축 반길이
    return np.array(
        [[nx, ux, cx - hw * nx - hh * ux],
         [ny, uy, cy - hw * ny - hh * uy]],
        dtype=np.float64,
    )


def shifted_affine(M: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """M의 평행이동 성분(3번째 열)에 (dx,dy)를 더한 새 어파인 — 패치 샘플링 좌표만 이동
    (D-16 흔들림 보정: 프레임이 (dx,dy)만큼 흔들렸을 때 X+dx,Y+dy에서 다시 샘플링).

    회전·스케일 성분(첫 두 열)은 그대로 둔다 — 평행이동 보정 전용(회전·줌 보정은 범위 밖).
    """
    out = M.copy()
    out[0, 2] += dx
    out[1, 2] += dy
    return out


def extract_patch(gray: np.ndarray, M: np.ndarray, w_p: int, h_p: int,
                  interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """어파인 M으로 (h_p, w_p) 패치 추출.  borderValue=0(밴드 밖은 배경 취급)."""
    return cv2.warpAffine(
        gray, M, (w_p, h_p),
        flags=cv2.WARP_INVERSE_MAP | interp, borderValue=0,
    )


def fast_slice_patch(gray: np.ndarray, point: PointSpec, frame_w: int, frame_h: int,
                     downstream: str) -> np.ndarray | None:
    """angle=0·direction±x·정수 중심·프레임 안이면 순수 슬라이싱(warp와 비트 동일), 아니면 None."""
    if point.angle_deg != 0.0 or downstream not in ("+x", "-x"):
        return None
    w_p, h_p = patch_dims(point, frame_w, frame_h)
    cx, cy = point.cx * frame_w, point.cy * frame_h
    row0 = cy - (w_p - 1) / 2.0     # angle0: 폭축=frame y
    col0 = cx - (h_p - 1) / 2.0     # angle0: 흐름축=frame x
    if row0 != round(row0) or col0 != round(col0):
        return None                 # 반정수 중심 → warp가 처리
    row0, col0 = int(round(row0)), int(round(col0))
    if row0 < 0 or col0 < 0 or row0 + w_p > frame_h or col0 + h_p > frame_w:
        return None                 # 가장자리 → warp가 borderValue로 처리
    crop = gray[row0:row0 + w_p, col0:col0 + h_p]      # (W_p, H_p)
    patch = np.ascontiguousarray(crop.T)               # (H_p, W_p): 행=흐름(x), 열=폭(y)
    if downstream == "-x":
        patch = np.ascontiguousarray(patch[::-1])      # +y_p가 하류(-x)가 되도록 뒤집기
    return patch


def normalize_rect_angle(rect) -> float:
    """minAreaRect 장축 방향각을 PointSpec.angle_deg 규약으로 [-90, 90) 반환.

    OpenCV 버전별 angle 필드(#19749) 대신 꼭짓점 기하로 직접 산출.  a line은 180°
    대칭이므로 [-90, 90)로 접는다.  +y가 아래이므로 dy를 부호 반전해 "위=양수".
    """
    box = cv2.boxPoints(rect)
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    long_edge = e01 if float(np.hypot(*e01)) >= float(np.hypot(*e12)) else e12
    dx, dy = float(long_edge[0]), float(long_edge[1])
    ang = math.degrees(math.atan2(-dy, dx))
    while ang >= 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return ang
