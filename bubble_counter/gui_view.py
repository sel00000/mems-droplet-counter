"""멀티웨이 GUI 표시 계층 — tkinter를 import하지 않는 순수 모듈(설계 §3).

뷰는 3×3 어파인 행렬 하나로 관리한다(회전 없는 균일 스케일 + 평행이동):
    screen = scale * image + (tx, ty)
ROI 사각형의 회전은 이 뷰와 무관한 '객체' 회전이며 PointSpec.angle_deg가 담당한다.

tkinter가 없는 환경(이 WSL dev 머신)에서도 단위 테스트할 수 있도록 gui_multiway.py와
분리되어 있다(gui_params.py와 같은 패턴). PhotoImage 표시용 PPM bytes 생성도 여기 둔다.
"""

from __future__ import annotations

import numpy as np


class ViewTransform:
    """이미지 픽셀 좌표 ↔ 캔버스 화면 좌표 변환. 줌/팬 상태를 담는다."""

    def __init__(self, scale: float, tx: float, ty: float,
                 min_scale: float = 0.05, max_scale: float = 64.0) -> None:
        self.scale = float(scale)
        self.tx = float(tx)
        self.ty = float(ty)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    @classmethod
    def fit(cls, image_w: int, image_h: int, canvas_w: int, canvas_h: int,
            min_scale: float = 0.05, max_scale: float = 64.0) -> "ViewTransform":
        scale = min(canvas_w / image_w, canvas_h / image_h)
        tx = (canvas_w - scale * image_w) / 2.0
        ty = (canvas_h - scale * image_h) / 2.0
        return cls(scale, tx, ty, min_scale, max_scale)

    @property
    def matrix(self) -> tuple:
        return (
            (self.scale, 0.0, self.tx),
            (0.0, self.scale, self.ty),
            (0.0, 0.0, 1.0),
        )

    def image_to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return (self.scale * ix + self.tx, self.scale * iy + self.ty)

    def screen_to_image(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.tx) / self.scale, (sy - self.ty) / self.scale)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """커서 아래 이미지점을 고정한 채 확대/축소. 스케일은 [min,max]로 클램프."""
        ix, iy = self.screen_to_image(sx, sy)          # 줌 전 커서 아래 이미지점
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        self.scale = new_scale
        # image_to_screen(ix,iy) == (sx,sy) 가 되도록 평행이동 재설정
        self.tx = sx - new_scale * ix
        self.ty = sy - new_scale * iy

    def pan(self, dx: float, dy: float) -> None:
        self.tx += dx
        self.ty += dy

    def clamp_pan(self, image_w: int, image_h: int, canvas_w: int, canvas_h: int,
                  margin: float = 0.0) -> None:
        # 이미지 우측 끝(tx + scale*image_w)이 margin보다 왼쪽으로 사라지지 않게,
        # 이미지 좌측 끝(tx)이 canvas_w - margin보다 오른쪽으로 사라지지 않게.
        lo_x = margin - self.scale * image_w
        hi_x = canvas_w - margin
        self.tx = max(lo_x, min(hi_x, self.tx))
        lo_y = margin - self.scale * image_h
        hi_y = canvas_h - margin
        self.ty = max(lo_y, min(hi_y, self.ty))


def ppm_p6_bytes(rgb: np.ndarray) -> bytes:
    """(H,W,3) uint8 RGB → PPM(P6) 바이트. tk.PhotoImage(data=..., format='PPM')용.

    호출자는 OpenCV BGR을 RGB로 뒤집어(예: frame[:, :, ::-1]) 넘긴다.
    """
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb는 uint8 배열이어야 합니다: {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb는 (H, W, 3) RGB 배열이어야 합니다: {rgb.shape}")
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + np.ascontiguousarray(rgb).tobytes()
