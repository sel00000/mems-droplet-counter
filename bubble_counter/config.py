"""Configuration dataclasses: ROI, counting line, and full counter config."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoiSpec:
    """Region of interest as fractions of the full frame, all in [0, 1]."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    def __post_init__(self) -> None:
        if not (0 <= self.x < 1):
            raise ValueError(f"x는 [0, 1) 범위여야 합니다: {self.x}")
        if not (0 <= self.y < 1):
            raise ValueError(f"y는 [0, 1) 범위여야 합니다: {self.y}")
        if not (0 < self.w <= 1):
            raise ValueError(f"w는 (0, 1] 범위여야 합니다: {self.w}")
        if not (0 < self.h <= 1):
            raise ValueError(f"h는 (0, 1] 범위여야 합니다: {self.h}")
        if self.x + self.w > 1 + 1e-9:
            raise ValueError(f"x + w는 1 이하여야 합니다: {self.x} + {self.w}")
        if self.y + self.h > 1 + 1e-9:
            raise ValueError(f"y + h는 1 이하여야 합니다: {self.y} + {self.h}")

    def to_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, w_px, h_px) ints, clipped to frame, w_px>=1, h_px>=1."""
        x0, w_px = _axis_to_pixels(self.x, self.w, frame_w)
        y0, h_px = _axis_to_pixels(self.y, self.h, frame_h)
        return x0, y0, w_px, h_px


def _axis_to_pixels(start_frac: float, len_frac: float, total_px: int) -> tuple[int, int]:
    """Resolve one axis of a fractional ROI to (start_px, length_px).

    Rounds independently, then clips so the result stays within
    [0, total_px] and guarantees length_px >= 1 (when total_px >= 1).
    """
    start = max(0, min(total_px - 1, round(start_frac * total_px)))
    length = max(1, round(len_frac * total_px))
    length = min(length, max(1, total_px - start))
    return start, length


@dataclass(frozen=True)
class LineSpec:
    """The virtual counting line: axis, position within the ROI, and band width."""

    axis: str = "h"  # "h": horizontal line (objects move vertically) / "v": vertical line
    pos: float = 0.5  # fraction along ROI height (axis="h") or ROI width (axis="v"), [0,1]
    band_px: int = 5  # band thickness in processing-scale px, >=1

    def __post_init__(self) -> None:
        if self.axis not in ("h", "v"):
            raise ValueError(f"axis는 'h' 또는 'v'여야 합니다: {self.axis!r}")
        if not (0 <= self.pos <= 1):
            raise ValueError(f"pos는 [0, 1] 범위여야 합니다: {self.pos}")
        if self.band_px < 1:
            raise ValueError(f"band_px는 1 이상이어야 합니다: {self.band_px}")


@dataclass
class CounterConfig:
    """All tunable parameters for one counting run."""

    roi: RoiSpec = field(default_factory=RoiSpec)
    line: LineSpec = field(default_factory=LineSpec)
    scale: float = 1.0  # 0 < scale <= 1, downscale after ROI crop
    history: int = 300  # MOG2 history
    var_threshold: float = 16.0  # MOG2 varThreshold
    morph_kernel: int = 3  # ellipse open kernel size; 0 disables
    row_close_px: int = 3  # 1D closing along the band row; 0 disables
    merge_gap_frames: int = 1  # temporal gap tolerance in RhythmCounter
    min_width_px: int = 2
    min_area_px: int = 4
    warmup_frames: int = 60  # frames fed to MOG2 but not counted
    bucket_seconds: float = 1.0
    fps_override: float | None = None
    annotate: bool = False
    save_rhythm: bool = False
    max_frames: int | None = None  # process at most N frames (None = all)

    def __post_init__(self) -> None:
        if not self.bucket_seconds > 0:  # 0·음수·NaN 모두 거부
            raise ValueError(f"bucket_seconds는 0보다 커야 합니다: {self.bucket_seconds}")
        if self.fps_override is not None and not (
                math.isfinite(self.fps_override) and self.fps_override > 0):
            raise ValueError(f"fps_override는 0보다 큰 유한수여야 합니다: {self.fps_override}")
