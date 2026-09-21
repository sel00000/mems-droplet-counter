"""Line/band/strip geometry math shared by the counting pipeline.

All functions here operate in processing-scale pixels (after ROI crop and
downscale) along a single axis — the axis perpendicular to the counting
line. Callers normalize a vertical line ("v") to this same convention by
transposing the strip before calling into this module.
"""

import numpy as np


def line_center_px(roi_len: int, pos: float) -> int:
    """Convert a fractional line position to a processing-scale pixel index.

    round(pos * (roi_len - 1)) clipped to [0, roi_len - 1]. roi_len >= 1.
    """
    center = round(pos * (roi_len - 1))
    return max(0, min(roi_len - 1, center))


def strip_bounds(roi_len: int, center: int, band_px: int, pad: int) -> tuple[int, int, int, int]:
    """Compute strip (band + morphology padding) bounds along one axis.

    band spans [center - band_px // 2, same + band_px), then clipped to
    [0, roi_len]. strip = band expanded by pad on each side, clipped to
    [0, roi_len]. Returns (strip_lo, strip_hi, band_lo, band_hi) where
    band_lo/band_hi are RELATIVE to strip_lo. Raises ValueError if the
    clipped band is empty.

    Guarantees: 0 <= strip_lo < strip_hi <= roi_len,
    0 <= band_lo < band_hi <= strip_hi - strip_lo.
    """
    band_lo_abs = center - band_px // 2
    band_hi_abs = band_lo_abs + band_px
    band_lo_clip = max(0, min(roi_len, band_lo_abs))
    band_hi_clip = max(0, min(roi_len, band_hi_abs))
    if band_hi_clip <= band_lo_clip:
        raise ValueError(
            f"band is empty after clipping to roi_len={roi_len} "
            f"(center={center}, band_px={band_px})"
        )

    strip_lo = max(0, min(roi_len, band_lo_clip - pad))
    strip_hi = max(0, min(roi_len, band_hi_clip + pad))

    return strip_lo, strip_hi, band_lo_clip - strip_lo, band_hi_clip - strip_lo


def extract_band_row(strip_mask: np.ndarray, band_lo: int, band_hi: int) -> np.ndarray:
    """Collapse the band rows of a strip mask into one row via per-column max.

    strip_mask: 2D uint8/bool (H_strip, W). Return bool (W,): per-column
    max (i.e. any nonzero) over the band rows [band_lo, band_hi).
    """
    return strip_mask[band_lo:band_hi].any(axis=0)
