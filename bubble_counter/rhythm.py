"""Streaming Visual-Rhythm mark counting.

Rows of the (time x line) rhythm image are fed one per frame and
8-connected components ("marks") are labeled incrementally with a
union-find over per-row runs.  One finalized mark == one bubble.

Memory is O(line width + total marks created): only the runs of each open
component's last active row plus its accumulated stats are kept between
frames, but the union-find forest (``_parent``) grows by one entry per mark
ever created and is never compacted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mark:
    """One connected component in the (time x line) rhythm image == one bubble."""

    first_frame: int    # frame index of first active row
    last_frame: int     # frame index of last active row (inclusive)
    position_px: float  # area-weighted centroid along the line
    width_px: int       # max_x - min_x + 1 over whole mark
    area_px: int        # total active pixels


# Indices into the per-root accumulated stats list.
_FIRST, _LAST, _MIN_X, _MAX_X, _AREA, _XSUM = range(6)


class RhythmCounter:
    """Streaming 8-connected component labeling over rows fed one per frame.

    merge_gap_frames=g: a segment stays connectable for g missing rows
    (a run at row t connects to a segment last seen at row t' iff
    t - t' <= g + 1 and x-ranges overlap after +-1 dilation).  g=0
    reproduces exactly cv2.connectedComponentsWithStats(connectivity=8)
    semantics.  Filters (min_width_px / min_area_px) are applied on
    FINALIZED whole-mark stats only.
    """

    def __init__(self, line_len: int, *, min_width_px: int = 2, min_area_px: int = 4,
                 merge_gap_frames: int = 1):
        if line_len < 1:
            raise ValueError(f"line_len must be >= 1, got {line_len}")
        if merge_gap_frames < 0:
            # merge_gap_frames는 --merge-gap/GUI 입력이 그대로 도달하므로 다른
            # 사용자 대면 에러처럼 한국어 (line_len 등 내부 가드는 영어 유지).
            raise ValueError(f"merge_gap_frames는 0 이상이어야 합니다: {merge_gap_frames}")
        self._line_len = int(line_len)
        self._min_width = int(min_width_px)
        self._min_area = int(min_area_px)
        self._gap = int(merge_gap_frames)
        self._parent: list[int] = []            # union-find forest
        # root id -> [first_frame, last_frame, min_x, max_x, area, xsum]
        self._stats: dict[int, list[int]] = {}
        # Open runs (start, end, frame, comp): each open component's runs from
        # its last active row.  Kept sorted by start; intervals never overlap
        # even after +-1 dilation (overlapping runs are always united).
        self._open: list[tuple[int, int, int, int]] = []
        self._total = 0
        self._last_fed: int | None = None

    # -- union-find ---------------------------------------------------------

    def _find(self, x: int) -> int:
        parent = self._parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def _union(self, ra: int, rb: int) -> int:
        """Merge roots ra and rb (both canonical); return the surviving root."""
        if ra == rb:
            return ra
        sa = self._stats[ra]
        sb = self._stats.pop(rb)
        self._parent[rb] = ra
        if sb[_FIRST] < sa[_FIRST]:
            sa[_FIRST] = sb[_FIRST]
        if sb[_LAST] > sa[_LAST]:
            sa[_LAST] = sb[_LAST]
        if sb[_MIN_X] < sa[_MIN_X]:
            sa[_MIN_X] = sb[_MIN_X]
        if sb[_MAX_X] > sa[_MAX_X]:
            sa[_MAX_X] = sb[_MAX_X]
        sa[_AREA] += sb[_AREA]
        sa[_XSUM] += sb[_XSUM]
        return ra

    # -- run extraction -------------------------------------------------------

    def _runs(self, row: np.ndarray) -> tuple[list[int], list[int]]:
        arr = np.asarray(row)
        if arr.ndim != 1 or arr.shape[0] != self._line_len:
            raise ValueError(
                f"row must be 1-D of length {self._line_len}, got shape {arr.shape}")
        u8 = arr.view(np.uint8) if arr.dtype == np.bool_ else (arr > 0).astype(np.uint8)
        d = np.diff(np.concatenate(([0], u8, [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        return starts.tolist(), ends.tolist()

    # -- finalization -----------------------------------------------------------

    def _finalize(self, root: int) -> Mark | None:
        """Pop a root, apply whole-mark filters; return a Mark iff it passes."""
        st = self._stats.pop(root)
        width = st[_MAX_X] - st[_MIN_X] + 1
        if width < self._min_width or st[_AREA] < self._min_area:
            return None
        self._total += 1
        return Mark(first_frame=st[_FIRST], last_frame=st[_LAST],
                    position_px=st[_XSUM] / st[_AREA],
                    width_px=width, area_px=st[_AREA])

    # -- public API -------------------------------------------------------------

    def feed(self, row: np.ndarray, frame_idx: int) -> list[Mark]:
        """Process one band row.

        row: bool/uint8 (line_len,), nonzero = active.  frame_idx must be
        strictly increasing but MAY jump (e.g. warmup skip); a jump of k
        frames behaves exactly as k-1 empty rows because both the gap
        bridging and the finalization tests use frame numbers.
        Returns the marks finalized by this call (those passing filters).
        """
        frame_idx = int(frame_idx)
        if self._last_fed is not None and frame_idx <= self._last_fed:
            raise ValueError(
                f"frame_idx must be strictly increasing, got {frame_idx} "
                f"after {self._last_fed}")
        self._last_fed = frame_idx

        starts, ends = self._runs(row)
        stats = self._stats
        open_runs = self._open
        n_open = len(open_runs)
        reach = self._gap + 1

        # Match new runs against open runs (both sorted by start, intervals
        # non-overlapping): two-pointer sweep, O(#runs).  A new run [s, e)
        # 8-connects to an open run [s', e') iff s <= e' and e >= s'.
        new_entries: list[tuple[int, int, int, int]] = []
        i = 0
        for s, e in zip(starts, ends):
            while i < n_open and open_runs[i][1] < s:
                i += 1
            root = -1
            j = i  # open runs may match several new runs: do not consume them
            while j < n_open and open_runs[j][0] <= e:
                if frame_idx - open_runs[j][2] <= reach:
                    r = self._find(open_runs[j][3])
                    root = r if root < 0 else self._union(root, r)
                j += 1
            xsum = (s + e - 1) * (e - s) // 2  # sum of x over [s, e)
            if root < 0:
                root = len(self._parent)
                self._parent.append(root)
                stats[root] = [frame_idx, frame_idx, s, e - 1, e - s, xsum]
            else:
                st = stats[root]
                st[_LAST] = frame_idx
                if s < st[_MIN_X]:
                    st[_MIN_X] = s
                if e - 1 > st[_MAX_X]:
                    st[_MAX_X] = e - 1
                st[_AREA] += e - s
                st[_XSUM] += xsum
            new_entries.append((s, e, frame_idx, root))

        # Components not reachable from any future row are done: a future row
        # t > frame_idx has t - last > gap + 1 whenever frame_idx - last > gap.
        finalized: list[Mark] = []
        for root in [r for r, st in stats.items() if frame_idx - st[_LAST] > self._gap]:
            mark = self._finalize(root)
            if mark is not None:
                finalized.append(mark)

        # Open runs for the next rows: this row's runs, plus runs of still-live
        # components that were not active in this row (their root keeps runs of
        # its own last active row).  Refreshed components' old runs are replaced.
        kept = [run for run in open_runs
                if (rr := self._find(run[3])) in stats and stats[rr][_LAST] < frame_idx]
        self._open = sorted(kept + new_entries)
        return finalized

    def flush(self) -> list[Mark]:
        """Finalize every remaining active mark.  Call exactly once at stream end."""
        marks = []
        for root in list(self._stats.keys()):
            mark = self._finalize(root)
            if mark is not None:
                marks.append(mark)
        self._open = []
        return marks

    @property
    def total_count(self) -> int:
        """Number of marks counted so far (post-filter, including flushed)."""
        return self._total
