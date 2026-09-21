"""Tests for bubble_counter.rhythm — streaming union-find mark counting.

The differential test against cv2.connectedComponentsWithStats defines
correctness for merge_gap_frames=0: the streaming counter must reproduce
offline 8-connectivity component stats exactly.
"""

import numpy as np
import cv2
import pytest
from collections import Counter

from bubble_counter.rhythm import RhythmCounter, Mark


def run_stream(strip, **kw):
    rc = RhythmCounter(strip.shape[1], **kw)
    marks = []
    for t in range(strip.shape[0]):
        marks += rc.feed(strip[t], t)
    marks += rc.flush()
    return rc, marks


def offline_stats(strip):
    n, _, stats, cent = cv2.connectedComponentsWithStats(
        (strip > 0).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        out.append((y, y + h - 1, w, int(area), round(float(cent[i][0]), 4)))
    return Counter(out)


def rows(*patterns):
    """Build a bool strip from iterables of 0/1."""
    return np.array(patterns, dtype=bool)


def naive_reference(strip, gap):
    """Brute-force reference with pixel-set bookkeeping (no union-find).

    Same connectivity semantics as RhythmCounter: a component keeps only the
    runs of its last active row connectable, for up to gap missing rows.
    Stats are recomputed from raw pixel sets, so this independently checks
    the streaming counter's incremental stat merging.
    """
    comps = []  # each: {"pixels": set[(t, x)], "last": int, "runs": [(s, e)]}
    finished = []
    n_rows, width = strip.shape
    for t in range(n_rows):
        runs = []
        x = 0
        while x < width:
            if strip[t, x]:
                s = x
                while x < width and strip[t, x]:
                    x += 1
                runs.append((s, x))
            else:
                x += 1
        hitsets = []
        for s, e in runs:
            hitsets.append({ci for ci, c in enumerate(comps)
                            if t - c["last"] <= gap + 1
                            and any(s <= oe and e >= os_ for os_, oe in c["runs"])})
        # Group runs and components that are transitively connected.
        groups = []  # (set of run indices, set of comp indices)
        for k, hs in enumerate(hitsets):
            g_runs, g_comps = {k}, set(hs)
            for g in [g for g in groups if g[1] & g_comps]:
                g_runs |= g[0]
                g_comps |= g[1]
                groups.remove(g)
            groups.append((g_runs, g_comps))
        merged_away = set()
        new_comps = []
        for g_runs, g_comps in groups:
            pixels = set()
            for ci in g_comps:
                pixels |= comps[ci]["pixels"]
            merged_away |= g_comps
            for k in g_runs:
                s, e = runs[k]
                pixels |= {(t, x) for x in range(s, e)}
            new_comps.append({"pixels": pixels, "last": t,
                              "runs": sorted(runs[k] for k in g_runs)})
        survivors = []
        for ci, c in enumerate(comps):
            if ci in merged_away:
                continue
            (finished if t - c["last"] > gap else survivors).append(c)
        comps = survivors + new_comps
    finished += comps
    out = []
    for c in finished:
        ts = [t for t, _ in c["pixels"]]
        xs = [x for _, x in c["pixels"]]
        out.append((min(ts), max(ts), max(xs) - min(xs) + 1, len(xs),
                    round(sum(xs) / len(xs), 4)))
    return Counter(out)


# ---------------------------------------------------------------------------
# Differential test: streaming (gap=0, filters off) == cv2 8-connectivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("density,seed,shape", [
    (0.02, 1, (300, 80)), (0.10, 2, (300, 80)),
    (0.30, 3, (200, 120)), (0.55, 4, (150, 60)), (0.85, 5, (80, 40))])
def test_streaming_equals_cv2(density, seed, shape):
    rng = np.random.default_rng(seed)
    strip = (rng.random(shape) < density)
    _, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    got = Counter((m.first_frame, m.last_frame, m.width_px, m.area_px,
                   round(m.position_px, 4)) for m in marks)
    assert got == offline_stats(strip)


@pytest.mark.parametrize("gap", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_streaming_gap_equals_naive_reference(gap, seed):
    rng = np.random.default_rng(seed)
    strip = rng.random((120, 40)) < 0.18
    strip[rng.random(120) < 0.35] = False  # blank rows exercise gap bridging
    _, marks = run_stream(strip, min_width_px=1, min_area_px=1,
                          merge_gap_frames=gap)
    got = Counter((m.first_frame, m.last_frame, m.width_px, m.area_px,
                   round(m.position_px, 4)) for m in marks)
    assert got == naive_reference(strip, gap)


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------

def test_single_row_mark_fast_bubble():
    # A bubble visible in exactly one frame leaves a one-row mark.
    strip = rows([0, 1, 1, 1, 0, 0])
    rc, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    assert len(marks) == 1
    m = marks[0]
    assert m.first_frame == 0 and m.last_frame == 0
    assert m.width_px == 3 and m.area_px == 3
    assert m.position_px == pytest.approx(2.0)
    assert rc.total_count == 1


def test_diagonal_mark_is_one_component():
    # One pixel per row shifting x by +1 each frame: 8-connected diagonal.
    strip = np.zeros((5, 6), dtype=bool)
    for t in range(5):
        strip[t, t] = True
    _, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    assert len(marks) == 1
    m = marks[0]
    assert (m.first_frame, m.last_frame, m.width_px, m.area_px) == (0, 4, 5, 5)
    assert m.position_px == pytest.approx(2.0)


def test_two_nonadjacent_runs_are_two_marks():
    strip = rows([1, 1, 0, 0, 0, 1, 1, 1])
    _, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    got = sorted((m.width_px, m.area_px, m.position_px) for m in marks)
    assert len(marks) == 2
    assert got == [(2, 2, pytest.approx(0.5)), (3, 3, pytest.approx(6.0))]
    assert all(m.first_frame == 0 and m.last_frame == 0 for m in marks)


def test_y_merge_keeps_earliest_first_frame():
    # Segment A starts at t=0, segment B at t=1, both join at t=2.
    strip = rows(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # t=0: A
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],   # t=1: A, B
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],   # t=2: bridge joins A and B
    )
    rc, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    assert len(marks) == 1
    m = marks[0]
    assert m.first_frame == 0          # earlier of the two merged segments
    assert m.last_frame == 2
    assert m.width_px == 10            # min_x=0 .. max_x=9
    assert m.area_px == 14             # 2 + 4 + 8 active pixels
    # xsum = (0+1) + (0+1+8+9) + (1+..+8) = 1 + 18 + 36 = 55
    assert m.position_px == pytest.approx(55 / 14)
    assert rc.total_count == 1


def test_gap_zero_splits_gap_one_bridges():
    strip = rows(
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
    )
    # gap=0: two separate one-row marks; the first is finalized by feed(t=1).
    rc = RhythmCounter(6, min_width_px=1, min_area_px=1, merge_gap_frames=0)
    assert rc.feed(strip[0], 0) == []
    mid = rc.feed(strip[1], 1)
    assert len(mid) == 1 and mid[0].first_frame == 0 and mid[0].last_frame == 0
    assert rc.feed(strip[2], 2) == []
    tail = rc.flush()
    assert len(tail) == 1 and tail[0].first_frame == 2
    assert rc.total_count == 2

    # gap=1: one mark bridging the missing row; area counts active pixels only.
    rc, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=1)
    assert len(marks) == 1
    m = marks[0]
    assert (m.first_frame, m.last_frame) == (0, 2)
    assert m.area_px == 4              # not 6: the empty row adds nothing
    assert m.width_px == 2
    assert m.position_px == pytest.approx(3.5)
    assert rc.total_count == 1


def test_gap_one_does_not_bridge_two_missing_rows():
    strip = rows(
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    )
    _, marks = run_stream(strip, min_width_px=1, min_area_px=1, merge_gap_frames=1)
    assert len(marks) == 2


def test_frame_idx_jump_splits_even_with_gap():
    # Active at frames 0 and 1, then the next feed jumps to frame 10:
    # the jump acts as 8 empty rows, so gap=1 cannot bridge it.
    row = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    rc = RhythmCounter(6, min_width_px=1, min_area_px=1, merge_gap_frames=1)
    assert rc.feed(row, 0) == []
    assert rc.feed(row, 1) == []
    finalized = rc.feed(row, 10)
    assert len(finalized) == 1
    assert (finalized[0].first_frame, finalized[0].last_frame) == (0, 1)
    tail = rc.flush()
    assert len(tail) == 1
    assert (tail[0].first_frame, tail[0].last_frame) == (10, 10)
    assert rc.total_count == 2


def test_frame_idx_jump_equals_empty_rows():
    # Skipping frame 1 in feed() is the same as feeding one empty row:
    # gap=1 bridges it (2 - 0 <= gap + 1).
    row = np.array([0, 1, 1, 0], dtype=bool)
    rc = RhythmCounter(4, min_width_px=1, min_area_px=1, merge_gap_frames=1)
    assert rc.feed(row, 0) == []
    assert rc.feed(row, 2) == []
    marks = rc.flush()
    assert len(marks) == 1
    assert (marks[0].first_frame, marks[0].last_frame) == (0, 2)
    assert marks[0].area_px == 4


def test_min_width_filter_rejects_and_excludes_from_total():
    # A 5-row, 1-px-wide mark is rejected by min_width_px=2.
    strip = np.zeros((5, 4), dtype=bool)
    strip[:, 2] = True
    rc, marks = run_stream(strip, min_width_px=2, min_area_px=1, merge_gap_frames=0)
    assert marks == []
    assert rc.total_count == 0


def test_min_area_filter():
    strip = rows([1, 1, 0])
    rc, marks = run_stream(strip, min_width_px=1, min_area_px=3, merge_gap_frames=0)
    assert marks == [] and rc.total_count == 0

    strip = rows([1, 1, 0], [1, 1, 0])
    rc, marks = run_stream(strip, min_width_px=1, min_area_px=3, merge_gap_frames=0)
    assert len(marks) == 1 and marks[0].area_px == 4
    assert rc.total_count == 1


def test_filters_apply_per_mark_on_whole_mark_stats():
    # One 3x3 blob (passes defaults) + one lone pixel (fails defaults).
    strip = np.zeros((3, 10), dtype=bool)
    strip[:, 0:3] = True
    strip[0, 8] = True
    rc, marks = run_stream(strip, min_width_px=2, min_area_px=4, merge_gap_frames=0)
    assert len(marks) == 1
    assert marks[0].width_px == 3 and marks[0].area_px == 9
    assert rc.total_count == 1


def test_total_count_matches_returned_marks():
    rng = np.random.default_rng(7)
    strip = (rng.random((200, 50)) < 0.3)
    rc, marks = run_stream(strip, min_width_px=2, min_area_px=4, merge_gap_frames=0)
    assert rc.total_count == len(marks)
    # Cross-check the post-filter count against cv2 whole-mark stats.
    expected = sum(1 for (_, _, w, area, _) in offline_stats(strip).elements()
                   if w >= 2 and area >= 4)
    assert rc.total_count == expected


def test_uint8_255_row_equals_bool_row():
    row_bool = np.array([0, 1, 1, 0], dtype=bool)
    row_u8 = np.array([0, 255, 255, 0], dtype=np.uint8)
    for row in (row_bool, row_u8):
        rc = RhythmCounter(4, min_width_px=1, min_area_px=1, merge_gap_frames=0)
        rc.feed(row, 0)
        marks = rc.flush()
        assert len(marks) == 1
        assert marks[0].area_px == 2 and marks[0].position_px == pytest.approx(1.5)


def test_all_empty_stream_counts_nothing():
    strip = np.zeros((10, 8), dtype=bool)
    rc, marks = run_stream(strip)
    assert marks == [] and rc.total_count == 0


def test_flush_finalizes_everything_and_second_flush_is_empty():
    row = np.array([1, 1, 1, 1], dtype=bool)
    rc = RhythmCounter(4, min_width_px=1, min_area_px=1, merge_gap_frames=1)
    rc.feed(row, 0)
    marks = rc.flush()
    assert len(marks) == 1 and rc.total_count == 1
    assert rc.flush() == []
    assert rc.total_count == 1


def test_row_length_mismatch_raises():
    rc = RhythmCounter(4)
    with pytest.raises(ValueError):
        rc.feed(np.zeros(5, dtype=bool), 0)


def test_non_increasing_frame_idx_raises():
    rc = RhythmCounter(4)
    rc.feed(np.zeros(4, dtype=bool), 3)
    with pytest.raises(ValueError):
        rc.feed(np.zeros(4, dtype=bool), 3)


def test_invalid_constructor_args_raise():
    with pytest.raises(ValueError):
        RhythmCounter(0)
    with pytest.raises(ValueError):
        RhythmCounter(4, merge_gap_frames=-1)


def test_merge_gap_error_message_is_korean():
    with pytest.raises(ValueError, match="0 이상이어야 합니다"):
        RhythmCounter(10, merge_gap_frames=-1)


def test_mark_is_a_dataclass_with_contract_fields():
    m = Mark(first_frame=1, last_frame=2, position_px=3.5, width_px=2, area_px=4)
    assert (m.first_frame, m.last_frame, m.position_px, m.width_px, m.area_px) \
        == (1, 2, 3.5, 2, 4)
