"""§4 검증 게이트: 승인된 정답표 4구간 이벤트 1:1 매칭 (누락 0 · 중복 0)."""

import csv

import pytest

from bubble_counter.multiway.engine import run_multiway_chapter
from bubble_counter.multiway.model import BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec
from tests.multiway.gt_fixtures import SEGMENTS

pytestmark = pytest.mark.gt_gate   # 실행: pytest -m gt_gate (클립 필요, WSL/Windows 공용)


def _config(lines, band_len_px=60):
    pts = tuple(
        PointSpec(n, cx / 1280, cy / 768, band_len_px / 1280, 90 / 768, ang)
        for n, (cx, cy, ang) in sorted(lines.items())
    )
    return MultiwayConfig(
        recording=RecordingSpec(1280, 768, 300.0, "dma"),
        points=pts, bubble=BubbleSizeSpec("circle", 50.0, 50.0),
        order_gate=False,  # D-21: 구 §4 게이트는 독립 계수 기준
    )


def _load_gt(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tol = 3 if "±3" in r.get("frame_불확실성", "") else None
            rows.append((int(r["point"].lstrip("P")), int(r["clip_frame"]), tol))
    return rows


def _match(gt_rows, events, default_tol):
    """이벤트 단위 1:1 그리디 매칭. 반환: (누락 목록, 중복=미매칭 관측 목록)."""
    by_point_gt = {}
    for p, f, tol in gt_rows:
        by_point_gt.setdefault(p, []).append([f, tol, False])
    by_point_ev = {}
    for e in events:
        by_point_ev.setdefault(e.point, []).append([e.frame, False])
    missed, extra = [], []
    for p, gts in by_point_gt.items():
        evs = sorted(by_point_ev.get(p, []))
        for g in sorted(gts):
            tol = g[1] if g[1] is not None else default_tol
            best = None
            for ev in evs:
                if ev[1]:
                    continue
                if abs(ev[0] - g[0]) <= tol and (best is None or abs(ev[0] - g[0]) < abs(best[0] - g[0])):
                    best = ev
            if best is None:
                missed.append((p, g[0]))
            else:
                best[1] = True
                g[2] = True
    for p, evs in by_point_ev.items():
        extra.extend((p, ev[0]) for ev in evs if not ev[1])
    return missed, extra


@pytest.mark.parametrize("name,clip,gt_csv,lines,tol,band_len", SEGMENTS,
                         ids=[s[0] for s in SEGMENTS])
def test_gt_segment_zero_error(tmp_path, name, clip, gt_csv, lines, tol, band_len):
    if not clip.exists():
        pytest.skip(f"검증 클립 없음: {clip}")
    gt_rows = _load_gt(gt_csv)
    ch = run_multiway_chapter(str(clip), _config(lines, band_len), tmp_path / name)
    missed, extra = _match(gt_rows, ch.events, tol)
    assert missed == [], f"[{name}] 누락 {len(missed)}건: {missed[:10]}"
    assert extra == [], f"[{name}] 중복/오검출 {len(extra)}건: {extra[:10]}"


def test_gt_no_reflux_suspects(tmp_path):
    """역류 탐지 폐기 후: 실클립 챕터 suspects에 kind==reflux 가 없어야 한다."""
    seg_by_name = {s[0]: s for s in SEGMENTS}
    checked = 0
    for name, clip, _gt, lines, _tol, band in SEGMENTS:
        if not clip.exists():
            continue
        ch = run_multiway_chapter(str(clip), _config(lines, band), tmp_path / name)
        reflux = [s for s in ch.suspects if s.kind == "reflux"]
        assert reflux == [], f"[{name}] reflux 의심이 남아 있음: {reflux[:5]}"
        checked += 1
    if checked == 0:
        pytest.skip("검증 클립 없음")
