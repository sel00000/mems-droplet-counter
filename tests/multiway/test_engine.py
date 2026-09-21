"""multiway/engine.py — 프라이밍·계수·방향·크기·e2e·preflight."""

import math

import numpy as np
import cv2
import pytest

from bubble_counter.config import CounterConfig
from bubble_counter.multiway.model import (
    BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec,
)
from bubble_counter.multiway import engine


# ---- 공용 빌더 -------------------------------------------------------------

FW, FH = 200, 120       # 프레임 폭·높이 (합성 테스트 표준)


def make_cfg(points, **rec_kw):
    rec = RecordingSpec(width=rec_kw.get("width", FW), height=rec_kw.get("height", FH),
                        fps=rec_kw.get("fps", 300.0), mode=rec_kw.get("mode", "dma"))
    bub = BubbleSizeSpec("circle", 44.0, 44.0)      # GT across~45px
    return MultiwayConfig(recording=rec, points=tuple(points), bubble=bub,
                          global_direction=rec_kw.get("global_direction", "+x"))


def one_point_cfg(**kw):
    # 가로 채널 가운데 Point: 흐름축 x, H_p=0.5×200=100, W_p=0.5×120=60
    p = PointSpec(1, cx=0.5, cy=0.5, length=0.5, width=0.5, angle_deg=0.0, **kw)
    return make_cfg([p])


def wide_point_cfg(**kw):
    # oversize 테스트 전용(D-2): W_p≈100px 확보 → merge_suspect(79.2px) 발동 가능
    # 흐름축 x, H_p=0.5×200=100, W_p=(100/120)×120=100
    p = PointSpec(1, cx=0.5, cy=0.5, length=0.5, width=100 / FH, angle_deg=0.0, **kw)
    return make_cfg([p])


def bg_frame(val=40):
    return np.full((FH, FW), val, np.uint8)


class TestSampleIndices:
    def test_short_video_samples_all(self):
        assert engine.sample_indices(30) == list(range(30))

    def test_long_video_caps_at_priming_frames(self):
        assert engine.sample_indices(5000) == list(range(engine.PRIMING_FRAMES))

    def test_unknown_length_uses_priming_frames(self):
        assert engine.sample_indices(0) == list(range(engine.PRIMING_FRAMES))


class TestEstimateBackground:
    def test_median_rejects_transient_bright(self):
        bg = np.full((11, 7), 40, np.uint8)
        stack = [bg.copy() for _ in range(20)]
        for t in range(0, 20, 5):           # 20%에만 밝은 줄
            stack[t][:, 3] = 220
        med = engine.estimate_background(stack)
        assert np.array_equal(med, bg)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            engine.estimate_background([])


class TestPointRunnerPriming:
    def test_dims_and_downstream(self):
        r = engine.PointRunner(one_point_cfg().points[0], one_point_cfg(), FW, FH)
        assert (r.w_p, r.h_p) == (60, 100)
        assert r.downstream == "+x"

    def test_prime_learns_flat_background(self):
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        r.prime([bg_frame() for _ in range(10)])
        assert r.background_patch.shape == (100, 60)
        assert int(r.background_patch.mean()) == 40
        # 프라이밍 후 배경 프레임은 전경이 거의 없어야 (동결 apply lr=0)
        fg = r._mog2.apply(r.extract(bg_frame()), learningRate=0.0)
        assert float((fg > 0).mean()) < 0.02


def draw_ring(frame_w, frame_h, cx, cy, radius, bg=40, fg=200, thick=3):
    img = np.full((frame_h, frame_w), bg, np.uint8)
    cv2.circle(img, (int(cx), int(cy)), int(radius), int(fg), thick)
    return img


def prime_and_drive(runner, moving_frames, n_prime=15):
    runner.prime([bg_frame() for _ in range(n_prime)])
    evs = []
    for i, g in enumerate(moving_frames):
        evs += runner.process(g, i)
    evs += runner.flush()
    return evs


def crossing_frames(x_start, x_stop, step, radius=22, cy=60, pad=4):
    """앞뒤 배경 pad장 + x_start→x_stop 이동 링 프레임들."""
    frames = [bg_frame() for _ in range(pad)]
    x = x_start
    while (step > 0 and x <= x_stop) or (step < 0 and x >= x_stop):
        frames.append(draw_ring(FW, FH, x, cy, radius))
        x += step
    frames += [bg_frame() for _ in range(pad)]
    return frames


class TestCountingDirection:
    def test_forward_crossing_counts_one(self):
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        evs = prime_and_drive(r, crossing_frames(55, 150, 12))   # +x 정방향
        assert len(evs) == 1
        assert evs[0].point == 1
        assert 0.0 <= evs[0].subframe <= 1.0
        assert r.filtered_out == []

    def test_dn_first_still_counted_no_reflux_drop(self):
        # 역류 탐지 폐기: dn이 up보다 먼저여도 size 통과 시 계수 (drop/reflux 목록 없음).
        from bubble_counter.rhythm import Mark
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        dn1 = Mark(first_frame=10, last_frame=11, position_px=50.0, width_px=44, area_px=800)
        up1 = Mark(first_frame=15, last_frame=16, position_px=50.0, width_px=44, area_px=800)
        r._up_marks = [up1]
        r.dn_marks = [dn1]
        r._pending_dn = [dn1]
        r._last_frame = 60
        evs = r.flush()
        assert len(evs) == 1
        assert evs[0].point == 1
        assert not hasattr(r, "reflux") or getattr(r, "reflux", []) == []

    def test_reverse_crossing_may_count(self):
        # -x 방향 통과도 역류 드롭 없이 size 경로로 계수될 수 있음.
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        evs = prime_and_drive(r, crossing_frames(150, 55, -12))
        assert len(evs) >= 1
        assert not hasattr(r, "reflux")

    def test_undersize_filtered_not_counted(self):
        cfg = one_point_cfg()      # min_accept = 44×0.5 = 22px
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        # D-11 ①: 고립 마크의 파티클 배제선은 WEAK_MIN_PX(20). radius=6 → 마크 폭 18 < 20 → 기각.
        evs = prime_and_drive(r, crossing_frames(55, 150, 12, radius=6))  # 폭 18 < WEAK_MIN 20
        assert evs == []
        assert any(pt == 1 and reason == "undersize" for (pt, frm, size, reason) in r.filtered_out)

    def test_weak_bubble_isolated_counted(self):
        # D-11 ①: WEAK_MIN ≤ width < min_accept 인 고립 소형 마크(직후 대형 없음)는 약기포로 계수.
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        evs = prime_and_drive(r, crossing_frames(55, 150, 12, radius=7))  # 폭 20 = WEAK_MIN(포함)
        assert len(evs) == 1
        assert "weak-bubble" in evs[0].flags
        assert r.filtered_out == []

    def test_oversize_flagged_merge_suspect(self):
        cfg = wide_point_cfg()     # W_p=100px, merge_suspect = 44×1.8 = 79.2px
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        evs = prime_and_drive(r, crossing_frames(55, 150, 10, radius=42))  # 지름 84 > 79.2
        assert len(evs) == 1
        assert "merge-suspect" in evs[0].flags
        assert r.filtered_out == []

    def test_dedup_boundary_gap_over_window_counts_both(self):
        # DEDUP_WINDOW_FRAMES=10. 같은 x에서 dn 마크가 간격 12프레임(>10)으로 재도착하면
        # 오실레이션 dedup에 걸리지 않고 2회 모두 계수돼야 한다(경계 방향 고정 — A4 리뷰 권고).
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        frames = crossing_frames(55, 150, 12) + [bg_frame()] + crossing_frames(55, 150, 12)
        evs = prime_and_drive(r, frames)
        assert len(r.dn_marks) == 2
        gap = r.dn_marks[1].first_frame - r.dn_marks[0].last_frame
        assert gap == 12 and gap > engine.DEDUP_WINDOW_FRAMES
        assert len(evs) == 2
        assert r.filtered_out == []

    def test_size_boundary_exact_min_accept_is_included(self):
        # width_px == min_accept_px() 정확값 — 배제 조건이 "<"(strict)이므로 등호는
        # 포함돼야 한다(경계 방향 고정 — A4 리뷰 권고). 정지 블립(radius=8)이 실측으로
        # width_px=22=min_accept_px(44×0.5)를 정확히 만들어냄을 확인 후 고정.
        cfg = one_point_cfg()      # min_accept = 44×0.5 = 22px
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        frames = ([bg_frame() for _ in range(4)] + [draw_ring(FW, FH, 100, 60, radius=8)]
                  + [bg_frame() for _ in range(10)])
        evs = prime_and_drive(r, frames)
        assert len(r.dn_marks) == 1
        assert r.dn_marks[0].width_px == cfg.bubble.min_accept_px() == 22.0
        assert len(evs) == 1
        assert r.filtered_out == []

    def test_forward_pair_both_counted_no_false_reflux(self):
        # D-10 ②: 같은 채널을 9프레임 간격으로 통과하는 정방향 페어 → 둘 다 정방향 계수.
        # 인접 기포의 상류 마크를 자기 상류로 오인해 거짓 reflux/누락이 나면 안 된다.
        # (dedup=3이라 9간격 페어는 병합되지 않아야 함도 함께 검증.)
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        xs = list(range(45, 160, 20))          # 빠른 정방향 통과(역류 릴리스 물리와 유사)
        gap = 9                                # 도착 간격 9프레임(역류 P1 페어 실측)
        frames = [bg_frame() for _ in range(4)]
        for t in range(gap + len(xs)):
            img = bg_frame()
            if 0 <= t < len(xs):               # 기포 A
                cv2.circle(img, (xs[t], 60), 22, 200, 3)
            if 0 <= t - gap < len(xs):          # 기포 B (gap 프레임 지연)
                cv2.circle(img, (xs[t - gap], 60), 22, 200, 3)
            frames.append(img)
        frames += [bg_frame() for _ in range(6)]
        evs = prime_and_drive(r, frames)
        assert len(evs) == 2                    # 둘 다 계수
        assert all(e.point == 1 for e in evs)

    def test_boundary_bubble_at_clip_end_not_counted(self):
        # D-11 ③: 클립 끝까지 활성인 하류 마크라도 **상류 통과 증거(매칭 up)** 없으면 미완 → 미계수.
        # 링을 하류밴드에만(x−반지름 > 계수선) 두어 상류 마크가 아예 생기지 않게 한다(트레일링 패드 없음).
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        r.prime([bg_frame() for _ in range(15)])
        moving = [draw_ring(FW, FH, x, 60, 22) for x in (130, 135, 140)]  # 전부 하류(x−22 > 100)
        evs = []
        for i, g in enumerate(moving):
            evs += r.process(g, i)
        evs += r.flush()                         # 트레일링 패드 없음 → 경계 마크는 flush에서 판정
        assert evs == []                         # 상류 up 부재 + 클립끝 활성 → 미계수
        assert any(pt == 1 and reason == "boundary-incomplete" for (pt, frm, sz, reason) in r.filtered_out)
        assert r._up_marks == []                 # 상류 마크가 실제로 없어야 이 케이스가 유효

    def test_event_frame_is_mark_midpoint(self):
        # D-10 ①: 계수 이벤트 프레임 == dn 마크 중점 round((first+last)/2). 실제 마크 first/last로 정확값 대조.
        cfg = one_point_cfg()
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        evs = prime_and_drive(r, crossing_frames(55, 150, 12))   # 정방향 단일 통과
        assert len(evs) == 1
        m = next(mm for mm in r.dn_marks if mm.width_px >= cfg.bubble.min_accept_px())
        assert m.first_frame != m.last_frame                     # 다프레임 마크(중점≠last여야 의미)
        assert evs[0].frame == round((m.first_frame + m.last_frame) / 2)

    def test_time_context_leading_residual_filtered(self):
        # D-11 ①: 소형 마크 뒤 CONTEXT_WINDOW 내 같은 x로 대형 마크가 오면 소형=선행 링 잔차로 기각,
        # 대형만 계수. _judge lookahead 실경로를 마크 직접 구성으로 구동(합성 클립은 소형/대형이
        # 병합되거나 인접 상류에 reflux로 오검돼 이 경로만 격리 불가 — 마크 단위로 격리).
        # (대형 없음=약기포 계수 케이스는 test_weak_bubble_isolated_counted와 대비쌍.)
        from bubble_counter.rhythm import Mark
        cfg = one_point_cfg()          # min_accept 22, WEAK_MIN 20
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        small = Mark(first_frame=10, last_frame=11, position_px=50.0, width_px=20, area_px=220)
        big = Mark(first_frame=14, last_frame=16, position_px=50.0, width_px=44, area_px=800)
        up_s = Mark(first_frame=9, last_frame=10, position_px=50.0, width_px=20, area_px=150)
        up_b = Mark(first_frame=13, last_frame=14, position_px=50.0, width_px=44, area_px=600)
        r._up_marks = [up_s, up_b]     # 각 마크에 정방향 상류 부여 → reflux 오검 차단
        r.dn_marks = [small, big]
        r._pending_dn = [small, big]
        r._last_frame = 30             # 경계 아님(last < 끝)
        evs = r.flush()
        assert len(evs) == 1                                     # 대형만 계수(소형은 선행 잔차)
        assert "weak-bubble" not in evs[0].flags                 # 계수된 건 대형
        assert evs[0].frame == round((big.first_frame + big.last_frame) / 2)
        assert any(fr == small.last_frame and reason == "ring-residual" for (pt, fr, sz, reason) in r.filtered_out)  # 소형 기각 기록

    def test_accepted_widths_excludes_leading_residual(self):
        # D-14: engine.preflight 크기 히스토그램이 runner._counter_dn.feed()의 미필터 원시 폭을
        # 직접 모아 중공 링의 분할 arc·선행 잔차가 섞이면 중앙값이 저편향됐다(review D-14) —
        # 사용자가 기포를 작게 그리는 위험 방향에서 ±30% 가드와 우연히 "일치"해 무력화되는 문제였다.
        # 신규 계측 채널 accepted_widths는 _judge()가 승격(계수)한 마크 폭만 담아야 한다 — 위
        # test_time_context_leading_residual_filtered와 동일한 소형 잔차+대형 마크 구성으로,
        # 중앙값이 대형(44px) 기준으로만 나오는지 확인(소형 20px은 선행 잔차로 기각·배제).
        from bubble_counter.rhythm import Mark
        cfg = one_point_cfg()          # min_accept 22, WEAK_MIN 20
        r = engine.PointRunner(cfg.points[0], cfg, FW, FH)
        small = Mark(first_frame=10, last_frame=11, position_px=50.0, width_px=20, area_px=220)
        big = Mark(first_frame=14, last_frame=16, position_px=50.0, width_px=44, area_px=800)
        up_s = Mark(first_frame=9, last_frame=10, position_px=50.0, width_px=20, area_px=150)
        up_b = Mark(first_frame=13, last_frame=14, position_px=50.0, width_px=44, area_px=600)
        r._up_marks = [up_s, up_b]     # 각 마크에 정방향 상류 부여 → reflux 오검 차단
        r.dn_marks = [small, big]
        r._pending_dn = [small, big]
        r._last_frame = 30             # 경계 아님(last < 끝)
        r.flush()
        assert r.accepted_widths == [44.0]                       # 소형 잔차(20px)는 배제, 대형만 반영
        assert float(np.median(r.accepted_widths)) == 44.0        # 중앙값도 대형 기준

    def test_boundary_dropped_vs_normal_counted_contrast(self):
        # D-11 ③ 대비쌍: last<끝 flush 마크는 정상 계수 vs 클립 끝까지 활성인 마크는 드롭.
        cfg = one_point_cfg()
        r1 = engine.PointRunner(cfg.points[0], cfg, FW, FH)      # 정상: 트레일링 배경 → last<끝
        ev1 = prime_and_drive(r1, crossing_frames(55, 150, 12))
        assert len(ev1) == 1
        assert r1.dn_marks[0].last_frame < r1._last_frame        # 통과 후 밴드 이탈 완료
        r2 = engine.PointRunner(cfg.points[0], cfg, FW, FH)      # 경계: 하류밴드에만·트레일링 없음
        r2.prime([bg_frame() for _ in range(15)])
        ev2 = []
        for i, x in enumerate((130, 135, 140)):
            ev2 += r2.process(draw_ring(FW, FH, x, 60, 22), i)
        ev2 += r2.flush()
        assert ev2 == []                                         # 클립 끝 활성 → 미계수
        assert any(pt == 1 and reason == "boundary-incomplete" for (pt, frm, sz, reason) in r2.filtered_out)


def write_ffv1(path, frames):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 300.0, (FW, FH))
    assert w.isOpened(), "이 환경에서 FFV1 라이터를 열 수 없습니다"
    for g in frames:
        w.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    w.release()


# ---- D-16 합성 지터 전용 helper -------------------------------------------
# bg_frame()은 완전 균일(텍스처 0)이라 위상상관 기준으로 쓰면 링이 들어오는 순간
# response가 무너져 거짓 대형 이동이 나온다(회귀: test_stabilize.py 참고). 지터
# 주입 동등성 테스트는 실촬영처럼 정적 텍스처가 있는 배경이 필요해 별도로 둔다.
_JITTER_TEX = cv2.GaussianBlur(
    np.random.RandomState(7).normal(0, 25, (FH, FW)).astype(np.float32), (15, 15), 0)


def _jitter_bg_frame(val=40):
    return np.clip(val + _JITTER_TEX, 0, 255).astype(np.uint8)


def _jitter_crossing_frames(x_start, x_stop, step, radius=22, cy=60, pad=4):
    frames = [_jitter_bg_frame() for _ in range(pad)]
    x = x_start
    while (step > 0 and x <= x_stop) or (step < 0 and x >= x_stop):
        img = _jitter_bg_frame()
        cv2.circle(img, (int(x), int(cy)), int(radius), 200, 3)
        frames.append(img)
        x += step
    frames += [_jitter_bg_frame() for _ in range(pad)]
    return frames


def _jitter_walk(n, amplitude=3.0, seed=42, step_std=0.8):
    """±amplitude(px)로 벡터 크기를 클램프한 고정시드 랜덤워크(누적 이동량)."""
    rng = np.random.RandomState(seed)
    dx = dy = 0.0
    offsets = []
    for _ in range(n):
        dx += rng.normal(0, step_std)
        dy += rng.normal(0, step_std)
        mag = math.hypot(dx, dy)
        if mag > amplitude:
            dx, dy = dx * amplitude / mag, dy * amplitude / mag
        offsets.append((dx, dy))
    return offsets


def _apply_jitter(frames, offsets):
    """np.roll이 아니라 warpAffine으로 실제 카메라 흔들림처럼 서브픽셀 평행이동."""
    out = []
    for g, (dx, dy) in zip(frames, offsets):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(g, M, (FW, FH), borderMode=cv2.BORDER_REPLICATE))
    return out


class TestRunChapter:
    def test_single_point_forward_clip_counts(self, tmp_path):
        cfg = one_point_cfg()
        frames = [bg_frame() for _ in range(20)]                 # 프라이밍 여유
        frames += crossing_frames(55, 150, 12)                   # 정방향 1개
        clip = tmp_path / "one_ffv1.avi"
        write_ffv1(clip, frames)
        ch = engine.run_multiway_chapter(str(clip), cfg, tmp_path)
        assert ch.counts_auto == {1: 1}
        assert ch.processed_frames == len(frames)
        assert ch.violations == [] and ch.sets_completed == 1    # A5: way=1이라 이벤트 1개=set 1개
        assert ch.filtered_out == []                             # 정방향 1개 → 배제 없음

    def test_forward_clip_completes_no_violation(self, tmp_path):
        cfg = one_point_cfg()      # way=1 → 기포 1개마다 set 1
        frames = [bg_frame() for _ in range(20)] + crossing_frames(55, 150, 12)
        clip = tmp_path / "seq_ffv1.avi"
        write_ffv1(clip, frames)
        ch = engine.run_multiway_chapter(str(clip), cfg, tmp_path)
        assert ch.sets_completed == ch.counts_auto[1]     # way=1이라 계수=set
        assert ch.violations == []

    def test_reverse_clip_counts_no_reflux_suspect(self, tmp_path):
        # 역류 탐지 폐기: -x 통과도 계수 가능, suspects에 reflux 없음.
        cfg = one_point_cfg()
        frames = [bg_frame() for _ in range(20)] + crossing_frames(150, 55, -12)
        clip = tmp_path / "reverse_ffv1.avi"
        write_ffv1(clip, frames)
        ch = engine.run_multiway_chapter(str(clip), cfg, tmp_path)
        assert ch.counts_auto[1] >= 1
        assert not any(s.kind == "reflux" for s in ch.suspects)
        assert isinstance(ch.self_check_messages, list)

    def test_synthetic_jitter_matches_baseline_within_one_frame(self, tmp_path):
        """D-16 수용 기준 (2): ±3px 고정시드 랜덤워크 지터 주입 → 보정 ON 결과가
        원본(무지터) 클립과 이벤트 1:1 일치(±1프레임). 지터가 실제로 불감대를
        넘었는지(corrected_frames>0)도 함께 확인한다."""
        cfg = one_point_cfg()
        base_frames = [_jitter_bg_frame() for _ in range(20)] + _jitter_crossing_frames(55, 150, 12)
        offsets = _jitter_walk(len(base_frames))
        assert max(math.hypot(dx, dy) for dx, dy in offsets) <= 3.0 + 1e-9
        jittered_frames = _apply_jitter(base_frames, offsets)

        base_clip = tmp_path / "jitter_base_ffv1.avi"
        jit_clip = tmp_path / "jitter_shaken_ffv1.avi"
        write_ffv1(base_clip, base_frames)
        write_ffv1(jit_clip, jittered_frames)

        ch_base = engine.run_multiway_chapter(str(base_clip), cfg, tmp_path)
        ch_jit = engine.run_multiway_chapter(str(jit_clip), cfg, tmp_path)

        assert ch_jit.counts_auto == ch_base.counts_auto
        assert ch_jit.stabilization.corrected_frames > 0  # 지터가 불감대를 실제로 넘었다
        assert len(ch_jit.events) == len(ch_base.events)
        for e_base, e_jit in zip(ch_base.events, ch_jit.events):
            assert e_jit.point == e_base.point
            assert abs(e_jit.frame - e_base.frame) <= 1


class TestRectsOverlap:
    def test_disjoint_axis_aligned(self):
        a = ((0, 0), (0, 10), (10, 0), (10, 10))
        b = ((20, 20), (20, 30), (30, 20), (30, 30))
        assert engine._rects_overlap(a, b) is False

    def test_overlapping(self):
        a = ((0, 0), (0, 10), (10, 0), (10, 10))
        b = ((5, 5), (5, 15), (15, 5), (15, 15))
        assert engine._rects_overlap(a, b) is True

    def test_touching_edge_is_not_overlap(self):
        a = ((0, 0), (0, 10), (10, 0), (10, 10))
        b = ((10, 0), (10, 10), (20, 0), (20, 10))
        assert engine._rects_overlap(a, b) is False


class TestPreflight:
    def test_resolution_mismatch_blocks(self, tmp_path):
        cfg = one_point_cfg()                     # recording 200×120
        clip = tmp_path / "small_ffv1.avi"
        w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"FFV1"), 300.0, (64, 48))
        assert w.isOpened()
        for _ in range(5):
            w.write(np.full((48, 64, 3), 40, np.uint8))
        w.release()
        rep = engine.preflight(str(clip), cfg)
        assert any("해상도" in b for b in rep.blocking)

    def test_roi_overlap_warns_not_blocks(self, tmp_path):
        # D-12: 겹치는 두 Point → 차단이 아닌 경고(인접 계수선이 벽 영역서 겹치는 정상 배치 허용).
        p1 = PointSpec(1, 0.45, 0.5, 0.4, 0.5, 0.0)
        p2 = PointSpec(2, 0.55, 0.5, 0.4, 0.5, 0.0)     # x로 크게 겹침
        cfg = make_cfg([p1, p2])
        frames = [bg_frame() for _ in range(10)]
        clip = tmp_path / "ov_ffv1.avi"
        write_ffv1(clip, frames)
        rep = engine.preflight(str(clip), cfg)
        assert any("겹칩니다" in w or "겹침" in w for w in rep.warnings)      # 경고로 이동
        assert not any("겹칩니다" in b or "겹침" in b for b in rep.blocking)  # 더 이상 차단 아님

    @pytest.mark.parametrize("lines_name, band_px", [("LINES_1_3", 60), ("LINES_2_2", 45)],
                             ids=["1_3", "역류_2_2"])
    def test_gt_layout_overlap_not_blocking(self, lines_name, band_px):
        # D-12 회귀 고정: gt_fixtures 게이트 배치(1_3·역류 2_2)는 인접 계수선이 벽 영역서 겹치나,
        # preflight 기하 검사(영상 불필요)에서 차단(blocking)이 아니라 경고여야 한다.
        # (게이트를 통과한 이 배치들이 실전 preflight서 차단당한 최종 리스크의 회귀. 2_2는 겹침 3건.)
        import tests.multiway.gt_fixtures as gf
        lines = getattr(gf, lines_name)
        pts = tuple(PointSpec(n, cx / 1280, cy / 768, band_px / 1280, 90 / 768, ang)
                    for n, (cx, cy, ang) in sorted(lines.items()))
        cfg = MultiwayConfig(recording=RecordingSpec(1280, 768, 300.0, "dma"),
                             points=pts, bubble=BubbleSizeSpec("circle", 50.0, 50.0))
        rep = engine._geometry_check(cfg)                    # 순수 기하 검사(영상 불필요)
        assert rep.blocking == []                            # D-12: 게이트 배치 차단 0건
        assert any("겹칩니다" in w or "겹침" in w for w in rep.warnings)  # 겹침은 경고로(회귀 유의미)

    def test_direction_ambiguous_warns(self, tmp_path):
        # 흐름축 세로(angle -90 → u=(0,1))인데 downstream +x → 내적 0
        p = PointSpec(1, 0.5, 0.5, 0.3, 0.3, angle_deg=-90.0)
        cfg = make_cfg([p], global_direction="+x")
        clip = tmp_path / "amb_ffv1.avi"
        write_ffv1(clip, [bg_frame() for _ in range(10)])
        rep = engine.preflight(str(clip), cfg)
        assert any("방향" in w for w in rep.warnings)

    def test_good_clip_no_blocking_and_histogram_filled(self, tmp_path):
        cfg = one_point_cfg()
        frames = [bg_frame() for _ in range(20)]
        for _ in range(3):                          # 3회 통과 → 히스토그램 표본
            frames += crossing_frames(55, 150, 12)
        clip = tmp_path / "good_ffv1.avi"
        write_ffv1(clip, frames)
        rep = engine.preflight(str(clip), cfg)
        assert rep.blocking == []
        assert 1 in rep.size_histogram

    def test_segment_isolation_no_cross_contamination_at_boundary(self):
        # D-14 후속(review D-14 후속): _measure_point가 3구간을 이어붙여 1개 러너로 1회만
        # 처리+flush()하면, 구간 경계(~수 프레임)에서 DEDUP_WINDOW_FRAMES/CONTEXT_WINDOW가
        # 실제로는 수백 프레임 떨어진 무관한 다음 구간의 내용과 섞여 정상 마크가 "같은
        # 링의 분할 arc"로 오기각되거나 잔차가 오승격될 수 있었다. 구간마다 완전히 새
        # PointRunner로 처리하면(카운터 재생성) 한 구간의 판정(_counted/dn_marks)이 다른
        # 구간에 전혀 전달되지 않아야 한다 — 두 구간(각각 같은 위치에서 정상 통과 1회)을
        # 합쳐 처리한 결과가 각 구간을 독립적으로 처리한 결과의 합과 정확히 같아야 한다.
        cfg = one_point_cfg()
        seg_a = [bg_frame() for _ in range(20)] + crossing_frames(55, 150, 12)
        seg_b = [bg_frame() for _ in range(20)] + crossing_frames(55, 150, 12)

        combined = engine._measure_point(cfg, cfg.points[0], FW, FH, [seg_a, seg_b])
        alone_a = engine._measure_point(cfg, cfg.points[0], FW, FH, [seg_a])
        alone_b = engine._measure_point(cfg, cfg.points[0], FW, FH, [seg_b])
        assert sorted(combined["widths"]) == sorted(alone_a["widths"] + alone_b["widths"])
        assert len(combined["widths"]) == 2          # 두 통과 모두 계수(경계 억제/오염 없음)

    def test_preflight_no_bubble_blocks_but_runs_geometry(self, tmp_path):
        # D-13 ②: 기포 크기 미지정(None) → 크기 검사만 생략 + blocking 안내, 기하·해상도는 완주(크래시 X).
        import dataclasses
        cfg = dataclasses.replace(one_point_cfg(), bubble=None)
        clip = tmp_path / "nb_ffv1.avi"
        write_ffv1(clip, [bg_frame() for _ in range(10)])
        rep = engine.preflight(str(clip), cfg)          # None 접근 AttributeError 없이 완주해야 함
        assert any("기포 크기가 지정되지 않았습니다" in b for b in rep.blocking)
        assert rep.size_histogram == {}                 # 크기 검사 생략 → 히스토그램 비어야

    def test_effective_diameter_px_from_widths_only(self):
        assert engine._effective_diameter_px([]) == 0.0
        assert engine._effective_diameter_px([40.0, 50.0, 60.0]) == 50.0

    def test_band_length_warn_independent_of_user_bubble(self, tmp_path, monkeypatch):
        # 밴드 length 권장이 사용자 BubbleSizeSpec에 좌우되지 않음 (probe 폭만).
        import dataclasses
        import re
        fixed = {"widths": [40.0, 50.0, 60.0], "dpx": 80.0, "contact_ratio": 0.0}

        def _fake_measure(cfg, point, frame_w, frame_h, segments):
            return dict(fixed)

        monkeypatch.setattr(engine, "_measure_point", _fake_measure)
        short = PointSpec(1, 0.5, 0.5, 0.05, 0.25, angle_deg=0.0)  # h_p≈10
        base = make_cfg([short])
        clip = tmp_path / "band_indep_ffv1.avi"
        write_ffv1(clip, [bg_frame() for _ in range(10)])
        cfg30 = dataclasses.replace(base, bubble=BubbleSizeSpec("circle", 30.0, 30.0))
        cfg90 = dataclasses.replace(base, bubble=BubbleSizeSpec("circle", 90.0, 90.0))
        rep30 = engine.preflight(str(clip), cfg30)
        rep90 = engine.preflight(str(clip), cfg90)
        w30 = [w for w in rep30.warnings if "밴드가 짧" in w]
        w90 = [w for w in rep90.warnings if "밴드가 짧" in w]
        assert len(w30) == 1 and len(w90) == 1
        # dia_eff=median([40,50,60])=50 → rec=(80+50)/200=0.65
        assert "측정폭중앙 50" in w30[0] and "측정폭중앙 50" in w90[0]
        assert re.search(r"length≈0\.65", w30[0]) and re.search(r"length≈0\.65", w90[0])
