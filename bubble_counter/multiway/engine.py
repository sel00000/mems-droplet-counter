"""멀티웨이 코어 엔진: 2-pass 프라이밍 · Point별 밴드-리듬 계수 · 방향/크기 판정.

설계 결정 1~3 + 선행조사 반영.  Point마다 독립 MOG2(패치가 작아 비용 무시).
1차 패스: 앞부분 샘플의 픽셀 중앙값으로 무기포 배경 추정 → 모델 고정.
2차 패스: lr=0 상시 동결, 유휴 프레임에서만 소량 학습(조명 드리프트 흡수).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import cv2

from bubble_counter.live import LiveEmitter
from bubble_counter.pipeline import ProgressEvent
from bubble_counter.rhythm import Mark, RhythmCounter
from bubble_counter.video_io import VideoSource
from .model import (
    BubbleEvent, ChapterResult, MultiwayConfig, PointSpec, StabilizationStats, SuspectMoment,
)
from .warp import extract_patch, fast_slice_patch, patch_affine, patch_dims, shifted_affine
from . import sequence, stabilize, suspects, foreign

PRIMING_FRAMES = 120
PRIMING_APPLY_REPEAT = 5
WARMUP_FRAMES = 0
IDLE_FG_RATIO = 0.001
IDLE_LR = 0.005
DIRECTION_WINDOW = 15
DEDUP_WINDOW_FRAMES = 3      # D-10 ④: 오실레이션/근접 재크로싱 분리 (10→3)
CONTEXT_WINDOW = 9          # D-11 ①: 소형 마크 직후 이 창 안에 대형 마크가 오면 선행 링 잔차로 간주
                            # (실측 캘리브: 선행 링→주 마크 last_frame 간격 최대 8 → 9로 상향, 약기포 간격 ~46은 무해)
WEAK_MIN_PX = 20.0         # D-11 ①: 약기포 인정 하한(파티클 10~20px 배제선 유지)

# 소형 마크의 시간맥락 lookahead(CONTEXT_WINDOW)는 지연 판정 유예(DIRECTION_WINDOW)보다 짧아야
# 판정 시점에 후속 대형 마크가 이미 finalize돼 있다(pending 파이프라인 순서 보장). — D-11 불변식
assert CONTEXT_WINDOW < DIRECTION_WINDOW


def _effective_diameter_px(widths: list[float]) -> float:
    """밴드 length 권장용 지름(px). 사용자 BubbleSizeSpec 금지 — probe 인정 폭 중앙값만."""
    if not widths:
        return 0.0
    return float(np.median(widths))


def sample_indices(n_frames: int) -> list[int]:
    """1차 패스(배경 추정) 샘플 프레임 인덱스."""
    count = PRIMING_FRAMES if n_frames <= 0 else min(PRIMING_FRAMES, n_frames)
    return list(range(count))


def estimate_background(patches: "list[np.ndarray]") -> np.ndarray:
    """패치 스택의 픽셀별 중앙값 = 무기포 배경 추정."""
    if not patches:
        raise ValueError("배경 추정 샘플이 비어 있습니다")
    return np.median(np.stack(patches, axis=0), axis=0).astype(np.uint8)


class PointRunner:
    """Point 1개의 계수 상태: 고정 어파인 + 독립 MOG2 + 상/하류 RhythmCounter 2개."""

    def __init__(self, point: PointSpec, cfg: MultiwayConfig, frame_w: int, frame_h: int):
        self.point = point
        self.number = point.number
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.downstream = cfg.effective_direction(point)
        self.w_p, self.h_p = patch_dims(point, frame_w, frame_h)
        self._M = patch_affine(point, frame_w, frame_h, self.downstream)
        self._use_fast = point.angle_deg == 0.0 and self.downstream in ("+x", "-x")
        self._mid = self.h_p // 2       # 흐름축 상/하류 경계
        base = cfg.base
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=base.history, varThreshold=base.var_threshold, detectShadows=False)
        self._morph = (None if base.morph_kernel <= 0
                       else cv2.getStructuringElement(
                           cv2.MORPH_ELLIPSE, (base.morph_kernel, base.morph_kernel)))
        self._row_close = (None if base.row_close_px <= 1
                           else np.ones((1, base.row_close_px), np.uint8))
        self._counter_up = RhythmCounter(
            self.w_p, min_width_px=base.min_width_px, min_area_px=base.min_area_px,
            merge_gap_frames=base.merge_gap_frames)
        self._counter_dn = RhythmCounter(
            self.w_p, min_width_px=base.min_width_px, min_area_px=base.min_area_px,
            merge_gap_frames=base.merge_gap_frames)
        self.background_patch: np.ndarray | None = None
        # 계수 상태
        self._min_accept = cfg.bubble.min_accept_px()
        self._merge_suspect = cfg.bubble.merge_suspect_px()
        self._diameter = max(cfg.bubble.major_px, cfg.bubble.minor_px)  # 중공 링 분할 arc 병합용 (D-11 보완)
        self._up_marks: list[Mark] = []                       # 방향 매칭용 상류 마크 버퍼(주기적으로 가지치기됨)
        self._consumed_up: set[int] = set()                   # 1:1 배타 매칭용 소비된 상류 id (D-10 ②)
        self._pending_dn: list[Mark] = []                     # 지연 판정 대기 하류 마크 (D-2)
        self._counted: list[tuple[float, float, int]] = []    # (x_lo, x_hi, last_frame) dedup
        self._dn_ratio: dict[int, float] = {}                 # frame → 하류밴드 활성행 비율
        self._last_frame = -1
        self.filtered_out: list[tuple[int, int, float, str]] = []  # (point, frame, size_px, reason)
        self.dn_marks: list[Mark] = []                        # 하류 finalize 마크 전량(의심 판정 입력)
        self.accepted_widths: list[float] = []                # D-14: 인정(계수)된 마크 폭만(히스토그램용, dn_marks는 미필터 전량)

    def extract(self, gray: np.ndarray, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """이 Point의 (H_p, W_p) 패치.  shift=(dx,dy) 지정 시 샘플링 좌표를 그만큼 밀어
        추출한다(D-16 흔들림 보정 — 전체 프레임 워프가 아니라 패치 좌표만 이동).
        shift=(0,0)이고 angle=0·±x면 고속 슬라이싱(비트 동일), 그 외엔 warp."""
        if shift == (0.0, 0.0) and self._use_fast:
            fast = fast_slice_patch(gray, self.point, self.frame_w, self.frame_h, self.downstream)
            if fast is not None:
                return fast
        M = self._M if shift == (0.0, 0.0) else shifted_affine(self._M, *shift)
        return extract_patch(gray, M, self.w_p, self.h_p)

    def prime(self, samples: "list[np.ndarray]") -> None:
        """샘플들의 패치 중앙값을 배경으로 MOG2 모델 고정."""
        patches = [self.extract(g) for g in samples]
        if not patches:
            return
        self.background_patch = estimate_background(patches)
        for _ in range(PRIMING_APPLY_REPEAT):
            self._mog2.apply(self.background_patch, learningRate=1.0)

    def _close_row(self, row_bool: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(
            row_bool.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, self._row_close)[0] > 0

    def process(self, gray: np.ndarray, frame_idx: int,
                shift: tuple[float, float] = (0.0, 0.0)) -> list[BubbleEvent]:
        self._last_frame = frame_idx
        patch = self.extract(gray, shift)
        fg = self._mog2.apply(patch, learningRate=0.0)        # 상시 동결
        if self._morph is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._morph)
        fg_bool = fg > 0
        if float(fg_bool.mean()) < IDLE_FG_RATIO:             # 유휴 → 소량 학습(fg 버림)
            self._mog2.apply(patch, learningRate=IDLE_LR)
        up_band = fg_bool[: self._mid, :]
        dn_band = fg_bool[self._mid :, :]
        up_row = up_band.any(axis=0) if up_band.shape[0] else np.zeros(self.w_p, bool)
        dn_row = dn_band.any(axis=0)
        if self._row_close is not None:
            up_row = self._close_row(up_row)
            dn_row = self._close_row(dn_row)
        if dn_band.shape[0]:
            self._dn_ratio[frame_idx] = float(dn_band.any(axis=1).mean())
        for m in self._counter_up.feed(up_row, frame_idx):
            self._up_marks.append(m)
        for m in self._counter_dn.feed(dn_row, frame_idx):
            self.dn_marks.append(m)
            self._pending_dn.append(m)                        # 즉시 판정 금지 → 지연 버퍼 (D-2)
        # 지연 판정: DIRECTION_WINDOW만큼 유예해 상류 마크가 finalize될 시간을 준다 (no-upstream 매칭).
        ready, still = [], []
        for m in self._pending_dn:
            (ready if m.last_frame + DIRECTION_WINDOW <= frame_idx else still).append(m)
        ready.sort(key=lambda m: m.last_frame)
        self._pending_dn = still
        events: list[BubbleEvent] = []
        for m in ready:
            ev = self._judge(m)
            if ev is not None:
                events.append(ev)
        self._prune(frame_idx)
        return events

    def flush(self) -> list[BubbleEvent]:
        for m in self._counter_up.flush():
            self._up_marks.append(m)
        for m in self._counter_dn.flush():
            self.dn_marks.append(m)
            self._pending_dn.append(m)
        events: list[BubbleEvent] = []
        for m in sorted(self._pending_dn, key=lambda m: m.last_frame):  # 잔여 전량 판정 (D-2)
            if m.last_frame >= self._last_frame:   # 클립 끝까지 활성 상태로 flush된 마크 = 미완 통과 → 미계수
                # D-10 ③(전량 드롭)로 복원: band=60(D-11 ①)에서 완전통과 마크는 last_frame < 끝이라
                # 이 분기에 안 걸리므로, 여기 걸리는 건 GT가 제외하는 클립끝 미완 기포뿐.
                self.filtered_out.append((self.number, m.last_frame, float(m.width_px), "boundary-incomplete"))
                continue
            ev = self._judge(m)
            if ev is not None:
                events.append(ev)
        self._pending_dn = []
        return events

    def _match_up(self, dn_mark: Mark, lo: float, hi: float) -> Mark | None:
        """x-겹침(±1) + 창 내 미소비 상류 후보 중 |up.first−dn.first| 최소(동률 시 first 이른 쪽).
        선택된 상류는 소비(1:1 배타) — no-upstream 플래그·품질용 (D-10 ②)."""
        best, best_key = None, None
        for m in self._up_marks:
            if id(m) in self._consumed_up:
                continue
            mlo = m.position_px - m.width_px / 2.0
            mhi = m.position_px + m.width_px / 2.0
            if lo <= mhi + 1 and hi >= mlo - 1:
                dt = abs(m.first_frame - dn_mark.first_frame)
                if dt <= DIRECTION_WINDOW:
                    key = (dt, m.first_frame)
                    if best_key is None or key < best_key:
                        best, best_key = m, key
        if best is not None:
            self._consumed_up.add(id(best))
        return best

    def _judge(self, dn_mark: Mark) -> BubbleEvent | None:
        """하류 마크 1개 → BubbleEvent(계수) 또는 None(제외). 부수효과로 기록 채움.

        역류 탐지/드롭은 폐기 — dn 마크는 size·dedup·boundary만으로 계수 여부 결정.
        """
        lo = dn_mark.position_px - dn_mark.width_px / 2.0
        hi = dn_mark.position_px + dn_mark.width_px / 2.0
        flags: list[str] = []
        match = self._match_up(dn_mark, lo, hi)
        if match is None:
            flags.append("no-upstream")
        dn_center = (lo + hi) / 2.0
        for (clo, chi, cframe) in self._counted:               # 오실레이션 dedup + 중공 링 분할 arc 병합
            if dn_mark.first_frame - cframe <= DEDUP_WINDOW_FRAMES and (
                    (lo <= chi + 1 and hi >= clo - 1)           # x-겹침(오실레이션 재크로싱)
                    or abs(dn_center - (clo + chi) / 2.0) < self._diameter):  # 중심간 지름 이내 = 같은 링의 분할 arc (D-11 보완)
                self.filtered_out.append((self.number, dn_mark.last_frame, float(dn_mark.width_px), "duplicate"))
                return None
        size_px = float(dn_mark.width_px)
        if size_px < self._min_accept:
            # D-11 ①: 시간맥락 크기필터. 같은 x(±1)에서 직후 [last, last+CONTEXT_WINDOW]에 대형 마크가
            # 뒤따르면 선행 링 잔차로 기각; 아니면 약기포로 계수(단 WEAK_MIN_PX 하한 = 파티클 배제).
            lead_residual = any(
                om is not dn_mark and om.width_px >= self._min_accept
                and dn_mark.last_frame <= om.last_frame <= dn_mark.last_frame + CONTEXT_WINDOW
                and lo <= om.position_px + om.width_px / 2.0 + 1
                and hi >= om.position_px - om.width_px / 2.0 - 1
                for om in self.dn_marks)
            if lead_residual or size_px < WEAK_MIN_PX:          # 선행 잔차이거나 파티클 → 기각
                    reason = "ring-residual" if lead_residual else "undersize"   # R0 §1: 시간맥락 잔차 vs 파티클
                    self.filtered_out.append((self.number, dn_mark.last_frame, size_px, reason))
                    return None
            flags.append("weak-bubble")                        # 고립 약기포 → 계수(맥락 근거)
        if size_px > self._merge_suspect:
            flags.append("merge-suspect")
        if size_px <= self._min_accept * 1.2:
            flags.append("boundary-size")
        self._counted.append((lo, hi, dn_mark.last_frame))
        self.accepted_widths.append(size_px)   # D-14: 승격/기각 로직을 통과한 폭만 히스토그램에 반영
        subframe = min(1.0, max(0.0, self._dn_ratio.get(dn_mark.first_frame, 0.0)))
        # 계수 프레임 = 마크 중점(GT 통과프레임 정렬). dedup·방향매칭·subframe은 원시 시각 유지 (D-10 ①)
        event_frame = round((dn_mark.first_frame + dn_mark.last_frame) / 2)
        return BubbleEvent(point=self.number, frame=event_frame,
                           subframe=subframe, flags=tuple(flags),
                           position_px=dn_mark.position_px, size_px=size_px)

    def _prune(self, frame_idx: int) -> None:
        keep = 2 * DIRECTION_WINDOW + DEDUP_WINDOW_FRAMES + 5    # 지연 판정 유예분 보존 (D-2)
        cut = frame_idx - keep
        self._up_marks = [m for m in self._up_marks if m.last_frame >= cut]
        self._consumed_up &= {id(m) for m in self._up_marks}    # 소비 집합을 생존 up과 동기(id 재사용 방지)
        self._counted = [c for c in self._counted if c[2] >= cut]
        self._dn_ratio = {f: r for f, r in self._dn_ratio.items() if f >= frame_idx - 200}


def run_multiway_chapter(
    video_path, cfg: MultiwayConfig, out_dir,
    progress_cb: "Callable[[ProgressEvent], None] | None" = None,
    cancel_event=None,
    snapshot_save_fn: "Callable[[str, np.ndarray], None] | None" = None,
    live_cb: "Callable | None" = None,
    live_control=None,
    live_label: str = "",
) -> ChapterResult:
    """멀티웨이 1챕터(영상 1개) 2-pass 계수. 챕터 독립(매번 새 인스턴스).

    A5: violations/sets_completed는 sequence.analyze로 채워진다. A6:
    suspects(suspects.detect)/suspects_truncated/self_check_messages도 여기서 채워진다.
    time_base는 cfg.recording.fps. 역류 탐지/드롭은 폐기.

    live_cb/live_control: 관찰 전용(L4). 계수 분기·결과에 영향 없음.
    live_label: 페이즈 챕터 표시명(GUI).
    """
    source = VideoSource(video_path)                          # 전체 프레임, scale=1.0
    try:
        fw, fh = source.info.width, source.info.height
        runners = [PointRunner(p, cfg, fw, fh) for p in cfg.points]
        live = LiveEmitter(live_cb, live_control)
        total_hint = source.info.frame_count

        # 1차 패스: 프라이밍
        samples = source.sample_gray(sample_indices(source.info.frame_count))
        for r in runners:
            r.prime(samples)

        # D-16: 흔들림 기준(프라이밍 샘플 전체프레임 중앙값) + 추정기 1회 준비.
        # 불감대(SHAKE_DEADBAND_PX) 이하는 항등 취급 — 현재 저흔들림 영상은 결과가 완전히 그대로다.
        shake_ref = stabilize.reference_background(samples)
        shake_est = stabilize.ShakeEstimator(shake_ref)
        shake_tracker = stabilize.StabilizationTracker()

        # 2차 패스: 전 구간 계수 (프레임 스키핑 없음)
        events: list[BubbleEvent] = []
        counts_live = {r.number: 0 for r in runners}
        processed = 0
        cancelled = False
        start = time.perf_counter()
        last_shake = (0.0, 0.0)   # D-16: stride>1이면 미측정 프레임은 직전 추정치 재사용
        for frame_idx, gray, _bgr in source.frames():
            if frame_idx % stabilize.SHAKE_STRIDE == 0:
                last_shake = shake_est.estimate(gray)
            shift = shake_tracker.record(*last_shake) or (0.0, 0.0)
            for r in runners:
                new_ev = r.process(gray, frame_idx, shift)
                for e in new_ev:
                    counts_live[e.point] = counts_live.get(e.point, 0) + 1
                events.extend(new_ev)
            processed += 1
            elapsed = time.perf_counter() - start
            pfps = processed / elapsed if elapsed > 0 else 0.0
            if progress_cb is not None and processed % 200 == 0:
                progress_cb(ProgressEvent(frame_idx, total_hint, pfps,
                                          len(events), None))
            if live.enabled:
                live.pace()
                live.emit(frame_idx, total_hint, gray, counts_live,
                          live_label or str(video_path), pfps)
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
        for r in runners:
            new_ev = r.flush()
            for e in new_ev:
                counts_live[e.point] = counts_live.get(e.point, 0) + 1
            events.extend(new_ev)

        # D-21: order_gate=True → 순서 맞는 통과만 수락(count error=미계수). False → 구 동작.
        raw_events = events
        if getattr(cfg, "order_gate", True):
            gate = sequence.gate_analyze(
                raw_events, cfg.way, cfg.order_tolerance_frames)
            events = gate.accepted
            # absorbed=0: count error는 미계수라 set 항등식(sum==sets×way+residual)에 불포함
            outcome = sequence.SequenceOutcome(
                violations=list(gate.count_errors),
                sets_completed=gate.sets_completed,
                residual=dict(gate.residual),
                absorbed=0,
            )
        else:
            outcome = sequence.analyze(
                raw_events, cfg.way, cfg.order_tolerance_frames)
            events = raw_events

        # 의심 구간: 원시 통과 스트림 기준(게이트 전) — 물리 마크 휴리스틱 (역류 없음)
        marks_by_point = {r.number: r.dn_marks for r in runners}
        all_suspects, truncated = suspects.detect(
            marks_by_point, raw_events, cfg, cfg.recording.fps)

        # 스냅샷: 각 의심 프레임의 해당 Point 패치를 주입된 sink로 저장 (source 아직 열림)
        if snapshot_save_fn is not None and all_suspects:
            runner_by_num = {r.number: r for r in runners}
            grays = source.sample_gray([s.frame for s in all_suspects])
            saved: list[SuspectMoment] = []
            for idx, (s, gray) in enumerate(zip(all_suspects, grays)):
                rel = f"suspects/s{idx:04d}_p{s.point}_f{s.frame:06d}.png"
                snapshot_save_fn(rel, runner_by_num[s.point].extract(gray))
                saved.append(dataclasses.replace(s, snapshot=rel))
            all_suspects = saved

        counts_auto = {r.number: 0 for r in runners}
        for e in events:
            counts_auto[e.point] += 1
        filtered: list[tuple[int, int, float]] = []
        for r in runners:
            filtered.extend(r.filtered_out)

        # corrections=[]이므로 counts_final == counts_auto → self_check 입력에 counts_auto 사용
        self_check_messages = suspects.self_check(counts_auto, outcome, cfg.way)
        chapter = ChapterResult(
            video_path=str(video_path), recording=cfg.recording,
            counts_auto=counts_auto, corrections=[], events=events,
            violations=outcome.violations, suspects=all_suspects,
            sets_completed=outcome.sets_completed, processed_frames=processed,
            filtered_out=filtered, suspects_truncated=truncated,
            self_check_messages=self_check_messages, cancelled=cancelled,   # A0 정식 후행 필드
            stabilization=StabilizationStats(**shake_tracker.summary()))    # D-16
        return chapter
    finally:
        source.close()


# direction 문자열 → 화면 벡터
_DIR_VEC = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


@dataclass
class PreflightReport:
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)   # D-16: 상태 정보(문제 아님) — 흔들림 σ 등
    size_histogram: dict = field(default_factory=dict)
    foreign_objects: list = field(default_factory=list)  # D-17: 정지 이물 후보 (ForeignObject 리스트)


def _rects_overlap(ca, cb) -> bool:
    """두 회전 사각형(각 4꼭짓점, A0 corners_px 순서) SAT 교차.  맞닿음(=)은 비겹침."""
    axes = []
    for c in (ca, cb):
        # corners_px 순서: [(-u,-n),(-u,+n),(+u,-n),(+u,+n)] → 두 직교 축
        axes.append((c[1][0] - c[0][0], c[1][1] - c[0][1]))   # +n 축
        axes.append((c[2][0] - c[0][0], c[2][1] - c[0][1]))   # +u 축
    for ax, ay in axes:
        pa = [px * ax + py * ay for px, py in ca]
        pb = [px * ax + py * ay for px, py in cb]
        if max(pa) <= min(pb) or max(pb) <= min(pa):
            return False        # 분리축 발견 (맞닿음 포함)
    return True


def _measure_point(cfg: MultiwayConfig, point: PointSpec, frame_w: int, frame_h: int,
                    segments: "list[list[np.ndarray]]") -> dict:
    """구간(segments)별 mark 폭 분포 + 프레임당 흐름축 이동량 Δpx + 밀착 비율 측정.

    D-14: 폭 분포(widths)는 runner.process()/flush()를 그대로 구동해 프로덕션 계수 경로
    (D-11 arc 병합 + 시간맥락 승격/기각 + WEAK_MIN_PX)를 통과한 accepted_widths만 수집한다
    (재구현 금지 — review D-14). 이전에는 runner._counter_dn.feed()로 미필터 원시 마크 폭을
    직접 모아 중공 링의 분할 arc·선행 잔차(19~24px)가 섞여 중앙값이 저편향됐다 — 사용자가
    기포를 작게 그리는 위험 방향에서 ±30% 가드와 우연히 "일치"해 경고가 무력화되는 문제였다.

    D-14 후속(review D-14 후속): 3구간을 이어붙여 1개 러너로 1회만 처리+flush()하면 구간
    경계(~9~15프레임 근방)에서 CONTEXT_WINDOW/DIRECTION_WINDOW가 실제로는 수백 프레임
    떨어진 무관한 다음 구간의 내용과 섞여 잔차가 오승격될 수 있었다. 폭은 구간마다 완전히
    새 PointRunner로 처리+flush()한다(카운터 재생성이 가장 깨끗 — _up_marks/dn_marks는
    append-only라 단순 flush()만으로는 이전 구간 잔재가 남아 오염을 못 막는다).

    중심/밀착 비율 측정(dpx·contact_ratio)은 이번 버그와 무관해(프레임별 지역 계산 + 표본이
    커 경계 아티팩트가 중앙값에 무시 가능) 구간을 이어붙인 기존 방식을 그대로 둔다.
    """
    widths: list[float] = []
    for seg_frames in segments:
        if not seg_frames:
            continue
        runner = PointRunner(point, cfg, frame_w, frame_h)
        runner.prime(seg_frames[: min(len(seg_frames), 30)])
        for i, g in enumerate(seg_frames):
            runner.process(g, i)
        runner.flush()
        widths.extend(runner.accepted_widths)

    frames = [g for seg_frames in segments for g in seg_frames]
    cr = PointRunner(point, cfg, frame_w, frame_h)          # 중심/밀착 비율 전용(폭 집계와 무관)
    cr.prime(frames[: min(len(frames), 30)])
    centroids: list[float] = []
    contact_hits = 0
    for g in frames:
        patch = cr.extract(g)
        fg = cr._mog2.apply(patch, learningRate=0.0) > 0
        if cr._morph is not None:
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, cr._morph) > 0
        ys, xs = np.nonzero(fg)
        if ys.size:
            centroids.append(float(ys.mean()))          # 흐름축(행) 중심
            if float(fg.mean()) > 0.35:                  # 패치 1/3 이상 fg = 밀착/정체 후보
                contact_hits += 1
        else:
            centroids.append(float("nan"))
    deltas = [abs(centroids[k] - centroids[k - 1]) for k in range(1, len(centroids))
              if not (np.isnan(centroids[k]) or np.isnan(centroids[k - 1]))]
    dpx = float(np.median(deltas)) if deltas else 0.0
    contact_ratio = contact_hits / len(frames) if frames else 0.0
    return {"widths": widths, "dpx": dpx, "contact_ratio": contact_ratio}


def _geometry_check(cfg: MultiwayConfig) -> PreflightReport:
    """ROI 겹침·흐름↔방향 정합의 순수 기하 검사 (영상 불필요).  D-12: 겹침은 경고(차단 아님)."""
    rep = PreflightReport()
    w, h = cfg.recording.width, cfg.recording.height
    corners = {p.number: p.corners_px(w, h) for p in cfg.points}
    nums = [p.number for p in cfg.points]
    for a_i in range(len(nums)):
        for b_i in range(a_i + 1, len(nums)):
            na, nb = nums[a_i], nums[b_i]
            if _rects_overlap(corners[na], corners[nb]):
                # D-12: 인접 계수선 ROI가 채널 벽 영역에서 수 px 겹치는 건 정상 배치(게이트 통과 배치도
                # 실전 preflight서 차단됐음) → 차단이 아닌 경고로 완화.
                rep.warnings.append(
                    f"Point {na}과(와) {nb}의 ROI가 겹칩니다 — 의도된 배치인지 "
                    f"확인하세요(겹침 영역이 채널 벽이면 무해)")
    for p in cfg.points:
        ux, uy = p.flow_unit()
        dx, dy = _DIR_VEC[cfg.effective_direction(p)]
        if abs(ux * dx + uy * dy) < 0.1:
            rep.warnings.append(
                f"Point {p.number}: 흐름축과 하류 방향이 거의 수직입니다 — Point별 방향을 지정하세요")
    return rep


def preflight(video_path, cfg: MultiwayConfig, mode_checker=None) -> PreflightReport:
    rep = _geometry_check(cfg)          # 기하 검사 (영상 불필요)
    w, h = cfg.recording.width, cfg.recording.height

    source = VideoSource(video_path)
    try:
        if (source.info.width, source.info.height) != (w, h):
            rep.blocking.append(
                f"해상도가 설정과 다릅니다: 파일 {source.info.width}×{source.info.height} "
                f"≠ 설정 {w}×{h}")
            return rep          # 지오메트리 전제 붕괴 → 측정 검사 생략

        if mode_checker is not None:
            inferred = mode_checker(w, h, source.info.frame_count)
            if inferred != cfg.recording.mode:
                rep.warnings.append(
                    f"녹화 모드 모순: 파일 크기 기준 '{inferred}'로 추정되나 설정은 "
                    f"'{cfg.recording.mode}'입니다")

        # D-13: 기포 크기 미지정이면 크기 관련 검사만 생략(기하·해상도 검사는 위에서 수행)
        if cfg.bubble is None:
            rep.blocking.append("기포 크기가 지정되지 않았습니다 — 3단계에서 지정하세요")
            return rep

        # 측정 검사: 3구간×60 샘플(구간별 독립 처리 — D-14 후속: 시간맥락 창이 실제로는
        # 수백 프레임 떨어진 무관한 다음 구간과 섞이지 않도록 구간 경계를 유지해 넘긴다)
        n = source.info.frame_count
        seg = 60
        starts = [0, max(0, n // 2 - seg // 2), max(0, n - seg)] if n > 0 else [0]
        seen: set[int] = set()
        seg_idxs: list[list[int]] = []
        for s in starts:
            cur = [i for i in range(s, s + seg) if i not in seen]
            seen.update(cur)
            if cur:
                seg_idxs.append(cur)
        segments = [source.sample_gray(idxs) for idxs in seg_idxs]

        # D-16: 흔들림 정보행 — size_histogram과 동일 샘플 재사용(영상 추가 읽기 없음).
        # 중앙값 기준 각 샘플의 위상상관 |shift|로 σ·최대값 추정해 보정 개입 여부를 안내한다.
        shake_frames = [g for seg_frames in segments for g in seg_frames]
        if shake_frames:
            shake_ref = stabilize.reference_background(shake_frames)
            shake_est = stabilize.ShakeEstimator(shake_ref)
            mags = [float(np.hypot(*shake_est.estimate(g))) for g in shake_frames]
            verdict = "보정 불필요" if max(mags) <= stabilize.SHAKE_DEADBAND_PX else "자동 개입"
            rep.info.append(f"흔들림 σ={float(np.std(mags)):.2f}px — {verdict}")

        # D-17: 정지 이물 스캔 — 세그먼트 샘플 재사용(영상 추가 읽기 0).
        # 중앙값 프레임에서 움직이는 기포는 소멸하고 정지물만 잔존, std는 활성 채널 경로 드러냄.
        if cfg.bubble is not None:
            foreign_objs = foreign.scan_frames(shake_frames, cfg, source="preflight")
            if foreign_objs:
                near_objs = [fo for fo in foreign_objs if fo.near_points]
                rep.warnings.append(
                    f"정지 이물 후보 {len(foreign_objs)}개 — 측정은 가능하나 검토 센터에서 태깅 권장 "
                    f"(계수선 밴드±마진 걸치는 것 {len(near_objs)}개)")
            rep.foreign_objects = foreign_objs

        dia_in = min(cfg.bubble.major_px, cfg.bubble.minor_px)  # ±30% 입력 검증용만
        for p in cfg.points:
            _, h_p = patch_dims(p, w, h)
            meas = _measure_point(cfg, p, w, h, segments)
            rep.size_histogram[p.number] = meas["widths"]
            # 밴드 권장 지름 = probe 인정 폭 중앙값 (사용자 BubbleSizeSpec 비의존)
            dia_eff = _effective_diameter_px(meas["widths"])
            if h_p < meas["dpx"] - dia_eff:
                rec = round((meas["dpx"] + dia_eff) / w, 3)
                if meas["widths"]:
                    span_txt = (f"이동량 {meas['dpx']:.0f}px − 측정폭중앙 {dia_eff:.0f}px")
                else:
                    span_txt = f"이동량 {meas['dpx']:.0f}px (폭 샘플 없음)"
                rep.warnings.append(
                    f"Point {p.number}: 밴드가 짧아 빠른 기포를 놓칠 수 있습니다 "
                    f"(길이 {h_p}px < {span_txt}) — length≈{rec} 권장")
            if meas["contact_ratio"] > 0.3:
                rep.warnings.append(
                    f"Point {p.number}: 기포가 뭉쳐 지나갑니다(밀착 {meas['contact_ratio']:.0%}) — "
                    f"상류로 옮기세요")
            widths = meas["widths"]
            if widths:
                med = float(np.median(widths))
                if not (dia_in * 0.7 <= med <= max(cfg.bubble.major_px, cfg.bubble.minor_px) * 1.3):
                    rep.warnings.append(
                        f"Point {p.number}: 측정 크기 중앙값 {med:.0f}px가 입력 기포 크기와 "
                        f"±30% 밖입니다 — 기포 크기를 재측정하세요")
        return rep
    finally:
        source.close()
