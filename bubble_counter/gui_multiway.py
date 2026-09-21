"""멀티웨이 위저드 GUI 탭 — tkinter 렌더 계층(설계 §2).

이 모듈만 tkinter를 import한다. 순수 로직은 gui_view / gui_roi_edit /
gui_multiway_state / gui_graph (TDD 완료)에 있고, 여기서는 위젯 구성·이벤트
배선·그 로직 호출만 한다. WSL엔 tkinter가 없어 import 불가 — 검증은
tests/test_gui_multiway_syntax.py(ast) + Windows 실기 체크리스트.

A/C 트랙 산출물은 생성자 주입 콜백으로 받는다(통합 태스크에서 실제 콜러블 배선).
"""

from __future__ import annotations

import dataclasses
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from types import SimpleNamespace

import numpy as np

from bubble_counter.gui_view import ViewTransform, ppm_p6_bytes
from bubble_counter.gui_roi_edit import (
    handles_for, hit_test, move_point, resize_point, rotate_point, bubble_dims_from_drag,
)
from bubble_counter.gui_multiway_state import (
    WizardStep, WizardState, ChapterList,
    parse_frame_input, parse_target, parse_point_number,
    arrow_for, PreflightItem, PreflightDisplay,
)
from bubble_counter.gui_graph import error_bar_geometry, bar_color, value_label
# 통합 태스크(I1) 배선: A/C 산출물을 실제로 호출하는 임포트는 전부 모듈 상단에 둔다
# (워커 스레드 안에서 임포트하지 않는다 — 전체 코드리뷰 F5 교훈).
from bubble_counter.multiway import runner, sequence
from bubble_counter.multiway.engine import preflight
from bubble_counter.multiway.has_u2 import HAS_U2_MAX_FPS, infer_mode
from bubble_counter.multiway.stabilize import SHAKE_BANNER_PX
from bubble_counter.multiway.profiles import ProfileStore
from bubble_counter.live import LiveControl, queue_put_drop
from bubble_counter.resmon import Snapshot, format_line
from bubble_counter.video_io import VideoSource

# 표준 fps 스텝(표시용). 권위 있는 후보/최대치는 통합 시 has_u2 표에서 온다.
_FPS_STEPS = (60, 100, 150, 200, 250, 300, 400, 500, 800, 1000, 1500, 2000)
# 표 미주입 시 최소 폴백(이번 6way 영상 행). 통합에서 has_u2.HAS_U2_MAX_FPS로 대체.
_HAS_U2_FALLBACK = {(1280, 768): {"dma": 300, "memory": 500}}

_STEP_TITLES = {
    WizardStep.RECORDING: "1단계 · 녹화 설정",
    WizardStep.CHAPTERS: "2단계 · 챕터 목록",
    WizardStep.ROI: "3단계 · ROI 배치",
    WizardStep.PREFLIGHT: "4단계 · 프리플라이트",
    WizardStep.RUN: "5단계 · 측정 실행",
    WizardStep.RESULTS: "6단계 · 결과·검토",
}


# ---------------------------------------------------------------------------
# 통합 배선(Track I1) — A/C 산출물 ↔ B 주입 지점 어댑터.
# "B 주입 지점 배선 체크리스트" 4건을 전부 여기서 해소한다.
# ---------------------------------------------------------------------------


def _progress_view(ev):
    """체크리스트 1항: A 엔진 pipeline.ProgressEvent → B 표시모델(fraction/message/resource).

    B의 `_apply_progress()`는 getattr(ev, "fraction"/"message"/"resource", None)로
    방어적으로 읽으므로 이 속성 3개만 채우면 된다.
    """
    fraction = (ev.frame_idx / ev.total_frames) if ev.total_frames else 0.0
    message = f"{ev.count}개 · {ev.processing_fps:.0f}fps"
    resource = ""
    if getattr(ev, "cpu_pct", None) is not None or getattr(ev, "rss_mb", None) is not None:
        snap = Snapshot(ev.cpu_pct, ev.cpu_cores, ev.rss_mb, ev.sys_ram_pct, ev.gpu_pct)
        resource = format_line(snap, os.cpu_count() or 1)
    return SimpleNamespace(fraction=fraction, message=message, resource=resource)


def make_preflight_fn(video_provider):
    """B의 프리플라이트 표시모델(list[PreflightItem])로 변환.

    주의(rev-B9 발견 반영): B의 PreflightDisplay.from_items는 .severity/.message
    속성을 읽으므로 튜플이 아니라 PreflightItem 인스턴스를 반환해야 한다.
    """
    from bubble_counter.gui_multiway_state import PreflightItem

    def preflight_fn(cfg):
        rep = preflight(video_provider(), cfg, mode_checker=infer_mode)
        return ([PreflightItem("block", m) for m in rep.blocking]
                + [PreflightItem("warn", m) for m in rep.warnings]
                + [PreflightItem("info", m) for m in rep.info])  # D-16: 흔들림 σ 등
    return preflight_fn


def make_run_fn():
    """B의 run_fn(cfg, chapters, phase_name, progress_cb, cancel_event, live_*) 계약.

    결정(2026-07-10 리뷰 M-1): phase_name은 GUI 2단계 입력값을 B가 전달(사용자 편집 존중).
    out_root는 기존 단일선 모드 관례대로 첫 영상의 부모폴더로 유도(별도 폴더 선택 UI 없음).
    live_cb/live_control은 trailing optional — 관찰 전용(L4).
    """
    def run_fn(cfg, chapters, phase_name, progress_cb, cancel_event,
               live_cb=None, live_control=None):
        videos = [c.path for c in chapters]
        out_root = Path(videos[0]).parent
        def adapt(ev):   # pipeline.ProgressEvent → B 표시 모델 (체크리스트 1항)
            progress_cb(_progress_view(ev))
        return runner.run_phase(videos, cfg, phase_name, out_root,
                                progress_cb=adapt, cancel_event=cancel_event,
                                live_cb=live_cb, live_control=live_control)
    return run_fn


def make_recount_fn(cfg):
    """보정 반영 재계산: ChapterResult → 새 ChapterResult (자동값 원본 보존)."""
    def recount_fn(ch, correction):
        corrections = list(ch.corrections) + [correction]
        if getattr(cfg, "order_gate", True):
            # D-21: 게이트 ON이면 보정 후에도 순서 게이트로 set/count error 재계산.
            # counts_auto는 원본 수락 개수 유지(보정은 counts_final만 가산).
            gate = sequence.apply_corrections_gated(
                ch.events, corrections, cfg.way, cfg.order_tolerance_frames)
            return dataclasses.replace(
                ch, corrections=corrections,
                violations=list(gate.count_errors),
                sets_completed=gate.sets_completed)
        outcome = sequence.apply_corrections(
            ch.events, corrections, cfg.way, cfg.order_tolerance_frames)
        return dataclasses.replace(
            ch, corrections=corrections,
            violations=outcome.violations, sets_completed=outcome.sets_completed)
    return recount_fn


def _make_frame_provider(chapters_provider):
    """VideoSource 기반 frame_provider(frame_index) -> BGR ndarray. 직접 바인딩(체크리스트 4항).

    ROI 배치(3단계)는 항상 1번 챕터 영상을 기준으로 한다(단일선 모드의 "대표 영상"
    관례와 동일 — 별도 영상 선택 UI 없음). 같은 영상을 반복 요청하는 동안은 VideoSource를
    재사용하고, 챕터 목록의 1번째가 바뀌면 새로 연다.
    """
    cache: dict = {"path": None, "source": None}

    def frame_provider(frame_index: int):
        chapters = list(chapters_provider())
        if not chapters:
            raise ValueError("열린 챕터가 없습니다")
        path = chapters[0].path
        if cache["path"] != path:
            if cache["source"] is not None:
                cache["source"].close()
            cache["source"] = VideoSource(path)
            cache["path"] = path
        source = cache["source"]
        source.seek(frame_index)
        for _idx, _gray, bgr in source.frames(with_color=True):
            return bgr
        raise ValueError(f"프레임 {frame_index}을(를) 읽을 수 없습니다: {path}")
    return frame_provider


def _make_open_frame_fn(master, chapters_provider):
    """±10 프레임 뷰어 콜백. frame_provider와 함께 직접 바인딩(체크리스트 4항).

    D-6/B-15 계약 정합: `_on_open_video()`는 `self._open_frame_fn()`을 0-인자로
    호출(1번 챕터를 프레임 0에서 열기 — 리뷰 확정 D-6), `_open_violation()`은
    `self._open_frame_fn(ch.video_path, frame)`을 2-인자로 호출(그 영상의 그 프레임
    부근 열기 — B-15 계약)한다. 같은 `self._open_frame_fn` 속성을 두 호출부가 다른
    인자 수로 쓰므로, 인자를 전부 선택적으로 두어 하나의 콜백이 두 계약을 모두 만족시킨다.
    """
    def open_frame_fn(video_path=None, frame=None):
        if video_path is None:
            chapters = list(chapters_provider())
            if not chapters:
                messagebox.showinfo(
                    "영상 열기",
                    "미리 볼 영상이 없습니다.\n\n"
                    "진행 순서:\n"
                    "  1단계 — 해상도·fps 선택 후 [다음]\n"
                    "  2단계 — [영상 추가]로 측정할 파일(챕터) 선택\n"
                    "  다시 1단계로 돌아오면 [영상 열기]로 첫 챕터 미리보기\n\n"
                    "※ 챕터 = 측정에 넣을 영상 파일 1개")
                return
            video_path = chapters[0].path
        _FrameViewer(master, video_path, frame if frame is not None else 0)
    return open_frame_fn


class _FrameViewer(tk.Toplevel):
    """프레임 중심 ±10 뷰어 — 영상 열기(D-6)·순서 위반 더블클릭(B-15) 공용 팝업."""

    def __init__(self, master, video_path: str, center_frame: int) -> None:
        super().__init__(master)
        self.title(f"프레임 보기 — {os.path.basename(video_path)}")
        self._source = VideoSource(video_path)
        total = self._source.info.frame_count
        self._lo = max(0, center_frame - 10)
        self._hi = min(total - 1, center_frame + 10) if total > 0 else center_frame + 10
        self._frame = min(max(self._lo, center_frame), self._hi)
        self._photo = None

        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=4, pady=4)
        ttk.Button(nav, text="◀", width=3, command=lambda: self._go(-1)).pack(side="left")
        self._pos_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._pos_var).pack(side="left", padx=8)
        ttk.Button(nav, text="▶", width=3, command=lambda: self._go(+1)).pack(side="left")
        self._canvas = tk.Canvas(self, width=640, height=480, background="#202020",
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._render()

    def _go(self, delta: int) -> None:
        self._frame = max(self._lo, min(self._hi, self._frame + delta))
        self._render()

    def _render(self) -> None:
        self._source.seek(self._frame)
        bgr = None
        for _idx, _gray, b in self._source.frames(with_color=True):
            bgr = b
            break
        self._pos_var.set(f"프레임 {self._frame} ({self._lo}~{self._hi})")
        if bgr is None:
            return
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self._photo = tk.PhotoImage(data=ppm_p6_bytes(rgb), format='PPM')
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _on_close(self) -> None:
        self._source.close()
        self.destroy()


def attach_multiway_tab(notebook: ttk.Notebook, **kwargs) -> "MultiwayTab":
    """기존 노트북에 '멀티웨이' 탭을 추가한다(gui.py의 진입점이 호출).

    gui.py는 attach_multiway_tab(notebook)만 호출한다(수정 범위 밖 — 통합 태스크 I1).
    kwargs로 명시 주입하지 않는 한 ProfileStore·has_u2 표·프리플라이트/실행/재계산
    어댑터·프레임 콜백(frame_provider/open_frame_fn)을 여기서 기본 배선한다.
    """
    wired: dict = dict(kwargs)
    tab_box: dict = {}   # MultiwayTab 인스턴스가 생기기 전에 그 상태를 참조해야 하는
                         # 콜백(preflight_fn/recount_fn/frame_provider/open_frame_fn)의
                         # 지연 바인딩 상자 — tkinter는 단일 스레드라 경합 없음.

    def _chapters():
        return list(tab_box["tab"].chapters)

    wired.setdefault("has_u2_table", HAS_U2_MAX_FPS)
    wired.setdefault("profile_store", ProfileStore())
    wired.setdefault("preflight_fn", make_preflight_fn(lambda: _chapters()[0].path))
    wired.setdefault("run_fn", make_run_fn())
    wired.setdefault(
        "recount_fn",
        # D-13 후속 Critical(review-D13-verdict.md): _compose_config()를 여기서 재조립하면
        # "◀ 이전 → ROI를 프레임 밖으로 이동 → 다음 ▶"으로 RESULTS에 되돌아온 뒤 보정을
        # 누를 때 stale ROI로 ValueError가 재현된다. 재계수는 그 실행을 만든 cfg를 그대로
        # 쓴다(_start_run 성공 경로에서 보관 — _correct()가 없으면 배너 후 중단).
        lambda ch, corr: make_recount_fn(tab_box["tab"]._last_run_cfg)(ch, corr))
    wired.setdefault("frame_provider", _make_frame_provider(_chapters))
    wired.setdefault("open_frame_fn", _make_open_frame_fn(notebook, _chapters))

    tab = MultiwayTab(notebook, **wired)
    tab_box["tab"] = tab
    notebook.add(tab, text="멀티웨이")
    return tab


class MultiwayTab(ttk.Frame):
    """멀티웨이 6단계 위저드 탭. 단계별 패널을 스왑한다."""

    def __init__(self, master, *, frame_provider=None, preflight_fn=None,
                 run_fn=None, recount_fn=None, profile_store=None,
                 open_frame_fn=None, has_u2_table=None) -> None:
        super().__init__(master, padding=8)
        self._frame_provider = frame_provider
        self._preflight_fn = preflight_fn
        self._run_fn = run_fn
        self._recount_fn = recount_fn
        self._profile_store = profile_store
        self._open_frame_fn = open_frame_fn
        self._has_u2_table = has_u2_table or _HAS_U2_FALLBACK

        self.state = WizardState()
        self.chapters = ChapterList()
        self.points: list = []              # list[PointSpec]
        self.bubble = None                  # BubbleSizeSpec | None
        self.global_direction = "+x"
        self.phase_name = ""                # 2단계에서 확정, run_fn 3번째 인자로 전달
        self.frame_index = 0
        self.total_frames = 1
        self.results = None                 # PhaseResult | None (6단계)

        self._queue: queue.Queue = queue.Queue()
        self._cancel_event: threading.Event | None = None
        self._run_active: bool = False      # 5단계 실행 중 — 이탈/재클릭 가드(rev-B14)
        self._last_run_cfg = None           # D-13후속: 워커 __done__ 성공분기에서 cfg 보관(재계수용)
        self._live_panel = None
        self._live_control = LiveControl(1.0)
        self._live_queue: queue.Queue = queue.Queue(maxsize=2)

        self._build_shell()
        self._show_step()

    # ---- 셸: 제목 + 단계 패널 컨테이너 + 이전/다음 ----
    def _build_shell(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._title_var = tk.StringVar()
        ttk.Label(self, textvariable=self._title_var,
                  font=("", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self._panel = ttk.Frame(self)
        self._panel.grid(row=1, column=0, sticky="nsew")
        self._panel.columnconfigure(0, weight=1)
        self._panel.rowconfigure(0, weight=1)
        nav = ttk.Frame(self)
        nav.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._back_btn = ttk.Button(nav, text="◀ 이전", command=self._on_back)
        self._back_btn.pack(side="left")
        self._next_btn = ttk.Button(nav, text="다음 ▶", command=self._on_next)
        self._next_btn.pack(side="right")

    def _show_step(self) -> None:
        for w in self._panel.winfo_children():
            w.destroy()
        self._title_var.set(_STEP_TITLES[self.state.step])
        builder = {
            WizardStep.RECORDING: self._build_recording,
            WizardStep.CHAPTERS: self._build_chapters,
            WizardStep.ROI: self._build_roi,
            WizardStep.PREFLIGHT: self._build_preflight,
            WizardStep.RUN: self._build_run,
            WizardStep.RESULTS: self._build_results,
        }[self.state.step]
        builder(self._panel)
        self._back_btn.state(["!disabled"] if self.state.step > WizardStep.RECORDING else ["disabled"])
        self._next_btn.state(["!disabled"] if self.state.can_advance()[0] else ["disabled"])

    def _on_next(self) -> None:
        ok, msg = self.state.can_advance()
        if not ok:
            messagebox.showwarning("진행 불가", msg)
            return
        self.state.advance()
        self._show_step()

    def _on_back(self) -> None:
        if self._run_active:
            # rev-B14 Critical: 실행 중 _show_step()이 _run_bar 등을 destroy하면
            # 폴링이 무음으로 죽고 워커가 고아화된다 — 이전 이동 자체를 차단.
            messagebox.showwarning("이동 불가", "측정이 실행 중입니다 — 취소 후 이동하세요")
            return
        self.state.back()
        self._show_step()

    def _refresh_next(self) -> None:
        self._next_btn.state(["!disabled"] if self.state.can_advance()[0] else ["disabled"])

    # ---- 1단계: 녹화 설정(HAS-U2 표에서 선택) ----
    def _build_recording(self, parent) -> None:
        self._mode_var = tk.StringVar(value="dma")
        modes = ttk.Frame(parent)
        modes.grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(modes, text="모드:").pack(side="left")
        for label, val in (("DMA", "dma"), ("내장메모리", "memory")):
            ttk.Radiobutton(modes, text=label, value=val, variable=self._mode_var,
                            command=self._on_reco_changed).pack(side="left", padx=4)

        cols = ("res", "dma", "memory")
        self._reco_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        for c, t in zip(cols, ("해상도", "DMA 최대fps", "메모리 최대fps")):
            self._reco_tree.heading(c, text=t)
            self._reco_tree.column(c, width=130, anchor="center")
        for (w, h), fps in sorted(self._has_u2_table.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
            self._reco_tree.insert("", "end", iid=f"{w}x{h}",
                                   values=(f"{w}×{h}", fps["dma"], fps["memory"]))
        self._reco_tree.grid(row=1, column=0, sticky="nsew", pady=4)
        self._reco_tree.bind("<<TreeviewSelect>>", lambda e: self._on_reco_changed())

        fps_row = ttk.Frame(parent)
        fps_row.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(fps_row, text="촬영 fps:").pack(side="left")
        self._fps_combo = ttk.Combobox(fps_row, width=8, state="readonly")
        self._fps_combo.pack(side="left", padx=4)
        self._fps_combo.bind("<<ComboboxSelected>>", lambda e: self._on_reco_changed())
        self._reco_info = tk.StringVar(value="해상도 행과 fps를 선택하세요.")
        ttk.Label(parent, textvariable=self._reco_info, foreground="#666").grid(
            row=3, column=0, sticky="w", pady=4)
        parent.rowconfigure(1, weight=1)

        # [영상 열기] — 2단계 챕터(영상 파일) 추가 후 1번 챕터 미리보기.
        open_row = ttk.Frame(parent)
        open_row.grid(row=4, column=0, sticky="w", pady=(4, 0))
        self._open_video_btn = ttk.Button(open_row, text="영상 열기(미리보기)",
                                          command=self._on_open_video)
        self._open_video_btn.pack(side="left")
        ttk.Label(open_row,
                  text="  ← 2단계에서 영상 추가 후에 사용 (측정 파일 선택이 아님)",
                  foreground="#666").pack(side="left")
        self._refresh_open_video_btn()

    def _refresh_open_video_btn(self) -> None:
        self._open_video_btn.state(
            ["!disabled"] if self.state.video_open_enabled() else ["disabled"])

    def _on_open_video(self) -> None:
        if self._open_frame_fn is None:
            messagebox.showinfo("영상 열기", "미연결 — 통합 태스크에서 배선됩니다")
            return
        self._open_frame_fn()

    def _on_reco_changed(self) -> None:
        sel = self._reco_tree.selection()
        if not sel:
            self.state.recording = None
            self._refresh_next()
            self._refresh_open_video_btn()
            return
        w, h = (int(x) for x in sel[0].split("x"))
        mode = self._mode_var.get()
        max_fps = self._has_u2_table[(w, h)][mode]
        cands = [str(s) for s in _FPS_STEPS if s <= max_fps]
        self._fps_combo["values"] = cands
        if self._fps_combo.get() not in cands:
            self._fps_combo.set(cands[-1] if cands else "")
        try:
            fps = float(self._fps_combo.get())
        except ValueError:
            self.state.recording = None
            self._refresh_next()
            self._refresh_open_video_btn()
            return
        from bubble_counter.multiway.model import RecordingSpec
        self.state.recording = RecordingSpec(w, h, fps, mode)
        self._reco_info.set(f"선택: {w}×{h} @ {fps:g}fps · {mode}  (이 값이 모든 시간 환산 기준)")
        self._refresh_next()
        self._refresh_open_video_btn()

    # ---- 2단계: 챕터 목록 ----
    def _build_chapters(self, parent) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Button(bar, text="영상 추가",
                   command=self._add_chapter_files).pack(side="left")
        ttk.Button(bar, text="선택 제거",
                   command=self._remove_chapter).pack(side="left", padx=4)
        ttk.Label(bar, text="페이즈 이름:").pack(side="left", padx=(12, 2))
        self._phase_var = tk.StringVar(value=self.phase_name or self.chapters.phase_name())
        # 단계 전환 시 이 패널은 파괴되므로 페이즈명을 탭 상태(self.phase_name)에 상시 반영.
        # 5단계 run_fn 호출은 위젯이 아니라 self.phase_name을 읽는다(M-1 계약).
        self.phase_name = self._phase_var.get()
        self._phase_var.trace_add(
            "write", lambda *_: setattr(self, "phase_name", self._phase_var.get()))
        ttk.Entry(bar, textvariable=self._phase_var, width=24).pack(side="left")

        cols = ("path", "res", "frames", "warn")
        self._chap_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        for c, t, wdt in zip(cols, ("파일", "해상도", "프레임", "경고"), (280, 90, 80, 200)):
            self._chap_tree.heading(c, text=t)
            self._chap_tree.column(c, width=wdt, anchor="w")
        self._chap_tree.grid(row=1, column=0, sticky="nsew", pady=4)
        parent.rowconfigure(1, weight=1)
        self._refresh_chapters()

    def _add_chapter_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="챕터 영상 선택", filetypes=[("영상", "*.avi *.mp4 *.mkv"), ("모두", "*.*")])
        for p in paths:
            w, h, n = self._probe_video(p)
            self.chapters.add(p, w, h, n)
        self.state.chapter_count = len(self.chapters)
        self._phase_var.set(self.chapters.phase_name())
        self._refresh_chapters()
        self._refresh_next()
        # 1단계 [영상 열기] 활성 조건에 chapter_count 포함

    def _remove_chapter(self) -> None:
        sel = self._chap_tree.selection()
        if not sel:
            return
        self.chapters.remove(self._chap_tree.index(sel[0]))
        self.state.chapter_count = len(self.chapters)
        self._refresh_chapters()
        self._refresh_next()

    def _probe_video(self, path: str):
        """파일 실해상도·프레임 수. frame_provider 미주입 시 (0,0,0)."""
        if self._frame_provider is None:
            return (0, 0, 0)
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return (w, h, n)
        except Exception:
            return (0, 0, 0)

    def _refresh_chapters(self) -> None:
        import os
        for row in self._chap_tree.get_children():
            self._chap_tree.delete(row)
        rec = self.state.recording
        res_bad = set(self.chapters.resolution_mismatches(rec)) if rec else set()
        mode_bad = set(self.chapters.mode_contradictions(rec)) if rec else set()
        for i, c in enumerate(self.chapters):
            warns = []
            if i in res_bad:
                warns.append("해상도 불일치")
            if i in mode_bad:
                warns.append("2GB 초과인데 메모리 모드")
            self._chap_tree.insert(
                "", "end",
                values=(os.path.basename(c.path), f"{c.width}×{c.height}",
                        c.n_frames, " · ".join(warns)))

    # ---- 3단계: ROI 편집기 ----
    def _build_roi(self, parent) -> None:
        self._roi_mode = "select"           # "select" | "draw" | "bubble"
        self._selected = None               # PointSpec.number | None
        self._drag = None                   # 진행 중 드래그 상태 dict
        self._bubble_shape = "circle"
        self._half_toggle = tk.BooleanVar(value=False)
        self._view = None                   # ViewTransform (첫 렌더에서 fit)
        self._photo = None                  # PhotoImage 참조 유지(GC 방지)

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        self._build_roi_toolbar(parent)

        self._canvas = tk.Canvas(parent, background="#202020", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", lambda e: self._render_roi())
        self._canvas.bind("<MouseWheel>", self._on_wheel)               # Windows/Mac
        self._canvas.bind("<Button-4>", lambda e: self._on_wheel_linux(e, 1.1))
        self._canvas.bind("<Button-5>", lambda e: self._on_wheel_linux(e, 1 / 1.1))
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)        # 휠클릭 패닝
        self._canvas.bind("<B2-Motion>", self._on_pan_move)
        self._canvas.bind("<Double-Button-1>", self._on_double)
        self._canvas.bind_all("<Delete>", self._on_delete)

    def _build_roi_toolbar(self, parent) -> None:
        tb = ttk.Frame(parent)
        tb.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for label, mode in (("선택", "select"), ("그리기", "draw"), ("기포 크기 재기", "bubble")):
            ttk.Button(tb, text=label,
                       command=lambda m=mode: self._set_mode(m)).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        # 프레임 탐색: ±1/±10/±100 + 직접 입력 (시간 슬라이더 없음 — 프레임이 1차 좌표)
        for step in (-100, -10, -1, +1, +10, +100):
            ttk.Button(tb, text=f"{step:+d}", width=4,
                       command=lambda s=step: self._step_frame(s)).pack(side="left")
        self._frame_var = tk.StringVar(value="0")
        ent = ttk.Entry(tb, textvariable=self._frame_var, width=8)
        ent.pack(side="left", padx=4)
        ent.bind("<Return>", lambda e: self._goto_frame())
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        # 전역 방향 4버튼
        for d in ("+x", "-x", "+y", "-y"):
            ttk.Button(tb, text=arrow_for(d), width=3,
                       command=lambda dd=d: self._set_global_dir(dd)).pack(side="left")
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Checkbutton(tb, text="반쪽 기포도 셈", variable=self._half_toggle).pack(side="left")
        # 프로파일
        self._profile_combo = ttk.Combobox(tb, width=14, state="readonly",
                                            values=self._profile_names())
        self._profile_combo.pack(side="left", padx=(8, 2))
        self._profile_combo.bind("<<ComboboxSelected>>", lambda e: self._load_profile())
        ttk.Button(tb, text="현재 배치 저장", command=self._save_profile).pack(side="left")

    def _set_mode(self, mode: str) -> None:
        self._roi_mode = mode

    def _set_global_dir(self, d: str) -> None:
        self.global_direction = d
        self._render_roi()

    # ---- 프레임 탐색 ----
    def _step_frame(self, delta: int) -> None:
        self.frame_index = max(0, min(self.total_frames - 1, self.frame_index + delta))
        self._frame_var.set(str(self.frame_index))
        self._render_roi()

    def _goto_frame(self) -> None:
        try:
            self.frame_index = parse_frame_input(self._frame_var.get(), self.total_frames)
        except ValueError as e:
            messagebox.showwarning("프레임 번호", str(e))
            self._frame_var.set(str(self.frame_index))
            return
        self._render_roi()

    # ---- 렌더 ----
    def _current_frame_rgb(self):
        """현재 프레임 RGB ndarray. frame_provider 미주입 시 회색 배경."""
        rec = self.state.recording
        w = rec.width if rec else 1280
        h = rec.height if rec else 768
        if self._frame_provider is None:
            return np.full((h, w, 3), 64, dtype=np.uint8)
        bgr = self._frame_provider(self.frame_index)
        self.total_frames = max(self.total_frames, self.frame_index + 1)
        return np.ascontiguousarray(bgr[:, :, ::-1])   # BGR→RGB

    def _render_roi(self) -> None:
        cw, ch = self._canvas.winfo_width(), self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        rgb = self._current_frame_rgb()
        fh, fw = rgb.shape[:2]
        if self._view is None:
            self._view = ViewTransform.fit(fw, fh, cw, ch)
        vt = self._view
        sw, sh = max(1, round(fw * vt.scale)), max(1, round(fh * vt.scale))
        import cv2
        scaled = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        self._photo = tk.PhotoImage(data=ppm_p6_bytes(np.ascontiguousarray(scaled)),
                                    format='PPM')
        self._canvas.delete("all")
        self._canvas.create_image(vt.tx, vt.ty, anchor="nw", image=self._photo)
        self._draw_overlays(fw, fh)

    def _draw_overlays(self, fw: int, fh: int) -> None:
        vt = self._view
        for p in self.points:
            pts = [vt.image_to_screen(x, y) for (x, y) in p.corners_px(fw, fh)]
            # corners 순서 [0]=(-u,-n)[1]=(-u,+n)[3]=(+u,+n)[2]=(+u,-n) → 사각형 순회
            poly = [pts[0], pts[1], pts[3], pts[2]]
            flat = [v for xy in poly for v in xy]
            sel = (p.number == self._selected)
            self._canvas.create_polygon(*flat, outline="#4fc3f7" if sel else "#ffd54f",
                                        width=2, fill="")
            cxp, cyp = vt.image_to_screen(p.cx * fw, p.cy * fh)
            self._canvas.create_text(cxp, cyp, text=str(p.number),
                                     fill="#ffffff", font=("", 12, "bold"))
            # 하류 화살표(effective direction)
            d = p.direction or self.global_direction
            self._canvas.create_text(cxp, cyp + 16, text=arrow_for(d), fill="#ffffff")
            if sel:
                h = handles_for(p, fw, fh)
                for (hx, hy) in h.corners:
                    sx, sy = vt.image_to_screen(hx, hy)
                    self._canvas.create_rectangle(sx - 4, sy - 4, sx + 4, sy + 4,
                                                  fill="#4fc3f7", outline="")
                rx, ry = vt.image_to_screen(*h.rotate)
                self._canvas.create_oval(rx - 5, ry - 5, rx + 5, ry + 5,
                                         fill="#81c784", outline="")

    # ---- 줌/팬 ----
    def _on_wheel(self, event) -> None:
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        if self._view:
            self._view.zoom_at(event.x, event.y, factor)
            self._render_roi()

    def _on_wheel_linux(self, event, factor) -> None:
        if self._view:
            self._view.zoom_at(event.x, event.y, factor)
            self._render_roi()

    def _on_pan_start(self, event) -> None:
        self._pan_anchor = (event.x, event.y)

    def _on_pan_move(self, event) -> None:
        if self._view and getattr(self, "_pan_anchor", None):
            self._view.pan(event.x - self._pan_anchor[0], event.y - self._pan_anchor[1])
            self._pan_anchor = (event.x, event.y)
            self._render_roi()

    # ---- 마우스 좌클릭: 모드별 ----
    def _img_xy(self, event):
        return self._view.screen_to_image(event.x, event.y)

    def _frame_wh(self):
        rec = self.state.recording
        return (rec.width, rec.height) if rec else (1280, 768)

    def _on_press(self, event) -> None:
        if self._view is None:
            return
        ix, iy = self._img_xy(event)
        if self._roi_mode == "draw":
            self._drag = {"kind": "draw", "start": (ix, iy), "cur": (ix, iy)}
        elif self._roi_mode == "bubble":
            self._drag = {"kind": "bubble", "start": (ix, iy), "cur": (ix, iy)}
        else:  # select
            fw, fh = self._frame_wh()
            radius = 6 / self._view.scale
            hit = hit_test(self.points, (ix, iy), fw, fh, radius)
            self._selected = hit.number if hit.kind != "none" else None
            self._drag = {"kind": hit.kind, "number": hit.number,
                          "corner": hit.corner, "last": (ix, iy)} if hit.kind != "none" else None
            self._render_roi()

    def _on_motion(self, event) -> None:
        if self._drag is None or self._view is None:
            return
        ix, iy = self._img_xy(event)
        fw, fh = self._frame_wh()
        kind = self._drag["kind"]
        if kind in ("draw", "bubble"):
            self._drag["cur"] = (ix, iy)
            self._render_roi()
            self._draw_rubber(kind)
            return
        p = self._point(self._drag["number"])
        if p is None:
            return
        if kind == "body":
            dx = ix - self._drag["last"][0]
            dy = iy - self._drag["last"][1]
            self._replace_point(move_point(p, dx * fw / fw, dy * fh / fh, fw, fh))
            self._drag["last"] = (ix, iy)
        elif kind == "corner":
            self._replace_point(resize_point(p, self._drag["corner"], (ix, iy), fw, fh))
        elif kind == "rotate":
            self._replace_point(rotate_point(p, (ix, iy), fw, fh))
        self._render_roi()

    def _draw_rubber(self, kind: str) -> None:
        vt = self._view
        x0, y0 = vt.image_to_screen(*self._drag["start"])
        x1, y1 = vt.image_to_screen(*self._drag["cur"])
        if kind == "draw":
            self._canvas.create_rectangle(x0, y0, x1, y1, outline="#ffd54f", dash=(3, 2))
        else:  # bubble: 원/타원 미리보기
            self._canvas.create_oval(x0, y0, x1, y1, outline="#81c784", dash=(3, 2))

    def _on_release(self, event) -> None:
        if self._drag is None or self._view is None:
            return
        kind = self._drag["kind"]
        ix, iy = self._img_xy(event)
        fw, fh = self._frame_wh()
        if kind == "draw":
            self._finish_draw(self._drag["start"], (ix, iy), fw, fh)
        elif kind == "bubble":
            self._finish_bubble(self._drag["start"], (ix, iy))
        elif kind in ("body", "corner", "rotate"):
            # D-13후속(review-D13-verdict.md Critical-1): ROI 이동/크기변경/회전 →
            # 프리플라이트 재통과 없이 전진 불가(상태 위생).
            self.state.invalidate_preflight()
        self._drag = None
        self.state.roi_count = len(self.points)
        self._refresh_next()
        self._render_roi()

    def _finish_draw(self, start, end, fw, fh) -> None:
        from bubble_counter.multiway.model import PointSpec
        cx = (start[0] + end[0]) / 2 / fw
        cy = (start[1] + end[1]) / 2 / fh
        length = max(0.005, abs(end[0] - start[0]) / fw)
        width = max(0.005, abs(end[1] - start[1]) / fh)
        existing = [p.number for p in self.points]
        popup = _RoiPopup(self, existing)
        if popup.result is None:
            return
        number, direction, target = popup.result
        try:
            p = PointSpec(number, min(1.0, cx), min(1.0, cy), min(1.0, length),
                          min(1.0, width), 0.0, direction, target)
        except ValueError as e:
            messagebox.showwarning("ROI", str(e))
            return
        self.points.append(p)
        self._selected = number
        self.state.invalidate_preflight()   # D-13후속: ROI 추가 → 재프리플라이트 강제

    def _finish_bubble(self, start, end) -> None:
        from bubble_counter.multiway.model import BubbleSizeSpec
        dx, dy = end[0] - start[0], end[1] - start[1]
        maj, mnr = bubble_dims_from_drag(self._bubble_shape, dx, dy)
        try:
            self.bubble = BubbleSizeSpec(self._bubble_shape, maj, mnr,
                                         count_half_bubbles=self._half_toggle.get())
        except ValueError as e:
            messagebox.showwarning("기포 크기", str(e))
            return
        self.state.invalidate_preflight()   # D-13후속: 기포 크기 변경 → 프리플라이트 크기검사 재통과 필요
        messagebox.showinfo("기포 크기",
                            f"{self._bubble_shape}: {maj:g}×{mnr:g}px"
                            f"{' · 반쪽 포함' if self._half_toggle.get() else ''}")

    def _on_double(self, event) -> None:
        if self._selected is None:
            return
        p = self._point(self._selected)
        existing = [q.number for q in self.points if q.number != p.number]
        popup = _RoiPopup(self, existing, preset=(p.number, p.direction, p.target))
        if popup.result is None:
            return
        from dataclasses import replace
        number, direction, target = popup.result
        self._replace_point(replace(p, number=number, direction=direction, target=target),
                            old_number=self._selected)
        self._selected = number
        self._render_roi()

    def _on_delete(self, event) -> None:
        if self.state.step != WizardStep.ROI or self._selected is None:
            return
        self.points = [p for p in self.points if p.number != self._selected]
        self._selected = None
        self.state.roi_count = len(self.points)
        self.state.invalidate_preflight()   # D-13후속: ROI 삭제 → 재프리플라이트 강제
        self._refresh_next()
        self._render_roi()

    # ---- points 헬퍼 ----
    def _point(self, number):
        return next((p for p in self.points if p.number == number), None)

    def _replace_point(self, new_point, old_number=None) -> None:
        target = old_number if old_number is not None else new_point.number
        self.points = [new_point if p.number == target else p for p in self.points]

    # ---- 프로파일 ----
    def _profile_names(self):
        if self._profile_store is None:
            return []
        try:
            return list(self._profile_store.names())
        except (KeyError, IOError):        # L-2 계약: 콤보 채우기 실패는 빈 목록으로
            return []

    def _load_profile(self) -> None:
        if self._profile_store is None:
            messagebox.showinfo("프로파일", "미연결 — 통합 태스크에서 배선됩니다")
            return
        name = self._profile_combo.get()
        # L-2 계약: ProfileStore.load(없는 이름)→KeyError, save 실패→IOError.
        # 둘 다 잡아 안내(무음 실패 금지). 두 예외를 함께 잡아 핸들러 대칭 유지.
        try:
            cfg = self._profile_store.load(name)
        except (KeyError, IOError) as e:
            messagebox.showwarning("프로파일 불러오기 실패", f"'{name}': {e}")
            return
        self.points = list(cfg.points)
        self.bubble = cfg.bubble
        self.global_direction = cfg.global_direction
        self.state.recording = cfg.recording
        self.state.roi_count = len(self.points)
        self.state.invalidate_preflight()   # D-13후속: 프로파일 교체 = 배치 변경 → 재프리플라이트 강제
        self._refresh_next()
        self._render_roi()

    def _save_profile(self) -> None:
        if self._profile_store is None:
            messagebox.showinfo("프로파일", "미연결 — 통합 태스크에서 배선됩니다")
            return
        cfg = self._composed_config_or_warn()      # D-13: 기포 크기·ROI 프레임 이탈 가드
        if cfg is None:
            return
        name = self._profile_combo.get() or "새 프로파일"
        try:
            self._profile_store.save(name, cfg)
        except (KeyError, IOError) as e:
            messagebox.showwarning("프로파일 저장 실패", f"'{name}': {e}")
            return
        self._profile_combo["values"] = self._profile_names()

    def _compose_config(self):
        """현재 위저드 상태 → MultiwayConfig (프리플라이트/측정/프로파일 공용)."""
        from bubble_counter.multiway.model import MultiwayConfig
        return MultiwayConfig(
            recording=self.state.recording,
            points=tuple(self.points),
            bubble=self.bubble,
            global_direction=self.global_direction,
        )

    def _composed_config_or_warn(self):
        """_compose_config를 감싸 기포 크기 미지정·ROI 프레임 이탈을 친절 배너로 안내한다 (D-13).

        정상이면 MultiwayConfig, 문제면 messagebox 배너 표시 후 None(호출부는 None이면 중단).
        프리플라이트 실행·프로파일 저장·측정 시작의 공통 진입 가드.
        """
        if self.bubble is None:
            messagebox.showwarning(
                "측정 준비", "3단계에서 '기포 크기 재기'로 크기를 먼저 지정하세요")
            return None
        try:
            return self._compose_config()
        except ValueError as exc:
            messagebox.showwarning(
                "측정 준비", f"{exc}\n3단계에서 해당 ROI를 프레임 안으로 이동하세요")
            return None

    # ---- 4단계: 프리플라이트(검사는 A/C, B는 표시만) ----
    def _build_preflight(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Button(parent, text="검사 실행", command=self._run_preflight).grid(
            row=0, column=0, sticky="w", pady=4)
        self._pf_text = tk.Text(parent, height=16, wrap="word")
        self._pf_text.grid(row=1, column=0, sticky="nsew")
        self._pf_text.tag_configure("block", foreground="#e34948")
        self._pf_text.tag_configure("warn", foreground="#c77f00")
        self._pf_text.tag_configure("info", foreground="#666")   # D-16: 상태 정보(문제 아님)
        self._pf_text.insert("end", "'검사 실행'을 누르세요.\n")
        self._pf_text.configure(state="disabled")

    def _run_preflight(self) -> None:
        if self._preflight_fn is None:
            messagebox.showinfo("프리플라이트", "미연결 — 통합 태스크에서 배선됩니다")
            return
        cfg = self._composed_config_or_warn()      # D-13: 기포 크기·ROI 프레임 이탈 가드
        if cfg is None:
            return
        items = self._preflight_fn(cfg)
        disp = PreflightDisplay.from_items(items)
        self._pf_text.configure(state="normal")
        self._pf_text.delete("1.0", "end")
        if disp.blocking:
            self._pf_text.insert("end", "■ 차단 — 해결해야 시작할 수 있습니다\n", "block")
            for m in disp.blocking:
                self._pf_text.insert("end", f"  · {m}\n", "block")
        if disp.warnings:
            self._pf_text.insert("end", "△ 경고 — 확인 후 진행 가능\n", "warn")
            for m in disp.warnings:
                self._pf_text.insert("end", f"  · {m}\n", "warn")
        if not disp.blocking and not disp.warnings:
            self._pf_text.insert("end", "문제 없음 ✓\n")
        if disp.info:                              # D-16: 흔들림 σ 등 상태 정보(문제 아님)
            for m in disp.info:
                self._pf_text.insert("end", f"ℹ {m}\n", "info")
        self._pf_text.configure(state="disabled")
        self.state.preflight_passed = not disp.has_blocking
        self._refresh_next()

    # ---- 5단계: 측정 진행(기존 gui.py 스레딩 패턴 재사용) ----
    def _build_run(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        self._run_status = tk.StringVar(value="대기 중")
        self._run_res = tk.StringVar(value="")
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nw")
        ttk.Label(left, textvariable=self._run_status).grid(row=0, column=0, sticky="w", pady=4)
        self._run_bar = ttk.Progressbar(left, mode="determinate", length=420)
        self._run_bar.grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(left, textvariable=self._run_res, foreground="#666").grid(
            row=2, column=0, sticky="w")
        btns = ttk.Frame(left)
        btns.grid(row=3, column=0, sticky="w", pady=6)
        self._start_btn = ttk.Button(btns, text="측정 시작", command=self._start_run)
        self._start_btn.pack(side="left")
        self._cancel_btn = ttk.Button(btns, text="취소", command=self._cancel_run, state="disabled")
        self._cancel_btn.pack(side="left", padx=4)
        # 라이브 뷰 (관찰 전용 L4)
        self._live_control = LiveControl(1.0)
        self._live_queue: queue.Queue = queue.Queue(maxsize=2)
        try:
            from bubble_counter.gui_live import LiveViewPanel, draw_multiway_overlay

            def _overlay(bgr, snap):
                cfg = getattr(self, "_last_run_cfg", None) or self._composed_config_or_warn()
                if cfg is not None:
                    draw_multiway_overlay(bgr, snap, cfg.points)

            self._live_panel = LiveViewPanel(
                parent, self._live_control, overlay_draw=_overlay)
            self._live_panel.grid(row=0, column=1, sticky="nsew", padx=8)
        except Exception:
            self._live_panel = None

    def _start_run(self) -> None:
        if self._run_active:
            # rev-B14 Critical: 시작 버튼 재클릭으로 두 번째 워커가 같은 out_root에
            # 동시 쓰기하는 것을 방지 — 실행 중이면 조용히 무시.
            return
        if self._run_fn is None:
            messagebox.showinfo("측정", "미연결 — 통합 태스크에서 배선됩니다")
            return
        cfg = self._composed_config_or_warn()      # D-13: 시작 전 기포 크기·ROI 프레임 이탈 가드
        if cfg is None:
            return
        self._run_active = True
        self._cancel_event = threading.Event()
        self._start_btn.state(["disabled"])
        self._cancel_btn.state(["!disabled"])
        self._last_run_cfg = cfg
        if self._live_panel is not None:
            self._live_panel.reset()
            self._live_control.set_speed(1.0)
            self._live_control.set_display_paused(False)
        # 이전 라이브 큐 비우기
        try:
            while True:
                self._live_queue.get_nowait()
        except queue.Empty:
            pass

        def progress_cb(ev):
            self._queue.put(ev)

        def live_cb(snap):
            queue_put_drop(self._live_queue, snap)

        def worker():
            try:
                # M-1 계약: run_fn(..., live_cb, live_control) trailing optional.
                result = self._run_fn(
                    cfg, list(self.chapters), self.phase_name, progress_cb,
                    self._cancel_event, live_cb=live_cb, live_control=self._live_control)
                # D-13후속(Important): cfg는 클로저로 큐까지만 전달 — 대입은 소비 쪽에서.
                self._queue.put(("__done__", result, cfg))
            except TypeError:
                # 구형 run_fn 시그니처 폴백
                try:
                    result = self._run_fn(cfg, list(self.chapters),
                                          self.phase_name, progress_cb, self._cancel_event)
                    self._queue.put(("__done__", result, cfg))
                except Exception as exc:  # noqa: BLE001
                    self._queue.put(("__error__", str(exc)))
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("__error__", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_run)

    def _cancel_run(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def _poll_run(self) -> None:
        try:
            while True:
                snap = self._live_queue.get_nowait()
                if self._live_panel is not None:
                    self._live_panel.apply_snapshot(snap)
        except queue.Empty:
            pass
        try:
            while True:
                ev = self._queue.get_nowait()
                if isinstance(ev, tuple) and ev and ev[0] == "__done__":
                    self.results = ev[1]
                    # D-13후속(Important): 성공한 실행의 cfg만 보관 — 실패/취소된 재실행이
                    # 직전 성공 결과의 보정 cfg를 오염하지 않도록 큐 소비(메인스레드)의
                    # 성공분기에서만 대입한다(워커 스레드에서 직접 대입하지 않음).
                    self._last_run_cfg = ev[2]
                    self.state.run_complete = True
                    self._run_status.set("완료")
                    self._cancel_btn.state(["disabled"])
                    self._run_active = False  # rev-B14: 완료 후 이전 이동·재시작 허용
                    self._refresh_next()
                    return
                if isinstance(ev, tuple) and ev and ev[0] == "__error__":
                    self._run_status.set("오류")
                    messagebox.showerror("측정 오류", ev[1])
                    self._start_btn.state(["!disabled"])
                    self._cancel_btn.state(["disabled"])
                    self._run_active = False  # rev-B14: 오류 후 이전 이동·재시작 허용
                    return
                self._apply_progress(ev)
        except queue.Empty:
            pass
        self.after(100, self._poll_run)

    def _apply_progress(self, ev) -> None:
        frac = getattr(ev, "fraction", None)
        if frac is not None:
            self._run_bar["value"] = frac * 100
        msg = getattr(ev, "message", None)
        if msg:
            self._run_status.set(msg)
        res = getattr(ev, "resource", None)
        if res:
            self._run_res.set(res)

    def _build_results(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        if self.results is None:
            ttk.Label(parent, text="측정 결과가 없습니다(5단계 미완료).").grid(row=0, column=0)
            return
        nb = ttk.Notebook(parent)
        nb.grid(row=0, column=0, sticky="nsew")
        for ch in self.results.chapters:
            nb.add(self._build_chapter_view(nb, ch), text=self._chapter_label(ch))
        nb.add(self._build_phase_view(nb, self.results), text="페이즈 합산")

        # 검토 센터 열기 버튼 (완료된 결과가 있을 때만)
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btn_frame, text="검토 열기", command=self._open_review).pack(side="right")

    def _open_review(self) -> None:
        """검토 센터 열기 (gui_review.open_review_center 호출)."""
        if self.results is None:
            messagebox.showinfo("검토", "측정 결과가 없습니다(5단계 미완료).")
            return
        try:
            from bubble_counter.multiway.gui_review import open_review_center
        except Exception as exc:
            messagebox.showerror("검토 열기 실패", str(exc))
            return
        open_review_center(self, self.folder)

    def _chapter_label(self, ch) -> str:
        import os
        return os.path.basename(ch.video_path)

    def _targets_for(self):
        return {p.number: p.target for p in self.points}

    def _build_chapter_view(self, master, ch) -> ttk.Frame:
        frm = ttk.Frame(master, padding=6)
        frm.columnconfigure(0, weight=1)
        targets = self._targets_for()
        counts = ch.counts_final()
        errors = ch.errors(targets)

        # 자기진단 배너(L-1): 엔진 산출 메시지 + 편차>5% + 의심 잘림 경고를 모두 표시.
        # self_check_messages/suspects_truncated는 엔진이 ChapterResult에 덧붙이는 필드 —
        # A0 스텁엔 없으므로 getattr로 방어(통합 후 실제 값). '조용한 잘림 금지'(설계 §3 결정6).
        banner_frame = ttk.Frame(frm)
        banner_frame.grid(row=0, column=0, sticky="ew")
        banners = list(getattr(ch, "self_check_messages", []))
        vals = [counts.get(n, 0) for n in sorted(counts)]
        if vals and max(vals) > 0 and (max(vals) - min(vals)) / max(vals) > 0.05:
            worst = min(counts, key=lambda n: counts[n])
            banners.append(f"Point {worst}가 다른 Point 대비 낮습니다 — ROI 위치 확인 권장")
        if getattr(ch, "suspects_truncated", False):
            banners.append("의심 구간이 많습니다 — ROI 배치 재검토")
        stab = getattr(ch, "stabilization", None)          # D-16: 흔들림 큰 경우만 배너 표시
        if stab is not None and getattr(stab, "max_px", 0.0) > SHAKE_BANNER_PX:
            banners.append(f"흔들림 큼 — 보정 개입 {stab.corrected_frames}프레임")
        for msg in banners:
            ttk.Label(banner_frame, foreground="#c77f00",
                      text=f"⚠ 자기진단: {msg}").pack(anchor="w", pady=(0, 2))

        # Point별 표
        cols = ("point", "auto", "corr", "final", "target", "error")
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=7)
        for c, t in zip(cols, ("Point", "자동", "보정", "최종", "목표", "목표오차")):
            tree.heading(c, text=t)
            tree.column(c, width=70, anchor="center")
        for n in sorted(counts):
            corr = counts[n] - ch.counts_auto.get(n, 0)
            tree.insert("", "end", values=(n, ch.counts_auto.get(n, 0), f"{corr:+d}",
                                           counts[n], targets.get(n, 0), value_label(errors.get(n, 0))))
        tree.grid(row=1, column=0, sticky="ew", pady=4)

        # 목표 오차 그래프 (파일 count_error.png와 동일 의미 — 최종−target)
        ttk.Label(frm, text="목표 오차 (최종 − 목표)").grid(row=2, column=0, sticky="w")
        gcanvas = tk.Canvas(frm, width=560, height=220, background="#ffffff", highlightthickness=1)
        gcanvas.grid(row=3, column=0, pady=4)
        self._draw_error_graph(gcanvas, errors, theme="light")

        # count error = 순서 불일치 미계수 (D-21). 더블클릭 → ±10 프레임 뷰
        n_ce = len(ch.violations)
        ttk.Label(frm, text=f"count error (순서 미계수) — {n_ce}건").grid(
            row=4, column=0, sticky="w", pady=(6, 0))
        vtree = ttk.Treeview(frm, columns=("frame", "exp", "act"), show="headings", height=4)
        for c, t in zip(("frame", "exp", "act"), ("프레임", "기대", "실제")):
            vtree.heading(c, text=t)
            vtree.column(c, width=90, anchor="center")
        for v in ch.violations:
            vtree.insert("", "end", values=(v.frame, v.expected_point, v.actual_point))
        vtree.grid(row=5, column=0, sticky="ew")
        vtree.bind("<Double-1>", lambda e, c=ch: self._open_violation(vtree, c))

        # 의심 구간(선택 → 스냅샷/보정) — D-15: 확신(reflux 등급) 열 + 높음 우선 정렬
        ttk.Label(frm, text="의심 구간").grid(row=6, column=0, sticky="w", pady=(6, 0))
        stree = ttk.Treeview(frm, columns=("kind", "point", "frame", "detail", "confidence"),
                             show="headings", height=4)
        for c, t, wdt in zip(("kind", "point", "frame", "detail", "confidence"),
                             ("종류", "Point", "프레임", "사유", "확신"), (80, 60, 80, 260, 60)):
            stree.heading(c, text=t)
            stree.column(c, width=wdt, anchor="w")
        stree.tag_configure("high-confidence", foreground="#c77f00")  # 기존 자기진단 경고색 재사용(D-15)
        conf_order = {"높음": 0, "낮음": 1, "": 2}
        for s in sorted(ch.suspects, key=lambda s: conf_order.get(s.confidence, 2)):
            tags = ("high-confidence",) if s.confidence == "높음" else ()
            stree.insert("", "end", values=(s.kind, s.point, s.frame, s.detail, s.confidence),
                        tags=tags)
        stree.grid(row=7, column=0, sticky="ew")
        bfrm = ttk.Frame(frm)
        bfrm.grid(row=8, column=0, sticky="w", pady=4)
        ttk.Button(bfrm, text="보정 +1",
                   command=lambda c=ch, s=stree: self._correct(c, s, +1)).pack(side="left")
        ttk.Button(bfrm, text="보정 -1",
                   command=lambda c=ch, s=stree: self._correct(c, s, -1)).pack(side="left", padx=4)
        return frm

    def _draw_error_graph(self, canvas, errors, theme) -> None:
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        lay = error_bar_geometry(errors, w, h, bottom_margin=40, top_margin=30)
        canvas.create_line(16, lay.baseline_y, w - 16, lay.baseline_y, fill="#999")
        for b in lay.bars:
            canvas.create_rectangle(b.x0, b.y_top, b.x1, b.y_bottom,
                                    fill=bar_color(b.value, theme), outline="")
            # 값 라벨(−194)과 축 라벨(P1) 분리 — 큰 |e|에서도 겹침 방지
            canvas.create_text(b.label_xy[0], b.label_xy[1], text=b.label,
                               anchor="s" if b.value >= 0 else "n",
                               font=("", 9))
            canvas.create_text((b.x0 + b.x1) / 2, h - 6, text=f"P{b.number}",
                               anchor="s", font=("", 9), fill="#444")

    def _open_violation(self, tree, ch) -> None:
        sel = tree.selection()
        if not sel:
            return
        frame = int(tree.item(sel[0], "values")[0])
        if self._open_frame_fn is None:
            messagebox.showinfo("프레임 보기",
                                f"프레임 {frame} 부근(±10) — 뷰어 미연결(통합 태스크에서 배선)")
            return
        self._open_frame_fn(ch.video_path, frame)

    def _correct(self, ch, stree, delta) -> None:
        sel = stree.selection()
        if not sel:
            return
        _, point, frame, _, _ = stree.item(sel[0], "values")  # D-15: 확신 열 추가로 5칸(마지막 무시)
        reason = simpledialog.askstring("보정 사유", "사유를 입력하세요:", parent=self) or "수동 보정"
        if self._recount_fn is None:
            messagebox.showinfo("보정", "재계산 미연결 — 통합 태스크에서 배선됩니다")
            return
        if self._last_run_cfg is None:          # D-13후속: 실행 이력 없이 보정 진입 방어
            messagebox.showwarning("보정", "실행 이력이 없습니다 — 먼저 실행하세요")
            return
        from bubble_counter.multiway.model import Correction
        import datetime
        corr = Correction(int(point), int(frame), delta, reason or "수동 보정",
                          datetime.datetime.now().isoformat())
        try:
            updated = self._recount_fn(ch, corr)      # 자동값 원본 보존, 이벤트 재실행
        except ValueError as exc:               # D-13후속 심층방어(review-D13-verdict.md Critical-1)
            messagebox.showwarning(
                "보정", f"{exc}\n3단계에서 해당 ROI를 프레임 안으로 이동하세요")
            return
        self._replace_chapter(updated)
        self._show_step()                              # 표·그래프·set·위반 즉시 재계산

    def _replace_chapter(self, updated) -> None:
        self.results.chapters = [updated if c.video_path == updated.video_path else c
                                 for c in self.results.chapters]

    def _build_phase_view(self, master, phase) -> ttk.Frame:
        frm = ttk.Frame(master, padding=6)
        frm.columnconfigure(0, weight=1)
        n_ch = len(phase.chapters)
        targets = {n: t * n_ch for n, t in self._targets_for().items()}   # 챕터 목표×챕터수
        totals = phase.sum_counts_final()
        errors = {n: totals.get(n, 0) - targets.get(n, 0) for n in targets}
        ttk.Label(frm, text=f"페이즈 '{phase.name}' · 챕터 {n_ch}개 · 완성 set {phase.sum_sets()}"
                  ).grid(row=0, column=0, sticky="w", pady=4)
        cols = ("point", "final", "target", "error")
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=7)
        for c, t in zip(cols, ("Point", "합산 최종", "합산 목표", "오차")):
            tree.heading(c, text=t)
            tree.column(c, width=90, anchor="center")
        for n in sorted(targets):
            tree.insert("", "end", values=(n, totals.get(n, 0), targets[n],
                                           value_label(errors.get(n, 0))))
        tree.grid(row=1, column=0, sticky="ew", pady=4)
        gcanvas = tk.Canvas(frm, width=560, height=220, background="#ffffff", highlightthickness=1)
        gcanvas.grid(row=2, column=0, pady=4)
        self._draw_error_graph(gcanvas, errors, theme="light")
        return frm


class _RoiPopup(tk.Toplevel):
    """ROI 그리기/재입력 팝업: Point 번호·방향 예외·목표 개수."""

    def __init__(self, master, existing, preset=None):
        super().__init__(master)
        self.title("Point 설정")
        self.result = None
        default_num = preset[0] if preset else (max(existing, default=0) + 1)
        default_dir = preset[1] if preset else ""
        default_target = preset[2] if preset else 500
        self._existing = [n for n in existing if not (preset and n == preset[0])]

        self._num = tk.StringVar(value=str(default_num))
        self._dir = tk.StringVar(value=default_dir or "(전역 따름)")
        self._target = tk.StringVar(value=str(default_target))
        ttk.Label(self, text="Point 번호:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(self, textvariable=self._num, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(self, text="방향 예외:").grid(row=1, column=0, sticky="e", padx=4)
        ttk.Combobox(self, textvariable=self._dir, width=12, state="readonly",
                     values=["(전역 따름)", "+x", "-x", "+y", "-y"]).grid(row=1, column=1, padx=4)
        ttk.Label(self, text="목표 개수:").grid(row=2, column=0, sticky="e", padx=4)
        ttk.Entry(self, textvariable=self._target, width=8).grid(row=2, column=1, padx=4)
        ttk.Button(self, text="확인", command=self._ok).grid(row=3, column=0, columnspan=2, pady=6)
        self.transient(master)
        self.grab_set()
        self.wait_window(self)

    def _ok(self) -> None:
        try:
            number = parse_point_number(self._num.get(), self._existing)
            target = parse_target(self._target.get())
        except ValueError as e:
            messagebox.showwarning("입력", str(e), parent=self)
            return
        direction = None if self._dir.get().startswith("(") else self._dir.get()
        self.result = (number, direction, target)
        self.destroy()
