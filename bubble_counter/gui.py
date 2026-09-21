"""Tkinter desktop front end for bubble-counter.

Single-window GUI wrapping ``bubble_counter.pipeline.run_count``. This module
imports tkinter at the top level (that is fine — ``bubble_counter.cli`` only
imports this module lazily, inside the ``gui`` subcommand handler, so
environments without tkinter such as this WSL dev machine are unaffected).

Imports of other ``bubble_counter`` modules (``config``, ``pipeline``) are
kept inside functions so this module stays importable on its own and mirrors
the lazy-import contract described in the design brief.

Threading contract (must not be violated):
    - ``run_count`` runs on a background ``threading.Thread``.
    - Its ``progress_cb`` is invoked from that same background thread; it
      only pushes ``ProgressEvent`` objects onto a ``queue.Queue`` — it never
      touches a tk widget.
    - All tk widget updates happen on the main thread, driven by
      ``root.after`` polling the queue.
    - Cancellation is requested by setting a ``threading.Event`` that is
      passed into ``run_count`` as ``cancel_event``; the pipeline itself
      decides when to stop and still writes outputs.
"""

import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

_AXIS_LABELS = {"수평선(상승 기포)": "h", "수직선": "v"}
_AXIS_LABELS_REV = {value: key for key, value in _AXIS_LABELS.items()}

_POLL_INTERVAL_MS = 100


class BubbleCounterApp:
    """Main application window: parameters -> run -> progress -> result."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("기포 계수 툴")

        self._queue: queue.Queue = queue.Queue()
        self._cancel_event: threading.Event | None = None
        self._worker: threading.Thread | None = None

        self._build_vars()
        self._build_widgets()

    # ------------------------------------------------------------------
    # Widget / variable construction
    # ------------------------------------------------------------------

    def _build_vars(self) -> None:
        # Derive widget defaults from the config dataclasses so a future
        # default change in config.py follows through to the GUI, then overlay
        # the last-used settings (설계 스펙 2026-07-04 — "시작" 시 저장분).
        # Lazy imports keep this module's top-level imports stdlib-only.
        from bubble_counter.config import CounterConfig, LineSpec
        from bubble_counter.gui_settings import load_settings

        cfg = CounterConfig()
        line = LineSpec()
        stored = load_settings()

        self.video_var = tk.StringVar()
        self.out_dir_var = tk.StringVar()
        self.axis_label_var = tk.StringVar(
            value=_AXIS_LABELS_REV[stored.get("axis", line.axis)]
        )
        self.line_pos_var = tk.DoubleVar(value=stored.get("line_pos", line.pos))
        # 밴드 자동 환산 상태 (설계 스펙 §4): band_frac(비율)이 진실이고 px는
        # 그 투영이다. 단, 사용자가 직접 고친 px가 저장된 비율보다 항상 우선.
        self.band_auto_var = tk.BooleanVar(value=stored.get("band_auto", True))
        self._band_frac = stored.get("band_frac")  # float | None
        self._band_manual_dirty = False  # probe 전 수동 수정 → 최초 probe에서 채택
        self._band_updating = False      # 프로그램적 band_var.set 재진입 가드
        self._probed_info = None         # video_io.VideoInfo | None
        self._probed_path = None         # 마지막으로 probe한 경로 (재probe 방지)
        self._last_video_dir = stored.get("last_video_dir", "")
        self.video_info_var = tk.StringVar(value="")
        self.band_var = tk.IntVar(value=stored.get("band_px", line.band_px))
        self.band_var.trace_add("write", self._on_band_edited)
        self.scale_var = tk.DoubleVar(value=stored.get("scale", cfg.scale))
        # Live numeric readouts for the two sliders (ttk.Scale has no built-in
        # value display, which proved confusing in real use).
        self.line_pos_label_var = tk.StringVar(value=f"{self.line_pos_var.get():.2f}")
        self.line_pos_var.trace_add(
            "write",
            lambda *_: self.line_pos_label_var.set(f"{self.line_pos_var.get():.2f}"),
        )
        self.scale_label_var = tk.StringVar(value=f"{self.scale_var.get():.2f}")
        self.scale_var.trace_add(
            "write",
            lambda *_: self.scale_label_var.set(f"{self.scale_var.get():.2f}"),
        )
        self.scale_var.trace_add("write", lambda *_: self._recompute_band())
        self.bucket_var = tk.DoubleVar(
            value=stored.get("bucket_seconds", cfg.bucket_seconds)
        )
        self.var_threshold_var = tk.DoubleVar(
            value=stored.get("var_threshold", cfg.var_threshold)
        )
        self.merge_gap_var = tk.IntVar(
            value=stored.get("merge_gap_frames", cfg.merge_gap_frames)
        )
        self.annotate_var = tk.BooleanVar(value=stored.get("annotate", cfg.annotate))
        self.save_rhythm_var = tk.BooleanVar(
            value=stored.get("save_rhythm", cfg.save_rhythm)
        )
        # fps 강제는 의도적으로 복원하지 않는다 — 영상마다 다른 값이라 이월되면
        # 시간축이 조용히 틀어질 수 있다 (설계 스펙 §2-2).
        self.fps_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="대기 중")
        # 실행 중 자원 사용률(CPU/RAM/GPU) 표시 — 진행 이벤트에서 채운다.
        self.resource_var = tk.StringVar(value="")

    def _build_widgets(self) -> None:
        pad = {"padx": 6, "pady": 4}
        # 멀티웨이 업그레이드: root에 노트북을 두고 기존 단일선 UI를 첫 탭으로,
        # 멀티웨이 위저드를 둘째 탭으로 붙인다. 아래 단일선 위젯들은 그대로 frm에
        # 구성되므로 단일선 탭 내용은 불변(진입점 1곳만 추가).
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frm = ttk.Frame(notebook, padding=10)
        notebook.add(frm, text="단일선")
        try:
            from bubble_counter.gui_multiway import attach_multiway_tab
            attach_multiway_tab(notebook)
        except Exception as exc:  # noqa: BLE001 — 멀티웨이 탭 실패가 단일선 앱을 죽이면 안 됨(D-9)
            import sys
            print(f"[멀티웨이] 탭 로드 실패 — 단일선 모드만 사용 가능: {exc}", file=sys.stderr)

        row = 0
        ttk.Label(frm, text="영상 파일:").grid(row=row, column=0, sticky="w", **pad)
        video_entry = ttk.Entry(frm, textvariable=self.video_var, width=50)
        video_entry.grid(row=row, column=1, sticky="ew", **pad)
        # 경로를 직접 타이핑해도 probe되도록 (찾아보기 전용이 아님 — 스펙 §2-3)
        video_entry.bind("<Return>", self._probe_current_video)
        video_entry.bind("<FocusOut>", self._probe_current_video)
        ttk.Button(frm, text="찾아보기", command=self._browse_video).grid(
            row=row, column=2, **pad
        )
        row += 1

        ttk.Label(frm, textvariable=self.video_info_var).grid(
            row=row, column=1, columnspan=2, sticky="w", **pad
        )
        row += 1

        ttk.Label(frm, text="출력 폴더:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.out_dir_var, width=50).grid(
            row=row, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="찾아보기", command=self._browse_out_dir).grid(
            row=row, column=2, **pad
        )
        row += 1

        ttk.Label(frm, text="계수선 방향:").grid(row=row, column=0, sticky="w", **pad)
        axis_combo = ttk.Combobox(
            frm,
            textvariable=self.axis_label_var,
            values=list(_AXIS_LABELS.keys()),
            state="readonly",
            width=20,
        )
        axis_combo.grid(row=row, column=1, sticky="w", **pad)
        axis_combo.bind("<<ComboboxSelected>>", lambda _e: self._recompute_band())
        row += 1

        ttk.Label(frm, text="계수선 위치 (0=위, 1=아래):").grid(
            row=row, column=0, sticky="w", **pad
        )
        ttk.Scale(
            frm, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.line_pos_var
        ).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.line_pos_label_var, width=5).grid(
            row=row, column=2, **pad
        )
        row += 1

        ttk.Label(frm, text="밴드 폭(px) — 빠른 기포일수록 크게:").grid(
            row=row, column=0, sticky="w", **pad
        )
        ttk.Spinbox(frm, from_=1, to=512, textvariable=self.band_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        ttk.Checkbutton(
            frm,
            text="해상도에 맞춰 자동",
            variable=self.band_auto_var,
            command=self._recompute_band,
        ).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="축소 배율 (0.1~1.0):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Scale(
            frm, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.scale_var
        ).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.scale_label_var, width=5).grid(
            row=row, column=2, **pad
        )
        row += 1

        ttk.Label(frm, text="시간 버킷(초):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(
            frm, from_=0.1, to=3600, increment=0.5, textvariable=self.bucket_var, width=10
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="var-threshold:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(
            frm, from_=1, to=500, textvariable=self.var_threshold_var, width=10
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="merge-gap(프레임):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(frm, from_=0, to=1000, textvariable=self.merge_gap_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        ttk.Checkbutton(
            frm, text="주석 영상 저장 (annotate)", variable=self.annotate_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Checkbutton(
            frm, text="rhythm 이미지 저장 (save-rhythm)", variable=self.save_rhythm_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="fps 강제 (빈칸=자동):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.fps_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=row, column=0, columnspan=3, sticky="ew", **pad)
        self.start_btn = ttk.Button(btn_frm, text="시작", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(
            btn_frm, text="중지", command=self._on_cancel, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        self.open_folder_btn = ttk.Button(
            btn_frm, text="폴더 열기", command=self._on_open_folder
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=4)
        self.open_review_btn = ttk.Button(
            btn_frm, text="기존 결과 열기", command=self._on_open_review
        )
        self.open_review_btn.pack(side=tk.LEFT, padx=4)
        row += 1

        self.progress = ttk.Progressbar(frm, mode="determinate", length=400)
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", **pad)
        row += 1

        # 진행바 바로 아래 자원 사용률 라벨 (CPU/RAM/GPU, 실행 중 갱신).
        ttk.Label(frm, textvariable=self.resource_var).grid(
            row=row, column=0, columnspan=3, sticky="w", **pad
        )
        row += 1

        ttk.Label(frm, textvariable=self.status_var).grid(
            row=row, column=0, columnspan=3, sticky="w", **pad
        )
        row += 1

        # 라이브 뷰 (관찰 전용 L4)
        from bubble_counter.live import LiveControl

        self._live_control = LiveControl(1.0)
        self._live_queue: queue.Queue = queue.Queue(maxsize=2)
        self._live_panel = None
        self._run_cfg = None
        try:
            from bubble_counter.gui_live import LiveViewPanel, draw_single_overlay

            def _overlay(bgr, snap):
                if self._run_cfg is not None:
                    draw_single_overlay(bgr, snap, self._run_cfg)

            self._live_panel = LiveViewPanel(frm, self._live_control, overlay_draw=_overlay)
            self._live_panel.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        except Exception:
            self._live_panel = None

        frm.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # 밴드 해상도 자동 환산 + 영상 probe (설계 스펙 §4)
    # ------------------------------------------------------------------

    def _frame_axis_len(self):
        """진행축 원본 길이: 수평선(h)은 영상 높이, 수직선(v)은 너비. probe 전엔 None."""
        if self._probed_info is None:
            return None
        if _AXIS_LABELS[self.axis_label_var.get()] == "h":
            return self._probed_info.height
        return self._probed_info.width

    def _on_band_edited(self, *_args) -> None:
        """band px 직접 수정 → 그 값이 새 비율 (사용자 의도가 저장된 비율에 우선)."""
        if self._band_updating:
            return
        if not self.band_auto_var.get():
            # 자동이 꺼진 동안의 수정도 사용자 의도로 기록 — 나중에 자동을
            # 켜거나 다른 트리거가 와도 이 px가 저장된 비율에 덮이지 않는다.
            self._band_manual_dirty = True
            return
        self._adopt_frac_from_px()

    def _adopt_frac_from_px(self) -> None:
        from bubble_counter.gui_settings import frac_for

        try:
            band_px = self.band_var.get()
        except tk.TclError:  # 입력 도중의 빈 칸/비숫자 — 확정 전이므로 무시
            return
        axis_len = self._frame_axis_len()
        if axis_len is None:
            self._band_manual_dirty = True
            return
        self._band_frac = frac_for(band_px, axis_len, self.scale_var.get())
        self._band_manual_dirty = False

    def _recompute_band(self) -> None:
        """비율→px 재계산 (영상 probe·배율·방향 변경 시). 수동 수정이 보류 중이면 채택부터."""
        if not self.band_auto_var.get():
            return
        axis_len = self._frame_axis_len()
        if axis_len is None:
            return
        if self._band_manual_dirty or self._band_frac is None:
            self._adopt_frac_from_px()
            return
        from bubble_counter.gui_settings import band_px_for

        px = band_px_for(self._band_frac, axis_len, self.scale_var.get())
        self._band_updating = True
        try:
            self.band_var.set(px)
        finally:
            self._band_updating = False

    def _probe_current_video(self, *_args) -> None:
        """영상 경로의 메타데이터를 읽어 정보 라벨·자동 환산에 반영 (같은 경로 재probe 안 함)."""
        path = self.video_var.get().strip()
        if not path or path == self._probed_path or not Path(path).is_file():
            return
        from bubble_counter.video_io import probe_video

        self._probed_path = path
        try:
            self._probed_info = probe_video(path)
        except (OSError, ValueError):  # FileNotFoundError/IOError 포함
            self._probed_info = None
            self._probed_path = None  # 실패한 경로는 캐시하지 않는다 (잠긴 파일 재시도 허용)
            self.video_info_var.set("영상 정보 읽기 실패")
            return
        info = self._probed_info
        if math.isfinite(info.fps) and info.fps > 0:
            fps_text = f"{info.fps:.1f}fps"
        else:
            fps_text = "fps 불명 (fps 강제 칸에 직접 입력)"
        self.video_info_var.set(f"영상 정보: {info.width}×{info.height} · {fps_text}")
        self._recompute_band()

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="영상 파일 선택",
            initialdir=self._last_video_dir or None,
            filetypes=[
                ("영상 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return
        self.video_var.set(path)
        self._last_video_dir = str(Path(path).parent)
        if not self.out_dir_var.get().strip():
            self.out_dir_var.set(self._default_out_dir(path))
        self._probe_current_video()

    def _browse_out_dir(self) -> None:
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if not path:
            return
        self.out_dir_var.set(path)

    @staticmethod
    def _default_out_dir(video_path: str) -> str:
        p = Path(video_path)
        return str(p.with_name(p.stem + "_기포계수"))

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    def _build_config(self):
        """Build a CounterConfig from the current widget values.

        Imported lazily: pipeline/config may not be importable yet in some
        dev environments, and this keeps gui.py's module-level imports
        limited to the standard library.
        """
        from bubble_counter.config import CounterConfig, LineSpec, RoiSpec
        from bubble_counter.gui_params import validate_params

        fps_override = validate_params(
            band_px=self.band_var.get(),
            bucket_seconds=self.bucket_var.get(),
            var_threshold=self.var_threshold_var.get(),
            merge_gap_frames=self.merge_gap_var.get(),
            fps_text=self.fps_var.get(),
        )

        return CounterConfig(
            roi=RoiSpec(),
            line=LineSpec(
                axis=_AXIS_LABELS[self.axis_label_var.get()],
                pos=self.line_pos_var.get(),
                band_px=self.band_var.get(),
            ),
            scale=self.scale_var.get(),
            var_threshold=self.var_threshold_var.get(),
            merge_gap_frames=self.merge_gap_var.get(),
            bucket_seconds=self.bucket_var.get(),
            fps_override=fps_override,
            annotate=self.annotate_var.get(),
            save_rhythm=self.save_rhythm_var.get(),
        )

    def _save_settings(self, cfg) -> None:
        """시작이 확정된 설정을 저장 (설계 스펙 §2-5). 저장 실패는 계수를 막지 않는다."""
        from bubble_counter.gui_settings import frac_for, save_settings

        axis_len = self._frame_axis_len()
        if axis_len is not None:
            # 실제 돌리는 (px, 처리길이)로 비율 최종 채택; probe가 없었으면 기존 유지
            self._band_frac = frac_for(cfg.line.band_px, axis_len, cfg.scale)
        save_settings({
            "axis": cfg.line.axis,
            "line_pos": cfg.line.pos,
            "band_px": cfg.line.band_px,
            "band_frac": self._band_frac,
            "band_auto": self.band_auto_var.get(),
            "scale": cfg.scale,
            "bucket_seconds": cfg.bucket_seconds,
            "var_threshold": cfg.var_threshold,
            "merge_gap_frames": cfg.merge_gap_frames,
            "annotate": cfg.annotate,
            "save_rhythm": cfg.save_rhythm,
            "last_video_dir": self._last_video_dir,
        })

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        video_path = self.video_var.get().strip()
        if not video_path:
            messagebox.showerror("오류", "영상 파일을 선택하세요.")
            return
        if not Path(video_path).is_file():
            messagebox.showerror("오류", f"영상 파일을 찾을 수 없습니다: {video_path}")
            return

        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            out_dir = self._default_out_dir(video_path)
            self.out_dir_var.set(out_dir)

        try:
            cfg = self._build_config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("오류", f"입력값을 확인하세요: {exc}")
            return

        self._save_settings(cfg)

        self._queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._run_cfg = cfg
        if self._live_panel is not None:
            self._live_panel.reset()
            self._live_control.set_speed(1.0)
            self._live_control.set_display_paused(False)
        try:
            while True:
                self._live_queue.get_nowait()
        except queue.Empty:
            pass

        self.progress.stop()
        self.progress["mode"] = "indeterminate"
        self.progress.start(50)
        self.status_var.set("처리 중...")
        self.start_btn["state"] = tk.DISABLED
        self.cancel_btn["state"] = tk.NORMAL

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(video_path, out_dir, cfg, self._cancel_event, self._queue,
                  self._live_queue, self._live_control),
            daemon=True,
        )
        self._worker.start()
        self.root.after(_POLL_INTERVAL_MS, self._poll_queue)

    @staticmethod
    def _run_worker(video_path, out_dir, cfg, cancel_event, q: queue.Queue,
                    live_q=None, live_control=None) -> None:
        """Runs on a background thread. Must never touch a tk widget."""
        from bubble_counter.live import queue_put_drop
        from bubble_counter.pipeline import run_count

        def progress_cb(event) -> None:
            q.put(("progress", event))

        def live_cb(snap) -> None:
            if live_q is not None:
                queue_put_drop(live_q, snap)

        try:
            result = run_count(
                video_path,
                out_dir,
                cfg,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                resmon_interval=0.5,
                live_cb=live_cb if live_q is not None else None,
                live_control=live_control,
                live_label=Path(video_path).name,
            )
        except Exception as exc:  # surfaced to the user via messagebox
            q.put(("error", str(exc)))
        else:
            q.put(("done", result))

    def _on_cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.status_var.set("중지 요청됨...")
        self.cancel_btn["state"] = tk.DISABLED

    def _poll_queue(self) -> None:
        """Main-thread-only: drains the queue and updates widgets."""
        try:
            while True:
                snap = self._live_queue.get_nowait()
                if self._live_panel is not None:
                    self._live_panel.apply_snapshot(snap)
        except queue.Empty:
            pass
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    self._update_progress(payload)
                elif kind == "done":
                    self._on_done(payload)
                    return
                elif kind == "error":
                    self._on_error(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(_POLL_INTERVAL_MS, self._poll_queue)

    def _update_progress(self, event) -> None:
        if event.total_frames > 0:
            if str(self.progress["mode"]) != "determinate":
                self.progress.stop()
                self.progress["mode"] = "determinate"
                self.progress["maximum"] = event.total_frames
            self.progress["value"] = event.frame_idx
            total_text = str(event.total_frames)
        else:
            total_text = "?"
        eta_text = f"{event.eta_s:.0f}초" if event.eta_s is not None else "알 수 없음"
        self.status_var.set(
            f"프레임 {event.frame_idx}/{total_text} | "
            f"처리 속도 {event.processing_fps:.1f} fps | "
            f"카운트 {event.count} | ETA {eta_text}"
        )
        # 자원 사용률은 ProgressEvent에 실려 온다(측정 비활성/실패 시 필드가 None).
        # 값이 있을 때만 라벨을 갱신한다. os는 모듈 상단 import를 그대로 쓴다.
        if (
            getattr(event, "cpu_pct", None) is not None
            or getattr(event, "rss_mb", None) is not None
        ):
            from bubble_counter.resmon import Snapshot, format_line

            snap = Snapshot(
                event.cpu_pct,
                event.cpu_cores,
                event.rss_mb,
                event.sys_ram_pct,
                event.gpu_pct,
            )
            self.resource_var.set(format_line(snap, os.cpu_count() or 1))

    def _on_done(self, result) -> None:
        self.progress.stop()
        self.progress["mode"] = "determinate"
        self.progress["maximum"] = 100
        self.progress["value"] = 100
        self.start_btn["state"] = tk.NORMAL
        self.cancel_btn["state"] = tk.DISABLED
        self.resource_var.set("")  # 완료 시 마지막 자원 표시 정리

        cancelled = bool(result.summary.get("cancelled"))
        if cancelled:
            self.status_var.set(f"중지됨 | 카운트 {result.total}")
        else:
            self.status_var.set(f"완료 | 총 카운트 {result.total}")

        prefix = "중지되었습니다. " if cancelled else ""
        messagebox.showinfo(
            "완료",
            f"{prefix}총 {result.total}개의 기포를 세었습니다.\n출력 폴더: {result.out_dir}",
        )

    def _on_error(self, message: str) -> None:
        self.progress.stop()
        self.progress["mode"] = "determinate"
        self.progress["value"] = 0
        self.start_btn["state"] = tk.NORMAL
        self.cancel_btn["state"] = tk.DISABLED
        self.resource_var.set("")  # 오류 시 마지막 자원 표시 정리
        self.status_var.set("오류 발생")
        messagebox.showerror("오류", f"기포 계수 중 오류가 발생했습니다:\n{message}")

    def _on_open_folder(self) -> None:
        path = self.out_dir_var.get().strip()
        if not path:
            messagebox.showerror("오류", "출력 폴더가 지정되지 않았습니다.")
            return
        p = Path(path)
        if not p.is_dir():
            messagebox.showerror("오류", f"폴더가 존재하지 않습니다: {p}")
            return
        if hasattr(os, "startfile"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _on_open_review(self) -> None:
        """기존 결과 폴더를 선택해 검토 센터 열기."""
        folder = filedialog.askdirectory(title="검토할 결과 폴더 선택")
        if not folder:
            return
        try:
            from bubble_counter.multiway.gui_review import open_review_center
        except Exception as exc:
            messagebox.showerror("검토 열기 실패", str(exc))
            return
        open_review_center(self, Path(folder))

    def _on_open_folder(self) -> None:
        path = self.out_dir_var.get().strip()
        if not path:
            messagebox.showerror("오류", "출력 폴더가 지정되지 않았습니다.")
            return
        p = Path(path)
        if not p.is_dir():
            messagebox.showerror("오류", f"폴더가 존재하지 않습니다: {p}")
            return
        if hasattr(os, "startfile"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)])


def main() -> int:
    """Entry point used by ``bubble_counter.cli``'s ``gui`` subcommand."""
    root = tk.Tk()
    BubbleCounterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    from bubble_counter.bootstrap import run_gui

    sys.exit(run_gui())
