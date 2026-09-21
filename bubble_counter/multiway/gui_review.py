"""검토 센터 메인 GUI (P2, R0 §3.6).

- 탭: 이벤트/배제됨/의심/이물/감사
- 데이터 소스: ChapterView (loader.py)
- 상태: ReviewState (gui_review_state.py)
- 보정: corrections.py
- 감사 탭: P34에서 attach_audit_tab으로 주입
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import numpy as np
import cv2

from .loader import ChapterView, load_chapter, find_video, is_review_folder
from .gui_review_state import ReviewState, Tab, FilterMode
from .corrections import make_correction
from bubble_counter.gui_view import ViewTransform, ppm_p6_bytes
from ..video_io import VideoSource


class ReviewCenter(tk.Toplevel):
    """검토 센터 메인 윈도우 (탭 5개)."""

    def __init__(self, master: tk.Widget, folder: Path):
        super().__init__(master)
        self.title(f"검토 센터 — {folder.name}")
        self.geometry("1400x900")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.folder = Path(folder)
        self._view: ChapterView | None = None
        self._state: "ReviewState" | None = None
        self._photo = None
        self._canvas_img_id = None
        self._canvas_scale = 1.0
        self._canvas_offset = (0, 0)

        # UI 구성
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        # 상단 툴바
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=4, pady=4)

        ttk.Button(toolbar, text="🔄 새로고침", command=self._reload).pack(side="left")
        ttk.Button(toolbar, text="📂 다른 폴더", command=self._open_another).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)

        self._tab_var = tk.StringVar()
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=4, pady=4)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 탭별 프레임 생성
        self._frames = {}
        for tab_name in ["이벤트", "count error", "배제됨", "의심", "이물", "감사"]:
            frame = ttk.Frame(self._notebook)
            self._notebook.add(frame, text=tab_name)
            self._frames[tab_name] = frame

        # 각 탭 내용 구성
        self._build_events_tab(self._frames["이벤트"])
        self._build_count_errors_tab(self._frames["count error"])
        self._build_filtered_tab(self._frames["배제됨"])
        self._build_suspects_tab(self._frames["의심"])
        self._build_foreign_tab(self._frames["이물"])
        self._build_audit_tab(self._frames["감사"])

        # 하단 상태바
        self._status_var = tk.StringVar(value="준비")
        ttk.Label(self, textvariable=self._status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom", padx=4, pady=4)

    def _load_data(self):
        try:
            view = load_chapter(self.folder)
            self._view = view
            # 상태 초기화
            from .gui_review_state import ReviewState
            self._state = ReviewState(
                view=view,
                config=view.config if view.config else None,
                folder=self.folder,
                reload_callback=self._reload,
                open_frame_callback=self._open_frame_viewer,
                make_clip_callback=self._make_clip,
            )
            self._populate_all()
            self._update_status(f"로드 완료: {self.folder.name}")
        except Exception as e:
            messagebox.showerror("로드 실패", str(e))
            self.destroy()

    def _populate_all(self):
        self._refresh_events_tab()
        self._refresh_count_errors_tab()
        self._refresh_filtered_tab()
        self._refresh_suspects_tab()
        self._refresh_foreign_tab()
        # 감사 탭은 P34에서 attach

    def _refresh_events_tab(self):
        """이벤트 탭 리스트 갱신."""
        if not self._state:
            return
        tree = self._events_tree
        tree.delete(*tree.get_children())
        fps = self._chapter_fps()
        for idx, e in self._state.events_filtered():
            flags = "|".join(e.flags) if e.flags else ""
            pos = f"{e.position_px:.1f}" if e.position_px >= 0 else "미기록"
            size = f"{e.size_px:.1f}" if e.size_px >= 0 else "미기록"
            tree.insert("", "end", iid=str(idx), values=(
                e.point, e.frame, f"{e.subframe:.4f}",
                f"{e.frame / fps:.3f}" if fps > 0 else "",
                flags, pos, size
            ))
        self._update_counts_label()

    def _chapter_fps(self) -> float:
        """RecordingSpec fps (구버전 폴더는 summary에 recording이 없어 0.0 = 시각 미표기)."""
        if self._view and self._view.recording:
            return self._view.recording.fps
        return 0.0

    def _refresh_count_errors_tab(self):
        """count error 탭 리스트 갱신 (D-21 순서 미계수 사건)."""
        if not self._state:
            return
        tree = self._ce_tree
        tree.delete(*tree.get_children())
        fps = self._chapter_fps()
        for i, v in enumerate(self._state.violations_sorted()):
            t = f"{v.frame / fps:.3f}" if fps > 0 else ""
            tree.insert("", "end", iid=str(i),
                        values=(v.frame, t, f"P{v.expected_point}", f"P{v.actual_point}"))
        n = len(self._view.violations) if self._view else 0
        self._ce_info.config(text=f"count error(순서 미계수) — {n}건")

    def _refresh_filtered_tab(self):
        if not self._state:
            return
        tree = self._filtered_tree
        tree.delete(*tree.get_children())
        for idx, f in self._state.filtered_filtered():
            point, frame, size_px, reason = f
            tree.insert("", "end", iid=str(idx), values=(point, frame, f"{size_px:.1f}", reason))

    def _refresh_suspects_tab(self):
        if not self._state:
            return
        tree = self._suspects_tree
        tree.delete(*tree.get_children())
        fps = self._chapter_fps()
        for idx, s in self._state.suspects_filtered():
            conf = s.confidence if s.confidence else ""
            tree.insert("", "end", iid=str(idx), values=(
                s.kind, s.point, s.frame, f"{s.frame / fps:.2f}" if fps > 0 else "",
                s.detail, s.snapshot or "", conf
            ))

    def _refresh_foreign_tab(self):
        if not self._state:
            return
        tree = self._foreign_tree
        tree.delete(*tree.get_children())
        if not self._view.foreign:
            return
        for idx, fo in self._state.foreign_filtered():
            tree.insert("", "end", iid=str(idx), values=(
                fo.get("id", ""), f"{fo.get('x_px',0):.1f}", f"{fo.get('y_px',0):.1f}",
                f"{fo.get('w_px',0):.1f}", f"{fo.get('h_px',0):.1f}", f"{fo.get('area_px',0):.1f}",
                ",".join(map(str, fo.get("near_points", []))), fo.get("source",""),
                fo.get("tag",""), fo.get("tagged_at",""), fo.get("note","")
            ))

    def _build_events_tab(self, parent):
        # 툴바
        tb = ttk.Frame(parent)
        tb.pack(fill="x", padx=4, pady=4)
        ttk.Label(tb, text="필터:").pack(side="left")
        self._event_filter_vars = {}
        for fm in FilterMode:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(tb, text=fm.value, variable=var,
                                 command=lambda m=fm, v=var: self._toggle_event_filter(m, v))
            cb.pack(side="left", padx=2)
            self._event_filter_vars[fm] = var
        ttk.Button(tb, text="전체 선택", command=lambda: self._set_event_filters_all(True)).pack(side="left", padx=4)
        ttk.Button(tb, text="전체 해제", command=lambda: self._set_event_filters_all(False)).pack(side="left")

        # 리스트
        cols = ("point", "frame", "subframe", "time_s", "flags", "position_px", "size_px")
        self._events_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="extended")
        for c, w in zip(cols, (60, 70, 70, 70, 200, 80, 70)):
            self._events_tree.heading(c, text=c, command=lambda k=c: self._state.sort(k))
            self._events_tree.column(c, width=w, anchor="center" if c != "flags" else "w")
        self._events_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._events_tree.bind("<Double-1>", self._on_event_double)
        self._events_tree.bind("<<TreeviewSelect>>", self._on_event_select)

        # 하단 카운트/보정
        bottom = ttk.Frame(parent)
        bottom.pack(fill="x", padx=4, pady=4)
        self._counts_label = ttk.Label(bottom, text="")
        self._counts_label.pack(side="left")
        ttk.Button(bottom, text="➕ 선택 +1", command=lambda: self._apply_correction(+1)).pack(side="right", padx=2)
        ttk.Button(bottom, text="➖ 선택 -1", command=lambda: self._apply_correction(-1)).pack(side="right", padx=2)

    def _build_count_errors_tab(self, parent):
        """count error 탭 — 목록 + 프레임 뷰어(에러 간 순회) + 온디맨드 슬로모 클립."""
        tb = ttk.Frame(parent)
        tb.pack(fill="x", padx=4, pady=4)
        self._ce_info = ttk.Label(tb, text="count error(순서 미계수) — 0건")
        self._ce_info.pack(side="left")
        ttk.Button(tb, text="🎬 슬로모 클립 생성·열기",
                   command=self._clip_selected_count_error).pack(side="right", padx=2)
        ttk.Button(tb, text="▶ 프레임 뷰어",
                   command=self._open_selected_count_error).pack(side="right", padx=2)

        cols = ("frame", "time_s", "expected", "actual")
        self._ce_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        for c, t, w in zip(cols, ("프레임", "시각(초)", "기대 Point", "실제 Point"),
                           (90, 90, 100, 100)):
            self._ce_tree.heading(c, text=t)
            self._ce_tree.column(c, width=w, anchor="center")
        self._ce_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._ce_tree.bind("<Double-1>", self._on_count_error_double)

        hint = ("count error = 순서 게이트(D-21)가 미계수 처리한 사건. 행의 프레임은 기각된 통과의 "
                "시각이며, 누락이 원인이면 그 직전에 '기대 Point'의 기포가 빠진 것입니다. "
                "더블클릭=프레임 뷰어, 뷰어의 ◀◀/▶▶ 버튼으로 에러 간 이동.")
        ttk.Label(parent, text=hint, foreground="gray", wraplength=1300,
                  justify="left").pack(fill="x", padx=4, pady=(0, 4))

    def _build_filtered_tab(self, parent):
        tb = ttk.Frame(parent)
        tb.pack(fill="x", padx=4, pady=4)
        ttk.Label(tb, text="필터:").pack(side="left")
        self._filtered_filter_vars = {}
        for fm in (FilterMode.FILTERED, FilterMode.WEAK, FilterMode.DUPLICATE,
                   FilterMode.RING_RESIDUAL, FilterMode.BOUNDARY_INCOMPLETE, FilterMode.UNDERSIZE):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(tb, text=fm.value, variable=var,
                                 command=lambda m=fm, v=var: self._toggle_filtered_filter(m, v))
            cb.pack(side="left", padx=2)
            self._filtered_filter_vars[fm] = var

        cols = ("point", "frame", "size_px", "reason")
        self._filtered_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="extended")
        for c, w in zip(cols, (60, 70, 80, 200)):
            self._filtered_tree.heading(c, text=c)
            self._filtered_tree.column(c, width=w, anchor="center" if c != "reason" else "w")
        self._filtered_tree.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_suspects_tab(self, parent):
        tb = ttk.Frame(parent)
        tb.pack(fill="x", padx=4, pady=4)
        ttk.Label(tb, text="종류 필터:").pack(side="left")
        self._suspect_kind_vars = {}
        kinds = ["merge", "slow", "gap-anomaly", "undersize"]
        for k in kinds:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(tb, text=k, variable=var,
                                 command=lambda k=k, v=var: self._toggle_suspect_filter(k, v))
            cb.pack(side="left", padx=2)
            self._suspect_kind_vars[k] = var

        cols = ("kind", "point", "frame", "time_s", "detail", "snapshot", "confidence")
        self._suspects_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="extended")
        for c, w in zip(cols, (80, 60, 70, 70, 300, 150, 80)):
            self._suspects_tree.heading(c, text=c)
            self._suspects_tree.column(c, width=w, anchor="center" if c not in ("detail", "snapshot") else "w")
        self._suspects_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._suspects_tree.bind("<Double-1>", self._on_suspect_double)

    def _build_foreign_tab(self, parent):
        tb = ttk.Frame(parent)
        tb.pack(fill="x", padx=4, pady=4)
        ttk.Label(tb, text="태그:").pack(side="left")
        self._foreign_tag_var = tk.StringVar()
        cb = ttk.Combobox(tb, textvariable=self._foreign_tag_var, width=10, state="readonly")
        cb["values"] = ("", "파티클", "기포", "모름")
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_foreign_tab())

        cols = ("id", "x_px", "y_px", "w_px", "h_px", "area_px", "near_points", "source", "tag", "tagged_at", "note")
        self._foreign_tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="extended")
        for c, w in zip(cols, (40, 70, 70, 60, 60, 70, 90, 80, 60, 120, 150)):
            self._foreign_tree.heading(c, text=c)
            self._foreign_tree.column(c, width=w, anchor="center" if c not in ("note",) else "w")
        self._foreign_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self._foreign_tree.bind("<Double-1>", self._on_foreign_double)

        # 태깅 버튼
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=4, pady=4)
        for tag in ("파티클", "기포", "모름"):
            ttk.Button(btn_frame, text=f"태그: {tag}",
                       command=lambda t=tag: self._tag_selected_foreign(t)).pack(side="left", padx=2)

    def _build_audit_tab(self, parent):
        """P34에서 attach_audit_tab으로 주입됨. 여기선 플레이스홀더."""
        ttk.Label(parent, text="감사 탭은 P34 구현 후 attach_audit_tab으로 주입됩니다.",
                  foreground="gray").pack(expand=True)

    # ── 필터 토글 ──
    def _toggle_event_filter(self, mode: FilterMode, var: tk.BooleanVar):
        if var.get():
            self._state.event_filters.add(mode)
        else:
            self._state.event_filters.discard(mode)
        self._refresh_events_tab()

    def _set_event_filters_all(self, on: bool):
        for fm, var in self._event_filter_vars.items():
            var.set(on)
        self._state.event_filters = set(FilterMode) if on else set()
        self._refresh_events_tab()

    def _toggle_filtered_filter(self, mode: FilterMode, var: tk.BooleanVar):
        if var.get():
            self._state.filtered_filters.add(mode)
        else:
            self._state.filtered_filters.discard(mode)
        self._refresh_filtered_tab()

    def _toggle_suspect_filter(self, kind: str, var: tk.BooleanVar):
        if var.get():
            self._state.suspect_filters.add(kind)
        else:
            self._state.suspect_filters.discard(kind)
        self._refresh_suspects_tab()

    # ── 이벤트 핸들러 ──
    def _on_event_double(self, event):
        sel = self._events_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        # 원본 인덱스 찾기
        filtered = self._state.events_filtered()
        if idx < len(filtered):
            orig_idx, e = filtered[idx]
            self._open_frame_viewer(e.frame, e.point)

    def _on_event_select(self, event):
        sel = self._events_tree.selection()
        self._state.selected_event_indices = {int(s) for s in sel}

    def _on_suspect_double(self, event):
        sel = self._suspects_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        filtered = self._state.suspects_filtered()
        if idx < len(filtered):
            orig_idx, s = filtered[idx]
            self._open_frame_viewer(s.frame, s.point)

    def _on_foreign_double(self, event):
        sel = self._foreign_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        filtered = self._state.foreign_filtered()
        if idx < len(filtered):
            orig_idx, fo = filtered[idx]
            self._open_frame_viewer(fo.get("frame", 0), fo.get("near_points", [0])[0] if fo.get("near_points") else None)

    # ── count error 탭 핸들러 ──
    def _selected_ce_index(self) -> int | None:
        sel = self._ce_tree.selection()
        return int(sel[0]) if sel else None

    def _on_count_error_double(self, event):
        self._open_selected_count_error()

    def _open_selected_count_error(self):
        """선택 count error를 프레임 뷰어로 — targets 전달로 에러 간 순회 가능."""
        idx = self._selected_ce_index()
        if idx is None:
            messagebox.showinfo("알림", "count error 탭에서 항목을 선택하세요.")
            return
        if not self._view or not self._view.video_path:
            messagebox.showwarning("영상 없음", "원본 영상 파일을 찾을 수 없습니다.")
            return
        targets = self._state.violation_targets()
        if idx >= len(targets):
            return
        frame, point, _label = targets[idx]
        viewer = _FrameViewer(self, self._view.video_path, frame, point,
                              targets=targets, target_index=idx)
        viewer.grab_set()

    def _clip_selected_count_error(self):
        idx = self._selected_ce_index()
        if idx is None:
            messagebox.showinfo("알림", "count error 탭에서 항목을 선택하세요.")
            return
        vs = self._state.violations_sorted()
        if idx >= len(vs):
            return
        v = vs[idx]
        self._make_clip(v.actual_point, v.frame)

    # ── 보정 적용 ──
    def _apply_correction(self, delta: int):
        if not self._state.selected_event_indices:
            messagebox.showinfo("알림", "이벤트 탭에서 항목을 선택하세요.")
            return
        # 선택된 이벤트들의 point/frame 수집
        for idx in self._state.selected_event_indices:
            filtered = self._state.events_filtered()
            if idx < len(filtered):
                orig_idx, e = filtered[idx]
                reason = "수동 +" if delta > 0 else "수동 -"
                self._state.stage_correction(e.point, e.frame, delta, reason)
        self._state.commit_pending()
        self._refresh_events_tab()
        self._update_counts_label()

    def _update_counts_label(self):
        if not self._state:
            return
        auto = self._view.counts_auto
        final = self._state.counts_preview()
        delta = {k: final[k] - auto.get(k, 0) for k in final}
        text = " | ".join(f"P{k}: 자동 {auto.get(k,0)} / Σδ {delta.get(k,0):+d} = {final[k]}" for k in sorted(final))
        self._counts_label.config(text=text)

    # ── 이물 태깅 ──
    def _tag_selected_foreign(self, tag: str):
        sel = self._foreign_tree.selection()
        if not sel:
            return
        for idx_str in sel:
            idx = int(idx_str)
            filtered = self._state.foreign_filtered()
            if idx < len(filtered):
                orig_idx, fo = filtered[idx]
                fo["tag"] = tag
                fo["tagged_at"] = datetime.utcnow().isoformat(timespec="seconds")
        self._refresh_foreign_tab()
        # foreign_objects.json 저장
        if self._view.foreign:
            from .report import write_foreign_objects_json
            write_foreign_objects_json(
                self.folder / "foreign_objects.json",
                self._view.foreign.get("objects", []),
                source="review-scan",
                scanned_at=datetime.utcnow().isoformat(timespec="seconds"),
                sample_frames=60,
                band_margin_px=25.0,
            )

    # ── 프레임 뷰어 / 클립 ──
    def _open_frame_viewer(self, frame: int, point: int | None):
        """프레임 뷰어 (±10프레임 네비)."""
        if not self._view or not self._view.video_path:
            messagebox.showwarning("영상 없음", "원본 영상 파일을 찾을 수 없습니다.")
            return
        viewer = _FrameViewer(self, self._view.video_path, frame, point)
        viewer.grab_set()

    def _make_clip(self, point: int, frame: int) -> Path | None:
        """±25프레임 슬로모션 클립 생성 후 열기 (RI 배선 ② — clipgen 연동, 오버레이 포함)."""
        if not self._view or not self._view.video_path:
            messagebox.showwarning("영상 없음", "원본 영상 파일을 찾을 수 없어 클립을 만들 수 없습니다.")
            return None
        from .clipgen import ClipOverlay, clip_path, render_clip
        out = clip_path(self.folder, point, frame)
        overlay = None
        cfg = self._view.config
        if cfg is not None:
            overlay = ClipOverlay(points=tuple(cfg.points),
                                  frame_w=cfg.recording.width,
                                  frame_h=cfg.recording.height,
                                  global_direction=cfg.global_direction,
                                  highlight_point=point, event_frame=frame)
        self._update_status(f"클립 생성 중… P{point} f{frame}")
        self.update_idletasks()
        try:
            render_clip(str(self._view.video_path), out, frame, overlay=overlay)
        except Exception as e:
            messagebox.showerror("클립 생성 실패", str(e))
            self._update_status("클립 생성 실패")
            return None
        self._update_status(f"클립 저장: {out}")
        try:
            os.startfile(str(out))  # Windows 기본 플레이어로 재생
        except (AttributeError, OSError):
            messagebox.showinfo("클립 저장됨", f"저장 위치:\n{out}")
        return out

    def _on_tab_changed(self, event):
        tab_name = self._notebook.tab(self._notebook.select(), "text")
        if tab_name == "이벤트":
            self._state.current_tab = Tab.EVENTS
        elif tab_name == "count error":
            self._state.current_tab = Tab.COUNT_ERRORS
        elif tab_name == "배제됨":
            self._state.current_tab = Tab.FILTERED
        elif tab_name == "의심":
            self._state.current_tab = Tab.SUSPECTS
        elif tab_name == "이물":
            self._state.current_tab = Tab.FOREIGN
        elif tab_name == "감사":
            self._state.current_tab = Tab.AUDIT

    def _reload(self):
        self._load_data()

    def _open_another(self):
        path = filedialog.askdirectory(title="검토 폴더 선택")
        if path:
            ReviewCenter(self.master, Path(path))

    def _on_close(self):
        self.destroy()

    def _update_status(self, msg: str):
        self._status_var.set(msg)


class _FrameViewer(tk.Toplevel):
    """±10 프레임 네비게이션 뷰어.

    targets가 주어지면(count error 순회 등) ◀◀/▶▶ 버튼으로 대상 간 점프를
    지원한다 — 각 대상은 (center_frame, point, 라벨). 기존 2~4인자 호출은 무변경.
    """

    def __init__(self, master, video_path: str, center_frame: int, point: int | None = None,
                 targets: "list[tuple[int, int | None, str]] | None" = None,
                 target_index: int = 0):
        super().__init__(master)
        self._video_name = os.path.basename(video_path)
        self._source = VideoSource(video_path)
        self._total = self._source.info.frame_count
        self._point = point
        self._targets = list(targets) if targets else []
        self._target_index = max(0, min(target_index, len(self._targets) - 1)) if self._targets else 0
        self._photo = None

        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=4, pady=4)
        ttk.Button(nav, text="◀", width=3, command=lambda: self._go(-1)).pack(side="left")
        self._pos_var = tk.StringVar()
        ttk.Label(nav, textvariable=self._pos_var).pack(side="left", padx=8)
        ttk.Button(nav, text="▶", width=3, command=lambda: self._go(+1)).pack(side="left")
        if self._targets:
            ttk.Separator(nav, orient="vertical").pack(side="left", fill="y", padx=6)
            ttk.Button(nav, text="◀◀ 이전 에러",
                       command=lambda: self._jump(-1)).pack(side="left")
            self._target_var = tk.StringVar()
            ttk.Label(nav, textvariable=self._target_var).pack(side="left", padx=8)
            ttk.Button(nav, text="다음 에러 ▶▶",
                       command=lambda: self._jump(+1)).pack(side="left")
        self._canvas = tk.Canvas(self, width=640, height=480, background="#202020", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_center(center_frame)

    def _set_center(self, center_frame: int):
        """중심 프레임 재설정 — ±10 창·타이틀·대상 라벨 갱신 후 렌더."""
        self._lo = max(0, center_frame - 10)
        self._hi = min(self._total - 1, center_frame + 10)
        self._frame = max(self._lo, min(self._hi, center_frame))
        self.title(f"프레임 뷰어 — {self._video_name} (Point {self._point})")
        if self._targets:
            self._target_var.set(self._targets[self._target_index][2])
        self._render()

    def _jump(self, delta: int):
        """대상(count error) 간 이동 — 창을 유지한 채 중심만 재설정."""
        if not self._targets:
            return
        self._target_index = max(0, min(len(self._targets) - 1, self._target_index + delta))
        frame, point, _label = self._targets[self._target_index]
        self._point = point
        self._set_center(frame)

    def _go(self, delta: int):
        self._frame = max(self._lo, min(self._hi, self._frame + delta))
        self._render()

    def _render(self):
        self._source.seek(self._frame)
        for _idx, _gray, bgr in self._source.frames(with_color=True):
            h, w = bgr.shape[:2]
            scale = min(640 / w, 480 / h)
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(bgr, (nw, nh))
            self._photo = ppm_p6_bytes(img)
            self._canvas.delete("all")
            self._canvas.create_image(320, 240, image=self._photo, anchor="center")
            self._pos_var.set(f"프레임 {self._frame} / {self._hi}")
            break

    def _on_close(self):
        self._source.close()
        self.destroy()


def open_review_center(master: tk.Widget, folder: Path) -> ReviewCenter:
    """외부 진입점 (gui.py / gui_multiway.py에서 호출)."""
    return ReviewCenter(master, folder)