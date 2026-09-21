"""감사 탭 GUI (P34, tkinter-free 로직 분리 — audit.py는 순수 함수)."""

from __future__ import annotations

import datetime
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Optional

from .audit import AuditSample, append_round, load_audit, estimates, select_samples, snapshot_name
from .loader import ChapterView
from bubble_counter.gui_view import ppm_p6_bytes


@dataclass
class AuditState:
    """감사 탭 상태."""
    view: ChapterView
    folder: Path
    reload: Callable[[], None]
    open_frame: Callable[[int, int | None], None]
    make_clip: Callable[[int, int], Path] | None

    round_no: int = 1
    samples: list[AuditSample] = None
    verdicts: dict[int, str] = None  # sample_idx -> "O"/"X"/"모름"

    def __post_init__(self):
        if self.samples is None:
            self.samples = []
        if self.verdicts is None:
            self.verdicts = {}

    def current_estimates(self) -> dict:
        """현재까지의 estimates (라운드 1..현재)."""
        from .audit import estimates, load_audit
        audit = load_audit(self.folder)
        if audit:
            return estimates(audit.get("rounds", []))
        return {}


def attach_audit_tab(notebook: ttk.Notebook, ctx) -> None:
    """검토 센터 노트북에 감사 탭 부착 (P34 구현 시 호출)."""
    from .audit import AuditSample

    tab = ttk.Frame(notebook)
    notebook.add(tab, text="감사")

    # 상태 주입 (ReviewCenter에서 ctx로 넘겨줌)
    state = ctx["state"]  # ReviewState
    folder = ctx["folder"]
    audit_state = AuditState(
        view=state.view,
        folder=folder,
        reload=state.reload_callback,
        open_frame=state.open_frame_callback,
        make_clip=state.make_clip_callback,
    )

    # ── 상단: 라운드 제어 ──
    top = ttk.Frame(tab)
    top.pack(fill="x", padx=4, pady=4)
    ttk.Label(top, text="라운드:").pack(side="left")
    audit_state.round_var = tk.IntVar(value=1)
    ttk.Spinbox(top, from_=1, to=10, textvariable=audit_state.round_var, width=5).pack(side="left", padx=4)
    ttk.Button(top, text="표본 뽑기", command=lambda: _draw_samples(audit_state, tab)).pack(side="left", padx=4)
    ttk.Button(top, text="라운드 저장", command=lambda: _save_round(audit_state, tab)).pack(side="left", padx=4)

    # 추정치 표시
    audit_state.est_var = tk.StringVar(value="추정치: 대기")
    ttk.Label(top, textvariable=audit_state.est_var, foreground="blue").pack(side="right", padx=8)

    # ── 리스트 ──
    cols = ("idx", "kind", "point", "frame", "time_s", "verdict")
    tree = ttk.Treeview(tab, columns=cols, show="headings", selectmode="browse")
    for c, w, anchor in zip(("idx", "kind", "point", "frame", "time_s", "verdict"),
                            (40, 60, 60, 70, 70, 70),
                            ("center", "center", "center", "center", "center", "center")):
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor=anchor)
    tree.pack(fill="both", expand=True, padx=4, pady=4)
    audit_state.tree = tree

    # 버튼 바 (O/X/모름)
    btns = ttk.Frame(tab)
    btns.pack(fill="x", padx=4, pady=4)
    for label, val in [("✅ O", "O"), ("❌ X", "X"), ("❓ 모름", "모름")]:
        ttk.Button(btns, text=label,
                   command=lambda v=val: _set_verdict(audit_state, v)).pack(side="left", padx=2)
    ttk.Button(btns, text="스냅샷 보기", command=lambda: _open_snapshot(audit_state)).pack(side="left", padx=4)
    ttk.Button(btns, text="클립 열기", command=lambda: _open_clip(audit_state)).pack(side="left", padx=2)

    # 더블클릭: 스냅샷/클립
    tree.bind("<Double-1>", lambda e: _open_snapshot(audit_state))

    # 초기 표본 뽑기
    _draw_samples(audit_state, tab)

    # 블라인드 원칙: 표시 중 자동 카운트·플래그·정답 여부 숨김
    # (tree에 flags/정답 컬럼 없음)


def _draw_samples(state, tab):
    """현재 라운드 표본 선정 → 트리 갱신."""
    from .audit import select_samples, snapshot_name

    round_no = state.round_var.get()
    state.round_no = round_no

    # 이전 라운드 exclude
    exclude = set()
    from .audit import load_audit
    audit = load_audit(state.folder)
    if audit:
        for r in audit.get("rounds", []):
            if r["round"] < round_no:
                for s in r.get("samples", []):
                    exclude.add((s["kind"], s["point"], s["frame"]))

    samples = select_samples(
        state.view.events,
        state.view.processed_frames,
        state.view.config.way if state.view.config else 6,
        seed=20260711,
        round_no=round_no,
        exclude=exclude,
    )
    state.samples = samples
    state.verdicts = {}

    tree = state.tree
    tree.delete(*tree.get_children())
    for i, s in enumerate(samples):
        kind_disp = "이벤트" if s.kind == "event" else "무이벤트"
        time_s = s.frame / state.view.view.recording.fps if state.view and state.view.recording else 0
        window = f"[{s.window[0]},{s.window[1]}]" if s.window else ""
        tree.insert("", "end", iid=str(i), values=(i, kind_disp, s.point, s.frame, f"{time_s:.2f}", ""))

    # estimates 갱신
    from .audit import estimates, load_audit
    audit = load_audit(state.folder)
    if audit:
        est = estimates(audit.get("rounds", []))
        _update_est_label(state, est)


def _set_verdict(state, verdict):
    sel = state.tree.selection()
    if not sel:
        return
    idx = int(sel[0])
    state.verdicts[idx] = verdict
    state.tree.set(sel[0], "verdict", verdict)
    _refresh_estimates(state)


def _save_round(state, tab):
    """라운드 저장 → audit.json append + estimates 갱신."""
    from .audit import append_round
    verdicts_list = []
    for i, s in enumerate(state.samples):
        v = state.verdicts.get(i, "모름")
        verdicts_list.append({
            "kind": s.kind,
            "point": s.point,
            "frame": s.frame,
            "window": s.window,
            "verdict": v,
            "snapshot": snapshot_name(state.round_no, s.kind, i, s.point, s.frame),
        })
    append_round(state.folder, state.round_no, verdicts_list)
    messagebox.showinfo("저장", f"라운드 {state.round_no} 저장 완료")
    state.round_var.set(state.round_no + 1)
    _draw_samples(state, tab)


def _open_snapshot(state):
    sel = state.tree.selection()
    if not sel:
        return
    idx = int(sel[0])
    s = state.samples[idx]
    snap = snapshot_name(state.round_no, s.kind, idx, s.point, s.frame)
    path = state.folder / snap
    if path.exists():
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    else:
        messagebox.showinfo("스냅샷 없음", "스냅샷 파일이 없습니다 (감사 라운드 1은 측정 시 사전 저장됨).")


def _open_clip(state):
    sel = state.tree.selection()
    if not sel:
        return
    idx = int(sel[0])
    s = state.samples[idx]
    if state.make_clip:
        try:
            clip_path = state.make_clip(s.point, s.frame)
            import subprocess, sys
            if sys.platform == "win32":
                os.startfile(clip_path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", clip_path])
            else:
                subprocess.run(["xdg-open", clip_path])
        except Exception as e:
            messagebox.showerror("클립 열기 실패", str(e))
    else:
        messagebox.showinfo("클립 기능", "클립 생성 모듈(P34 clipgen)이 연결되지 않았습니다.")


def _refresh_estimates(state):
    from .audit import estimates, load_audit
    audit = load_audit(state.folder)
    if audit:
        est = estimates(audit.get("rounds", []))
        _update_est_label(state, est)


def _update_est_label(state, est):
    parts = []
    if est.get("overcount_rate") is not None:
        parts.append(f"과계수율 {est['overcount_rate']:.1%} (95% 상한 {est.get('overcount_upper95')})")
    if est.get("miss_rate") is not None:
        parts.append(f"미계수율 {est['miss_rate']:.1%} (95% 상한 {est.get('miss_upper95')})")
    if not parts:
        state.est_var.set("추정치: 표본 부족")
    else:
        state.est_var.set("추정: " + " | ".join(parts))