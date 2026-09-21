"""측정 중 라이브 뷰 패널 (tkinter). 로직은 live.py — 이 모듈은 표시만."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
import numpy as np

from bubble_counter.gui_view import ppm_p6_bytes
from bubble_counter.live import LiveControl, LiveSnapshot

_SPEEDS = ("1.0", "0.5", "0.25", "0.1")
_DELAY_NOTE = "카운트는 판정 확정 후 반영됩니다(짧은 지연 가능)."


class LiveViewPanel(ttk.LabelFrame):
    """영상 미리보기 + Point/총 카운트 + 배속 + 표시 일시정지."""

    def __init__(
        self,
        parent,
        live_control: LiveControl,
        *,
        overlay_draw: "Callable[[np.ndarray, LiveSnapshot], None] | None" = None,
        title: str = "라이브 뷰",
    ) -> None:
        super().__init__(parent, text=title, padding=4)
        self._control = live_control
        self._overlay_draw = overlay_draw
        self._photo = None
        self._frozen: LiveSnapshot | None = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        self._label_var = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self._label_var).pack(side="left")
        ttk.Label(top, text="배속").pack(side="left", padx=(12, 2))
        self._speed_var = tk.StringVar(value="1.0")
        sp = ttk.Combobox(
            top, textvariable=self._speed_var, values=_SPEEDS, width=6, state="readonly")
        sp.pack(side="left")
        sp.bind("<<ComboboxSelected>>", self._on_speed)
        self._pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top, text="화면 일시정지", variable=self._pause_var,
            command=self._on_pause).pack(side="left", padx=8)

        self._canvas = tk.Canvas(self, width=640, height=360, bg="#222", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, pady=4)

        self._counts_var = tk.StringVar(value="카운트: —")
        ttk.Label(self, textvariable=self._counts_var, font=("", 11, "bold")).pack(anchor="w")
        ttk.Label(self, text=_DELAY_NOTE, foreground="#666").pack(anchor="w")

    def _on_speed(self, _evt=None) -> None:
        try:
            self._control.set_speed(float(self._speed_var.get()))
        except ValueError:
            self._control.set_speed(1.0)

    def _on_pause(self) -> None:
        self._control.set_display_paused(self._pause_var.get())

    def reset(self) -> None:
        self._frozen = None
        self._counts_var.set("카운트: —")
        self._label_var.set("—")
        self._canvas.delete("all")

    def apply_snapshot(self, snap: LiveSnapshot) -> None:
        if self._control.is_display_paused():
            if self._frozen is None:
                self._frozen = snap
                self._show(snap)
            return
        self._frozen = None
        self._show(snap)

    def _show(self, snap: LiveSnapshot) -> None:
        self._label_var.set(
            f"{snap.label}  |  f={snap.frame_idx}"
            + (f"/{snap.total_frames}" if snap.total_frames > 0 else "")
            + f"  |  {snap.processing_fps:.0f} fps"
        )
        parts = []
        if "total" in snap.counts and len(snap.counts) == 1:
            parts.append(f"총 {snap.counts['total']}")
        else:
            for k in sorted(snap.counts.keys(), key=lambda x: (str(type(x)), x)):
                parts.append(f"P{k}={snap.counts[k]}")
        self._counts_var.set("카운트: " + ("  ".join(parts) if parts else "—"))

        arr = np.frombuffer(snap.jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if self._overlay_draw is not None:
            try:
                self._overlay_draw(bgr, snap)
            except Exception:
                pass
        rgb = bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        self._photo = tk.PhotoImage(data=ppm_p6_bytes(rgb), format="PPM")
        self._canvas.config(width=w, height=h)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)


def draw_multiway_overlay(bgr: np.ndarray, snap: LiveSnapshot, points) -> None:
    """points: iterable of PointSpec-like with corners_px(w,h) and number."""
    h, w = bgr.shape[:2]
    for p in points:
        try:
            corners = p.corners_px(w, h)
        except Exception:
            continue
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        # 계수선 ≈ 흐름 중점 가로
        c0 = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        c1 = ((corners[2][0] + corners[3][0]) / 2, (corners[2][1] + corners[3][1]) / 2)
        mid0 = ((c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2)
        # 폭 방향 선분 (중점 가로지름)
        n0 = ((corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2)
        n1 = ((corners[1][0] + corners[3][0]) / 2, (corners[1][1] + corners[3][1]) / 2)
        cv2.line(bgr, (int(n0[0]), int(n0[1])), (int(n1[0]), int(n1[1])),
                 (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"P{p.number}", (int(mid0[0]) + 4, int(mid0[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"f={snap.frame_idx}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_single_overlay(bgr: np.ndarray, snap: LiveSnapshot, cfg) -> None:
    """cfg: CounterConfig — ROI 전체 미리보기 가정 시 선만 그림."""
    h, w = bgr.shape[:2]
    axis = cfg.line.axis
    pos = cfg.line.pos
    if axis == "h":
        y = int(round(pos * (h - 1)))
        cv2.line(bgr, (0, y), (w - 1, y), (0, 0, 255), 1, cv2.LINE_AA)
    else:
        x = int(round(pos * (w - 1)))
        cv2.line(bgr, (x, 0), (x, h - 1), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"f={snap.frame_idx}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
