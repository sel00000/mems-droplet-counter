"""ROI 프로파일 저장소(~/.bubble_counter_profiles.json) — 설계 §1.1+§1.4.

프로파일은 recording·points·전역방향·순서허용창·**bubble**을 저장한다(트랙 B의
ProfileStore.load→MultiwayConfig 계약이 완전 config 복원을 요구). CounterConfig
(base)는 저장하지 않는다(기본값 재구성).

원자적 저장(mkstemp + os.replace)·BOM 로드 패턴은 gui_settings.py와 동일하다.
손상·부재 파일은 예외 없이 빈 상태로 폴백한다(설정 문제가 GUI 실행을 막지 않게).
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..config import CounterConfig
from .model import BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec

DEFAULT_PROFILES_PATH = Path.home() / ".bubble_counter_profiles.json"
PROFILE_VERSION = 1


def config_to_profile(config: MultiwayConfig) -> dict:
    """MultiwayConfig를 직렬화 가능한 dict로(bubble 포함, base 제외)."""
    r = config.recording
    return {
        "recording": {"width": r.width, "height": r.height, "fps": r.fps, "mode": r.mode},
        "global_direction": config.global_direction,
        "order_tolerance_frames": config.order_tolerance_frames,
        "order_gate": bool(getattr(config, "order_gate", True)),
        "points": [dataclasses.asdict(p) for p in config.points],
        # D-13: 기포 크기 미지정(None) 상태의 프로파일 저장도 허용(null 왕복)
        "bubble": dataclasses.asdict(config.bubble) if config.bubble is not None else None,
    }


def profile_to_config(layout: dict, base: "CounterConfig | None" = None) -> MultiwayConfig:
    """배치 dict로 완전 MultiwayConfig 복원(bubble도 layout에서).

    saved_at 등 여분 키는 무시한다. 값 위반은 각 dataclass의 __post_init__가
    한국어 ValueError로 거부한다(번호 연속성·프레임 이탈 포함).
    """
    r = layout["recording"]
    recording = RecordingSpec(
        width=int(r["width"]), height=int(r["height"]),
        fps=float(r["fps"]), mode=r["mode"])
    points = tuple(PointSpec(**pd) for pd in layout["points"])
    _bub = layout.get("bubble")                       # D-13: null(미지정) 허용
    bubble = BubbleSizeSpec(**_bub) if _bub is not None else None
    kwargs = dict(
        recording=recording, points=points, bubble=bubble,
        global_direction=layout.get("global_direction", "+x"),
        order_tolerance_frames=int(layout.get("order_tolerance_frames", 0)),
        order_gate=bool(layout.get("order_gate", True)),
    )
    if base is not None:
        kwargs["base"] = base
    return MultiwayConfig(**kwargs)


class ProfileStore:
    """~/.bubble_counter_profiles.json 저장소. gui_settings.py 원자 저장 패턴 재사용."""

    def __init__(self, path: "str | Path" = DEFAULT_PROFILES_PATH, *,
                 now: "Callable[[], str] | None" = None) -> None:
        self._path = Path(path)
        self._now = now or (lambda: datetime.now().isoformat(timespec="seconds"))

    def names(self) -> list[str]:
        profiles = self._read_all().get("profiles")
        return sorted(profiles) if isinstance(profiles, dict) else []

    def load(self, name: str) -> MultiwayConfig:
        profiles = self._read_all().get("profiles")
        entry = profiles.get(name) if isinstance(profiles, dict) else None
        if not isinstance(entry, dict):
            raise KeyError(f"프로파일이 없습니다: {name}")
        return profile_to_config(entry)

    def last_used(self) -> "str | None":
        value = self._read_all().get("last_used")
        return value if isinstance(value, str) else None

    def save(self, name: str, config: MultiwayConfig) -> None:
        data = self._read_all()
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
        entry = config_to_profile(config)
        entry["saved_at"] = self._now()
        profiles[name] = entry
        payload = {"version": PROFILE_VERSION, "last_used": name, "profiles": profiles}
        self._atomic_write(payload)

    def _read_all(self) -> dict:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _atomic_write(self, payload: dict) -> None:
        """gui_settings.py 원자 저장 패턴. 실패 시 IOError(무음 실패 없음)."""
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self._path)
        except OSError as exc:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise IOError(f"프로파일을 저장할 수 없습니다: {self._path}") from exc
