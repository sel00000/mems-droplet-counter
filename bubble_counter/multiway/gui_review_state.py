"""검토 센터 UI 상태기계 (P2, tkinter-free).

필터·선택·보정 스테이징 상태만 스테이징을 순수 로직으로 분리해 단위테스트 가능하게.
GUI(tkinter) 레이어는 이 상태만 조작/구독.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Iterable

from .loader import ChapterView
from .model import ChapterResult, Correction, MultiwayConfig


class Tab(Enum):
    EVENTS = auto()        # 이벤트 탭
    COUNT_ERRORS = auto()  # count error 탭 (D-21 순서 미계수 사건)
    FILTERED = auto()      # 배제됨 탭
    SUSPECTS = auto()      # 의심 탭
    FOREIGN = auto()       # 이물 탭
    AUDIT = auto()         # 감사 탭


class FilterMode(Enum):
    ALL = "전체"
    COUNTED = "계수됨"
    FILTERED = "배제됨"
    WEAK = "약기포"
    DUPLICATE = "중복"
    RING_RESIDUAL = "링 잔차"
    BOUNDARY_INCOMPLETE = "미완 통과"
    UNDERSIZE = "크기 미달"
    NO_UPSTREAM = "상류 없음"
    MERGE_SUSPECT = "겹침 의심"
    BOUNDARY_SIZE = "경계 크기"


@dataclass
class ReviewState:
    """검토 센터 전체 상태 (불변 아님 — GUI가 직접 수정)."""

    view: ChapterView
    config: MultiwayConfig
    folder: Path
    reload_callback: Callable[[], None] | None = None
    open_frame_callback: Callable[[int, int | None], None] | None = None
    make_clip_callback: Callable[[int, int], Path] | None = None

    # 현재 탭
    current_tab: Tab = Tab.EVENTS

    # 필터 상태
    event_filters: set[FilterMode] = field(default_factory=set)
    filtered_filters: set[FilterMode] = field(default_factory=set)
    suspect_filters: set[str] = field(default_factory=set)  # kind 기반
    foreign_tag_filter: str = ""  # "", "파티클", "기포", "모름"

    # 선택 상태 (인덱스 기반 — 데이터 변경 시 재조정 필요)
    selected_event_indices: set[int] = field(default_factory=set)
    selected_filtered_indices: set[int] = field(default_factory=set)
    selected_suspect_indices: set[int] = field(default_factory=set)
    selected_foreign_indices: set[int] = field(default_factory=set)

    # 보정 스테이징 (아직 로그에 반영되지 않은 pending 변경)
    pending_corrections: list[Correction] = field(default_factory=list)

    # UI 편의
    sort_key: str = "frame"
    sort_ascending: bool = True

    # ── 파생 데이터 접근 ──

    def events_filtered(self) -> list[tuple[int, int]]:
        """필터 적용된 이벤트 (인덱스, 이벤트) 리스트."""
        out = []
        for i, e in enumerate(self.view.events):
            if self._match_event_filters(e):
                out.append((i, e))
        return out

    def filtered_filtered(self) -> list[tuple[int, tuple]]:
        out = []
        for i, f in enumerate(self.view.filtered):
            if self._match_filtered_filters(f):
                out.append((i, f))
        return out

    def suspects_filtered(self) -> list[tuple[int, int]]:
        out = []
        for i, s in enumerate(self.view.suspects):
            if not self.suspect_filters or s.kind in self.suspect_filters:
                out.append((i, s))
        return out

    def foreign_filtered(self) -> list[tuple[int, int]]:
        out = []
        if not self.view.foreign:
            return out
        for i, fo in enumerate(self.view.foreign.get("objects", [])):
            tag = fo.get("tag", "")
            if not self.foreign_tag_filter or tag == self.foreign_tag_filter:
                out.append((i, fo))
        return out

    def violations_sorted(self) -> list:
        """count error(SequenceViolation) 프레임 순 정렬 — 탭 표와 뷰어 순회의 단일 소스."""
        return sorted(self.view.violations, key=lambda v: v.frame)

    def violation_targets(self) -> list[tuple[int, int | None, str]]:
        """프레임 뷰어 순회용 (frame, 강조 Point, 라벨). violations_sorted와 인덱스 정렬 일치."""
        vs = self.violations_sorted()
        n = len(vs)
        return [(v.frame, v.actual_point,
                 f"에러 {i + 1}/{n} — 기대 P{v.expected_point}, 실제 P{v.actual_point}")
                for i, v in enumerate(vs)]

    def _match_event_filters(self, e) -> bool:
        if not self.event_filters:
            return True
        flags = set(e.flags)
        if FilterMode.COUNTED in self.event_filters and not flags:
            return False
        if FilterMode.FILTERED in self.event_filters:
            return False  # 이벤트 탭엔 배제된 것 없음
        if FilterMode.WEAK in self.event_filters and "weak-bubble" not in flags:
            return False
        if FilterMode.DUPLICATE in self.event_filters and "duplicate" not in flags:
            return False
        if FilterMode.RING_RESIDUAL in self.event_filters and "ring-residual" not in flags:
            return False
        if FilterMode.BOUNDARY_INCOMPLETE in self.event_filters and "boundary-incomplete" not in flags:
            return False
        if FilterMode.UNDERSIZE in self.event_filters and "undersize" not in flags:
            return False
        if FilterMode.NO_UPSTREAM in self.event_filters and "no-upstream" not in flags:
            return False
        if FilterMode.MERGE_SUSPECT in self.event_filters and "merge-suspect" not in flags:
            return False
        if FilterMode.BOUNDARY_SIZE in self.event_filters and "boundary-size" not in flags:
            return False
        return True

    def _match_filtered_filters(self, f) -> bool:
        if not self.filtered_filters:
            return True
        reason = f[3] if len(f) > 3 else ""
        if FilterMode.FILTERED in self.filtered_filters:
            return True
        if FilterMode.WEAK in self.filtered_filters and reason == "undersize":
            return True
        if FilterMode.DUPLICATE in self.filtered_filters and reason == "duplicate":
            return True
        if FilterMode.RING_RESIDUAL in self.filtered_filters and reason == "ring-residual":
            return True
        if FilterMode.BOUNDARY_INCOMPLETE in self.filtered_filters and reason == "boundary-incomplete":
            return True
        if FilterMode.UNDERSIZE in self.filtered_filters and reason == "undersize":
            return True
        return False

    # ── 보정 스테이징 ──

    def stage_correction(self, point: int, frame: int, delta: int, reason: str,
                         evidence_path: str = "") -> None:
        """보정 스테이징 (아직 로그 미반영)."""
        from .corrections import make_correction
        c = make_correction(point, frame, delta, reason, evidence_path)
        self.pending_corrections.append(c)

    def commit_pending(self) -> None:
        """스테이징된 보정을 로그에 반영 + 뷰 갱신."""
        if not self.pending_corrections:
            return
        from .corrections import append_corrections
        append_corrections(self.folder, self.pending_corrections)
        self.pending_corrections.clear()
        if self.reload_callback:
            self.reload_callback()

    def discard_pending(self) -> None:
        self.pending_corrections.clear()

    def counts_preview(self) -> dict[int, int]:
        """현재 보정 반영된 미리보기 카운트."""
        from .corrections import counts_with
        base = self.view.counts_auto
        all_cs = list(self.view.corrections) + self.pending_corrections
        return counts_with(base, all_cs)

    # ── 정렬/선택 편의 ──

    def sort(self, key: str) -> None:
        if self.sort_key == key:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_key = key
            self.sort_ascending = True

    def select_all_events(self) -> None:
        self.selected_event_indices = {i for i, _ in self.events_filtered()}

    def clear_selection_events(self) -> None:
        self.selected_event_indices.clear()

    def toggle_event(self, idx: int) -> None:
        if idx in self.selected_event_indices:
            self.selected_event_indices.remove(idx)
        else:
            self.selected_event_indices.add(idx)