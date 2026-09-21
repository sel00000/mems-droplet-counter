"""멀티웨이 위저드 상태·입력 모델 — tkinter를 import하지 않는 순수 모듈.

설계 §2의 6단계 위저드 흐름(순서 강제), 챕터 목록, ROI 편집기 팝업/프레임 입력 파서,
방향 화살표, 프리플라이트 표시모델을 tkinter 없이 담는다. gui_multiway.py(tkinter)가
위젯 이벤트에서 이 클래스/함수를 호출한다. gui_params.py와 같은 분리 패턴.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum

from bubble_counter.multiway.model import RecordingSpec, VALID_DIRECTIONS


class WizardStep(IntEnum):
    RECORDING = 1      # 녹화 설정 선택
    CHAPTERS = 2       # 챕터 목록 구성
    ROI = 3            # ROI 배치
    PREFLIGHT = 4      # 프리플라이트
    RUN = 5            # 측정 실행
    RESULTS = 6        # 결과·검토


class WizardState:
    """위저드 1회 세션의 진행 상태. 순서 강제형(설계 §2.1, 방식 A)."""

    def __init__(self) -> None:
        self.step: WizardStep = WizardStep.RECORDING
        self.recording: RecordingSpec | None = None
        self.chapter_count: int = 0
        self.roi_count: int = 0
        self.preflight_passed: bool = False
        self.run_complete: bool = False

    def video_open_enabled(self) -> bool:
        # 1단계 [영상 열기] = 2단계에서 추가한 1번 챕터 미리보기(D-6).
        # 녹화 설정만으로는 열 영상이 없으므로 챕터≥1 필요.
        return self.recording is not None and self.chapter_count >= 1

    def can_advance(self) -> tuple[bool, str]:
        if self.step == WizardStep.RECORDING:
            if self.recording is None:
                return (False, "녹화 설정을 먼저 선택하세요")
        elif self.step == WizardStep.CHAPTERS:
            if self.chapter_count < 1:
                return (False, "챕터를 1개 이상 추가하세요")
        elif self.step == WizardStep.ROI:
            if self.roi_count < 1:
                return (False, "ROI를 1개 이상 배치하세요")
        elif self.step == WizardStep.PREFLIGHT:
            if not self.preflight_passed:
                return (False, "프리플라이트 차단 항목을 해결하세요")
        elif self.step == WizardStep.RUN:
            if not self.run_complete:
                return (False, "측정이 완료되지 않았습니다")
        else:  # RESULTS
            return (False, "마지막 단계입니다")
        return (True, "")

    def advance(self) -> None:
        if self.can_advance()[0] and self.step < WizardStep.RESULTS:
            self.step = WizardStep(self.step + 1)

    def back(self) -> None:
        if self.step > WizardStep.RECORDING:
            self.step = WizardStep(self.step - 1)

    def invalidate_preflight(self) -> None:
        """ROI 편집 후 상태 위생(D-13 후속, review-D13-verdict.md Critical-1).

        preflight_passed/run_complete는 한 번 True가 되면 되돌아가기(back)로도
        리셋되지 않는 stale 플래그라, "◀ 이전 → ROI 편집 → 다음 ▶"으로 되돌아오면
        프리플라이트 재검증 없이 전진할 수 있었다. ROI 추가/이동/회전/삭제 직후
        이를 호출해 4단계 재통과를 강제한다. run_complete는 건드리지 않는다 —
        6단계(RESULTS)의 기존 결과 열람은 계속 허용한다.
        """
        self.preflight_passed = False


@dataclass(frozen=True)
class Chapter:
    path: str
    width: int
    height: int
    n_frames: int


class ChapterList:
    """페이즈 1개를 이루는 챕터(영상 파일)들. 추가 순서 = 챕터 순서(설계 §2.2)."""

    def __init__(self) -> None:
        self._items: list[Chapter] = []

    def add(self, path: str, width: int, height: int, n_frames: int) -> None:
        self._items.append(Chapter(path, width, height, n_frames))

    def remove(self, index: int) -> None:
        del self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def phase_name(self) -> str:
        stems = [os.path.splitext(os.path.basename(c.path))[0] for c in self._items]
        if not stems:
            return "phase"
        prefix = os.path.commonprefix(stems).rstrip("_-. ")
        return prefix or "phase"

    def resolution_mismatches(self, rec: RecordingSpec) -> list[int]:
        return [i for i, c in enumerate(self._items)
                if (c.width, c.height) != (rec.width, rec.height)]

    def mode_contradictions(self, rec: RecordingSpec,
                            memory_bytes: int = 2 * 1024 ** 3) -> list[int]:
        if rec.mode != "memory":
            return []
        return [i for i, c in enumerate(self._items)
                if c.width * c.height * 3 * c.n_frames > memory_bytes]


DIRECTION_ARROWS = {"+x": "→", "-x": "←", "+y": "↓", "-y": "↑"}


def parse_frame_input(text: str, total_frames: int) -> int:
    try:
        n = int(str(text).strip())
    except ValueError:
        raise ValueError(f"프레임 번호는 정수여야 합니다: {text!r}") from None
    if not (0 <= n < total_frames):
        raise ValueError(f"프레임 번호는 0~{total_frames - 1} 범위여야 합니다: {n}")
    return n


def parse_target(text: str) -> int:
    try:
        n = int(str(text).strip())
    except ValueError:
        raise ValueError(f"목표 개수는 0 이상의 정수여야 합니다: {text!r}") from None
    if n < 0:
        raise ValueError(f"목표 개수는 0 이상의 정수여야 합니다: {n}")
    return n


def parse_point_number(text: str, existing) -> int:
    try:
        n = int(str(text).strip())
    except ValueError:
        raise ValueError(f"Point 번호는 1 이상의 정수여야 합니다: {text!r}") from None
    if n < 1:
        raise ValueError(f"Point 번호는 1 이상의 정수여야 합니다: {n}")
    if n in tuple(existing):
        raise ValueError(f"이미 사용 중인 Point 번호입니다: {n}")
    return n


def arrow_for(direction: str) -> str:
    if direction not in DIRECTION_ARROWS:
        raise ValueError(f"direction은 {VALID_DIRECTIONS} 중 하나여야 합니다: {direction!r}")
    return DIRECTION_ARROWS[direction]


@dataclass(frozen=True)
class PreflightItem:
    severity: str            # "block" | "warn" | "info"(D-16: 문제 아닌 상태 정보)
    message: str


class PreflightDisplay:
    """A트랙 engine.preflight()가 정규화해 넘긴 항목을 차단/경고/정보로 나눠 표시(설계 §2.3)."""

    def __init__(self, blocking: list[str], warnings: list[str],
                info: "list[str] | None" = None) -> None:
        self.blocking = blocking
        self.warnings = warnings
        self.info = info if info is not None else []   # D-16: 흔들림 σ 등(경고 아님)

    @property
    def has_blocking(self) -> bool:
        return bool(self.blocking)

    @classmethod
    def from_items(cls, items) -> "PreflightDisplay":
        blocking = [it.message for it in items if it.severity == "block"]
        warnings = [it.message for it in items if it.severity == "warn"]
        info = [it.message for it in items if it.severity == "info"]
        return cls(blocking, warnings, info)
