"""멀티웨이 데이터 모델: 설정 계열(설계 §1.1) + 결과 계열(§1.3).

각도 규약: angle_deg는 흐름축과 화면 +x의 각, 양수 = 화면에서 위로(반시계).
흐름 단위벡터 u = (cos a, -sin a), 폭 방향 n = (sin a, cos a)  [+y = 아래].
"""

import math
from dataclasses import dataclass, field

from ..config import CounterConfig

VALID_DIRECTIONS = ("+x", "-x", "+y", "-y")
VALID_MODES = ("dma", "memory")
VALID_SHAPES = ("circle", "ellipse")
MAX_POINTS = 9


@dataclass(frozen=True)
class PointSpec:
    """Point 1개 = 회전 사각형 ROI + 메타. 좌표·크기는 프레임 비율(0~1).

    length 흐름 방향 변(계수 밴드가 놓이는 축), width 채널을 가로지르는 변.
    """

    number: int
    cx: float
    cy: float
    length: float
    width: float
    angle_deg: float
    direction: str | None = None     # None 전역 방향 따름
    target: int = 500

    def __post_init__(self) -> None:
        if not (isinstance(self.number, int) and self.number >= 1):
            raise ValueError(f"number는 1 이상의 정수여야 합니다: {self.number}")
        if not (0 <= self.cx <= 1):
            raise ValueError(f"cx는 [0, 1] 범위여야 합니다: {self.cx}")
        if not (0 <= self.cy <= 1):
            raise ValueError(f"cy는 [0, 1] 범위여야 합니다: {self.cy}")
        if not (0 < self.length <= 1):
            raise ValueError(f"length는 (0, 1] 범위여야 합니다: {self.length}")
        if not (0 < self.width <= 1):
            raise ValueError(f"width는 (0, 1] 범위여야 합니다: {self.width}")
        if not math.isfinite(self.angle_deg):
            raise ValueError(f"angle_deg는 유한수여야 합니다: {self.angle_deg}")
        if self.direction is not None and self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction은 None 또는 {VALID_DIRECTIONS} 중 하나여야 합니다: {self.direction!r}")
        if not (isinstance(self.target, int) and self.target >= 0):
            raise ValueError(f"target은 0 이상의 정수여야 합니다: {self.target}")

    def flow_unit(self) -> tuple[float, float]:
        """흐름 단위벡터 (ux, uy) = (cos a, -sin a)."""
        a = math.radians(self.angle_deg)
        return (math.cos(a), -math.sin(a))

    def corners_px(self, frame_w: int, frame_h: int) -> tuple[tuple[float, float], ...]:
        """회전 사각형 4꼭짓점(px, 순서: 상류폭선 2점 → 하류폭선 2점)."""
        a = math.radians(self.angle_deg)
        ux, uy = math.cos(a), -math.sin(a)
        nx, ny = math.sin(a), math.cos(a)
        cx = self.cx * frame_w
        cy = self.cy * frame_h
        hl = self.length * frame_w / 2.0
        hw = self.width * frame_h / 2.0
        return (
            (cx - hl * ux - hw * nx, cy - hl * uy - hw * ny),
            (cx - hl * ux + hw * nx, cy - hl * uy + hw * ny),
            (cx + hl * ux - hw * nx, cy + hl * uy - hw * ny),
            (cx + hl * ux + hw * nx, cy + hl * uy + hw * ny),
        )


@dataclass(frozen=True)
class BubbleSizeSpec:
    """기포 기준 크기 — 사람이 GUI '기포 크기 재기'로 직접 지정(요구사항).

    크기 판정은 병합 후 리듬 마크 단위(P1 패스)에서 min_accept_px/merge_suspect_px로 수행.
    count_half_bubbles: 쪼개진 반쪽 기포도 각각 1개로 셀지(인터뷰 Q3 확정 토글).
      True면 인정 하한이 절반이 된다(반쪽 크기까지 수용).
    """

    shape: str            # "circle" | "ellipse"
    major_px: float
    minor_px: float
    accept_low: float = 0.5
    accept_high: float = 1.8
    count_half_bubbles: bool = False

    def __post_init__(self) -> None:
        if self.shape not in VALID_SHAPES:
            raise ValueError(f"shape는 {VALID_SHAPES} 중 하나여야 합니다: {self.shape!r}")
        if not (math.isfinite(self.major_px) and self.major_px > 0):
            raise ValueError(f"major_px는 0보다 큰 유한수여야 합니다: {self.major_px}")
        if not (math.isfinite(self.minor_px) and self.minor_px > 0):
            raise ValueError(f"minor_px는 0보다 큰 유한수여야 합니다: {self.minor_px}")
        if self.minor_px > self.major_px:
            raise ValueError(
                f"minor_px는 major_px 이하여야 합니다: {self.minor_px} > {self.major_px}")
        if self.shape == "circle" and self.major_px != self.minor_px:
            raise ValueError(
                f"circle은 major_px와 minor_px가 같아야 합니다: {self.major_px} != {self.minor_px}")
        if not (0 < self.accept_low < 1):
            raise ValueError(f"accept_low는 (0, 1) 범위여야 합니다: {self.accept_low}")
        if not (math.isfinite(self.accept_high) and self.accept_high > 1):
            raise ValueError(f"accept_high는 1보다 큰 유한수여야 합니다: {self.accept_high}")

    def min_accept_px(self) -> float:
        base = min(self.major_px, self.minor_px) * self.accept_low
        return base * 0.5 if self.count_half_bubbles else base

    def merge_suspect_px(self) -> float:
        return max(self.major_px, self.minor_px) * self.accept_high


@dataclass(frozen=True)
class RecordingSpec:
    """HAS-U2 녹화 설정 — 측정 전 선행 선택 필수. 컨테이너 fps는 절대 신뢰하지 않음."""

    width: int
    height: int
    fps: float          # 실촬영 fps — 모든 시간 환산의 유일한 기준
    mode: str           # "dma" | "memory"

    def __post_init__(self) -> None:
        if not (isinstance(self.width, int) and self.width > 0):
            raise ValueError(f"width는 1 이상의 정수여야 합니다: {self.width}")
        if not (isinstance(self.height, int) and self.height > 0):
            raise ValueError(f"height는 1 이상의 정수여야 합니다: {self.height}")
        if not (math.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps는 0보다 큰 유한수여야 합니다: {self.fps}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode는 {VALID_MODES} 중 하나여야 합니다: {self.mode!r}")


@dataclass(frozen=True)
class MultiwayConfig:
    """멀티웨이 1회 측정의 전체 설정. 기존 CounterConfig는 조합(상속 아님)."""

    recording: RecordingSpec
    points: tuple[PointSpec, ...]
    bubble: BubbleSizeSpec
    global_direction: str = "+x"
    order_tolerance_frames: int = 0
    order_gate: bool = True          # D-21: True=순서 맞을 때만 +1(count error 미계수)
    base: CounterConfig = field(default_factory=CounterConfig)

    def __post_init__(self) -> None:
        if not (1 <= len(self.points) <= MAX_POINTS):
            raise ValueError(
                f"points는 1~{MAX_POINTS}개여야 합니다: {len(self.points)}개")
        numbers = [p.number for p in self.points]
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        if dupes:
            raise ValueError(f"Point 번호가 중복됩니다: {dupes}")
        expected = set(range(1, len(self.points) + 1))
        missing = sorted(expected - set(numbers))
        if missing:
            raise ValueError(
                f"Point 번호는 1..{len(self.points)} 연속이어야 합니다 — 없는 번호: {missing}")
        if self.global_direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"global_direction은 {VALID_DIRECTIONS} 중 하나여야 합니다: "
                f"{self.global_direction!r}")
        if not (isinstance(self.order_tolerance_frames, int)
                and self.order_tolerance_frames >= 0):
            raise ValueError(
                f"order_tolerance_frames는 0 이상의 정수여야 합니다: "
                f"{self.order_tolerance_frames}")
        w, h = self.recording.width, self.recording.height
        for p in self.points:
            for (x, y) in p.corners_px(w, h):
                if not (-1e-6 <= x <= w + 1e-6 and -1e-6 <= y <= h + 1e-6):
                    raise ValueError(
                        f"Point {p.number}의 회전 사각형이 프레임을 벗어납니다: "
                        f"꼭짓점 ({x:.1f}, {y:.1f}), 프레임 {w}x{h}")

    @property
    def way(self) -> int:
        return len(self.points)

    def point_by_number(self, n: int) -> PointSpec:
        for p in self.points:
            if p.number == n:
                return p
        raise KeyError(f"Point {n}이(가) 없습니다")

    def effective_direction(self, p: PointSpec) -> str:
        return p.direction if p.direction is not None else self.global_direction


@dataclass(frozen=True)
class BubbleEvent:
    """기포 1개의 통과 이벤트 — 1차 데이터. 집계는 전부 파생값."""

    point: int
    frame: int                     # 프레임 번호 = 1차 좌표 (HAS-XViewer 대조용)
    subframe: float = 0.0          # 0~1 밴드 침투 깊이 보간 — 동일 프레임 tie-break
    flags: tuple[str, ...] = ()    # "reflux-nearby" | "merge-suspect" | "slow" 등
    # ─ 신규 (R0 §1) ─
    position_px: float = -1.0   # dn 마크 선축 중심 = Mark.position_px (패치 폭축 x_p 좌표계)
    size_px: float = -1.0       # dn 마크 폭 = float(Mark.width_px). -1.0 = 미기록(구버전)


@dataclass(frozen=True)
class SequenceViolation:
    frame: int
    expected_point: int
    actual_point: int


@dataclass(frozen=True)
class SuspectMoment:
    kind: str                      # "merge" | "reflux" | "slow" | "gap-anomaly" | "undersize"
    point: int
    frame: int
    detail: str                    # 사람이 읽는 사유 (한국어)
    snapshot: str | None = None    # suspects/ 내 PNG 상대경로
    confidence: str = ""           # D-15: reflux 계열 후처리 확신 등급 "높음"|"낮음"(그 외 kind는 "")


@dataclass(frozen=True)
class Correction:
    """수동 보정 이력 — 자동 측정 원본은 절대 덮어쓰지 않는다."""

    point: int
    frame: int
    delta: int                     # +1 / -1
    reason: str
    corrected_at: str              # ISO 시각
    # ─ 신규 (R0 §1) ─
    evidence_path: str = ""     # 결과 폴더 기준 상대경로(스냅샷/클립), 없으면 ""

    def __post_init__(self) -> None:
        if self.delta not in (-1, 1):
            raise ValueError(f"delta는 +1 또는 -1이어야 합니다: {self.delta}")


@dataclass(frozen=True)
class ForeignObject:
    """정지 이물 1개 (파티클/기포/모름). foreign.py가 생성, 검토 센터가 태깅. (R0 §1)"""

    id: int                        # 파일 내 0부터 연번
    x_px: float                    # 프레임 좌표 중심 (예: 1_2 파티클 ≈ (1118, 63))
    y_px: float
    w_px: float
    h_px: float
    area_px: float
    near_points: tuple[int, ...]   # 계수선 밴드±마진에 걸치는 Point 번호들 (빈 튜플 가능)
    source: str                    # "preflight" | "review-scan"
    tag: str = ""                  # "" 미태깅 | "파티클" | "기포" | "모름" (D7)
    tagged_at: str = ""            # ISO 시각, 미태깅이면 ""
    note: str = ""


@dataclass(frozen=True)
class AuditSample:
    """감사 표본 1개 (R0 §3.5)."""

    kind: str                    # "event" | "empty"
    point: int
    frame: int
    window: tuple[int, int] | None = None    # empty만: (시작, 끝) 프레임


@dataclass(frozen=True)
class StabilizationStats:
    """D-16: 챕터 전체 카메라 흔들림 보정 통계. 평행이동만 다룬다(회전·줌 보정은 범위 밖)."""

    mean_px: float = 0.0
    max_px: float = 0.0
    corrected_frames: int = 0


@dataclass
class ChapterResult:
    """챕터 = 영상 파일 1개, 완전 독립."""

    video_path: str
    recording: RecordingSpec
    counts_auto: dict[int, int]
    corrections: list[Correction]
    events: list[BubbleEvent]
    violations: list[SequenceViolation]
    suspects: list[SuspectMoment]
    sets_completed: int
    processed_frames: int
    # 후행 기본값 필드 (A 트랙 엔진이 채움 — 2026-07-10 추가, 순수 가법)
    filtered_out: list[tuple[int, int, float, str]] = field(default_factory=list)  # (point, frame, size_px, reason) R0 §1
    self_check_messages: list[str] = field(default_factory=list)
    suspects_truncated: bool = False
    cancelled: bool = False
    stabilization: StabilizationStats = field(default_factory=StabilizationStats)  # D-16 추가, 순수 가법

    def counts_final(self) -> dict[int, int]:
        final = dict(self.counts_auto)
        for c in self.corrections:
            final[c.point] = final.get(c.point, 0) + c.delta
        return final

    def errors(self, targets: dict[int, int]) -> dict[int, int]:
        final = self.counts_final()
        return {n: final.get(n, 0) - t for n, t in targets.items()}


@dataclass
class PhaseResult:
    """페이즈 = 챕터 묶음. 합산은 파생값."""

    name: str
    chapters: list[ChapterResult]

    def sum_counts_final(self) -> dict[int, int]:
        total: dict[int, int] = {}
        for ch in self.chapters:
            for n, v in ch.counts_final().items():
                total[n] = total.get(n, 0) + v
        return total

    def sum_sets(self) -> int:
        return sum(ch.sets_completed for ch in self.chapters)
