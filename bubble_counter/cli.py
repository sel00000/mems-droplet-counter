"""Command-line interface: ``count`` / ``synth`` / ``bench`` / ``gui``.

``gui`` is imported lazily (tkinter may be absent, e.g. on WSL); the other
subcommands work headless.  Argparse defaults mirror the library dataclass
defaults so the CLI and the programmatic API agree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from bubble_counter.config import CounterConfig, LineSpec, RoiSpec
from bubble_counter.multiway.model import (
    BubbleSizeSpec, MultiwayConfig, PointSpec, RecordingSpec,
)
from bubble_counter.synth import SynthConfig

_C = CounterConfig()
_L = _C.line
_S = SynthConfig()


# --------------------------------------------------------------------------
# config assembly
# --------------------------------------------------------------------------


def _counter_config_from_args(args: argparse.Namespace) -> CounterConfig:
    roi = RoiSpec(*args.roi) if args.roi is not None else RoiSpec()
    line = LineSpec(axis=args.line_axis, pos=args.line_pos, band_px=args.band)
    return CounterConfig(
        roi=roi, line=line, scale=args.scale, history=args.history,
        var_threshold=args.var_threshold, row_close_px=args.row_close,
        merge_gap_frames=args.merge_gap, min_width_px=args.min_width,
        min_area_px=args.min_area, warmup_frames=args.warmup,
        bucket_seconds=args.bucket, fps_override=args.fps,
        annotate=args.annotate, save_rhythm=args.save_rhythm,
        max_frames=args.max_frames,
    )


# --------------------------------------------------------------------------
# count
# --------------------------------------------------------------------------


def _cmd_count(args: argparse.Namespace) -> int:
    from bubble_counter.pipeline import run_count

    try:
        cfg = _counter_config_from_args(args)
    except ValueError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 1

    try:
        _print_geometry_echo(args.video, cfg)
    except (FileNotFoundError, IOError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_count(args.video, args.out, cfg, progress_cb=_print_progress,
                           resmon_interval=args.resmon_interval)
    except (FileNotFoundError, IOError, ValueError) as exc:
        print(file=sys.stderr)  # terminate any in-progress \r line
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(file=sys.stderr)  # newline after the progress line
    _print_summary(result)
    return 0


def _print_geometry_echo(video_path: str, cfg: CounterConfig) -> None:
    """Echo the effective processing geometry so mis-scaled bands are caught early."""
    from bubble_counter.geometry import line_center_px
    from bubble_counter.video_io import VideoSource

    with VideoSource(video_path, cfg.roi, cfg.scale) as source:
        out_w, out_h = source.out_size
    roi_len = out_h if cfg.line.axis == "h" else out_w
    center = line_center_px(roi_len, cfg.line.pos)
    axis_label = "y" if cfg.line.axis == "h" else "x"
    sys.stderr.write(
        f"처리 크기: {out_w}x{out_h} (scale {cfg.scale:.2f}) | "
        f"계수선 {axis_label}={center} (pos {cfg.line.pos:.2f}, "
        f"밴드 {cfg.line.band_px}px)\n"
    )


def _print_progress(event) -> None:
    if event.total_frames > 0:
        total = str(event.total_frames)
        pct = f"{100.0 * event.frame_idx / event.total_frames:5.1f}%"
    else:
        total = "?"
        pct = "   ?"
    eta = f"{event.eta_s:.0f}초" if event.eta_s is not None else "?"
    suffix = ""
    if getattr(event, "cpu_pct", None) is not None or getattr(event, "rss_mb", None) is not None:
        from bubble_counter.resmon import Snapshot, format_line
        snap = Snapshot(event.cpu_pct, event.cpu_cores, event.rss_mb,
                        event.sys_ram_pct, event.gpu_pct)
        suffix = " | " + format_line(snap, os.cpu_count() or 1)
    sys.stderr.write(
        f"\r프레임 {event.frame_idx}/{total} ({pct}) | "
        f"{event.processing_fps:7.1f} fps | ETA {eta} | 카운트 {event.count}{suffix}   "
    )
    sys.stderr.flush()


def _print_summary(result) -> None:
    summary = result.summary
    print("=== 기포 계수 완료 ===")
    if summary.get("cancelled"):
        print("(중지 요청으로 중단됨 — 부분 결과)")
    print(f"총 카운트: {result.total} 개")
    print(f"처리 프레임: {summary['processed_frames']} (계수: {summary['counted_frames']})")
    print(f"처리 속도: {summary['processing_fps']:.1f} fps")
    print(f"평균 발생률: {summary['mean_rate_per_s']:.3f} 개/초")
    print(f"산출물 폴더: {result.out_dir}")
    print("  - summary.json, counts.csv, marks.csv")
    if summary["config"]["annotate"]:
        print("  - annotated.avi")
    if summary["config"]["save_rhythm"]:
        print("  - rhythm_*.png")


# --------------------------------------------------------------------------
# synth
# --------------------------------------------------------------------------


def _cmd_synth(args: argparse.Namespace) -> int:
    from bubble_counter.synth import generate

    cfg = SynthConfig(
        width=args.width, height=args.height, fps=args.fps,
        duration_s=args.duration, rate_per_s=args.rate,
        speed_min=args.speed_min, speed_max=args.speed_max,
        radius_min=args.radius_min, radius_max=args.radius_max,
        brightness=args.brightness, alpha=args.alpha, noise_sigma=args.noise,
        lead_in_s=args.lead_in, line_pos=args.line_pos, seed=args.seed,
    )
    out = Path(args.out)
    gt_path = Path(str(out) + ".gt.json")
    try:
        gt = generate(cfg, out, gt_path)
    except (IOError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(f"합성 영상 생성 완료: {out}")
    print(f"정답(GT): {gt_path}")
    print(f"계수선 통과 총합(total_crossings): {gt['total_crossings']}")
    return 0


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------


def _cmd_bench(args: argparse.Namespace) -> int:
    from bubble_counter.pipeline import run_count
    from bubble_counter.video_io import VideoSource, resolve_fps

    try:
        roi = RoiSpec(*args.roi) if args.roi is not None else RoiSpec()
        line = LineSpec(axis=args.line_axis, pos=args.line_pos, band_px=args.band)
        # config를 앞에서 구성 → fps_override 등을 resolve_fps/n_bench 전에 검증
        # (count와 동일한 조기 fail-fast; max_frames는 n_bench 확정 후 replace로 채움)
        cfg = CounterConfig(roi=roi, line=line, scale=args.scale,
                            var_threshold=args.var_threshold, fps_override=args.fps)
    except ValueError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 1

    # --seconds도 fps와 동일 버그 클래스: 비유한/비양수면 아래 int(seconds*fps)에서
    # OverflowError/ValueError로 크래시하므로 영상 열기 전에 조기 거부.
    if not (math.isfinite(args.seconds) and args.seconds > 0):
        print(f"설정 오류: --seconds는 0보다 큰 유한수여야 합니다: {args.seconds}",
              file=sys.stderr)
        return 1

    try:
        source = VideoSource(args.video, roi, args.scale)
        fps = resolve_fps(source.info, args.fps)
    except (FileNotFoundError, IOError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    n_bench = max(1, int(args.seconds * fps))
    total_frames = source.info.frame_count

    # (1) decode-only pass
    decode_start = time.perf_counter()
    decoded = 0
    with source:
        for _frame in source.frames():
            decoded += 1
            if decoded >= n_bench:
                break
    decode_elapsed = time.perf_counter() - decode_start
    decode_fps = decoded / decode_elapsed if decode_elapsed > 0 else 0.0

    # (2) fast pipeline pass (temp out_dir, removed afterwards)
    cfg = replace(cfg, max_frames=n_bench)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bubble_bench_"))
    try:
        result = run_count(args.video, tmp_dir, cfg, resmon_interval=0)
    except (FileNotFoundError, IOError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    pipe_fps = result.summary["processing_fps"]

    _print_bench(args.seconds, decoded, decode_fps, result.summary["processed_frames"],
                 pipe_fps, total_frames, fps)
    return 0


def _print_bench(seconds: float, decoded: int, decode_fps: float, processed: int,
                 pipe_fps: float, total_frames: int, fps: float) -> None:
    print(f"=== 벤치마크 ({seconds:g}초 분량) ===")
    print(f"디코드 전용:      {decode_fps:8.1f} fps  ({decoded} 프레임)")
    print(f"fast 파이프라인:  {pipe_fps:8.1f} fps  ({processed} 프레임)")
    print()
    if total_frames > 0:
        video_s = total_frames / fps if fps > 0 else 0.0
        print(f"영상 전체 예상 (총 {total_frames} 프레임 / {video_s:.1f}초):")
        if decode_fps > 0:
            print(f"  디코드 전용 예상 소요:    {total_frames / decode_fps:7.1f} 초")
        if pipe_fps > 0:
            print(f"  fast 파이프라인 예상 소요: {total_frames / pipe_fps:7.1f} 초")
    else:
        print("영상 전체 프레임 수를 알 수 없어 전체 예상 시간을 계산할 수 없습니다.")


# --------------------------------------------------------------------------
# multiway
# --------------------------------------------------------------------------

_RECORDING_RE = re.compile(r"^(\d+)x(\d+)@(\d+(?:\.\d+)?):(\w+)$")
_BUBBLE_RE = re.compile(r"^(circle|ellipse):(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?$")


class _CliError(Exception):
    """CLI 조기 종료용 — 종료 코드와 한국어 메시지를 함께 나른다."""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code
        self.msg = msg


def _parse_recording(s: str) -> RecordingSpec:
    m = _RECORDING_RE.match(s.strip())
    if not m:
        raise _CliError(3, f"--recording 형식이 올바르지 않습니다 (WxH@fps:mode): {s!r}")
    w, h, fps, mode = m.groups()
    return RecordingSpec(int(w), int(h), float(fps), mode)   # mode 검증은 RecordingSpec


def _parse_bubble(s: str, count_half: bool) -> BubbleSizeSpec:
    m = _BUBBLE_RE.match(s.strip())
    if not m:
        raise _CliError(3, f"--bubble 형식이 올바르지 않습니다 (shape:major[xminor]): {s!r}")
    shape, major, minor = m.group(1), float(m.group(2)), m.group(3)
    minor_px = float(minor) if minor is not None else major
    return BubbleSizeSpec(shape, major, minor_px, count_half_bubbles=count_half)


def _load_points(path: str) -> tuple:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise _CliError(3, f"--points 파일을 읽을 수 없습니다: {path} ({exc})")
    if not isinstance(raw, list):
        raise _CliError(3, f"--points는 PointSpec 객체의 JSON 배열이어야 합니다: {path}")
    try:
        return tuple(PointSpec(**d) for d in raw)
    except TypeError as exc:
        raise _CliError(3, f"--points 항목 필드가 잘못되었습니다: {exc}")
    # PointSpec 값 범위 ValueError는 전파 → _cmd_multiway에서 설정 오류 3으로 매핑


def _load_profile_config(name: str) -> "MultiwayConfig | None":
    """프로파일에서 완전 config 로드(주입 가능한 seam). 없으면 None. 테스트가 monkeypatch."""
    from bubble_counter.multiway.profiles import ProfileStore
    try:
        return ProfileStore().load(name)
    except KeyError:
        return None


def _build_multiway_config(args: argparse.Namespace) -> MultiwayConfig:
    has_profile = bool(args.profile)
    has_direct = bool(args.recording) or bool(args.points)
    if has_profile and has_direct:
        raise _CliError(3, "--profile과 --recording/--points는 동시에 쓸 수 없습니다")
    if has_profile:
        config = _load_profile_config(args.profile)
        if config is None:
            raise _CliError(3, f"프로파일을 찾을 수 없습니다: {args.profile!r}")
        if args.bubble:      # 선택적 override(측정 세션마다 기포 크기 재지정 가능)
            # count_half_bubbles가 명시(--count-half-bubbles/--no-count-half-bubbles)되지
            # 않았으면(None) 프로파일에 저장된 값을 유지한다 — 인터뷰 Q3 확정 토글을 --bubble
            # override의 부작용으로 침묵 리셋하지 않기 위함.
            count_half = (args.count_half_bubbles if args.count_half_bubbles is not None
                         else config.bubble.count_half_bubbles)
            config = replace(config, bubble=_parse_bubble(args.bubble, count_half))
        return config
    if not (args.recording and args.points):
        raise _CliError(3, "--profile 또는 (--recording + --points)가 필요합니다")
    if not args.bubble:
        raise _CliError(3, "--bubble이 필요합니다 (예: circle:50, ellipse:60x45)")
    count_half = bool(args.count_half_bubbles)   # 직접 경로: 미지정(None)이면 False
    bubble = _parse_bubble(args.bubble, count_half)
    return MultiwayConfig(recording=_parse_recording(args.recording),
                          points=_load_points(args.points), bubble=bubble)


def _resolve_videos(args: argparse.Namespace) -> list:
    if not args.videos:
        raise _CliError(3, "영상 파일을 하나 이상 지정해야 합니다")
    missing = [v for v in args.videos if not Path(v).exists()]
    if missing:
        raise _CliError(3, f"영상 파일을 찾을 수 없습니다: {missing[0]}")
    return list(args.videos)


def _run_multiway(config: MultiwayConfig, videos: list, phase_name: str,
                  out_root: str) -> int:
    """runner(트랙 I) 위임. PreflightBlocked→2, 파일/설정 모순→3, 정상→0."""
    from bubble_counter.multiway.runner import PreflightBlocked, run_phase
    try:
        run_phase(videos, config, phase_name, out_root)
    except PreflightBlocked as exc:
        print(f"프리플라이트 차단: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, IOError) as exc:
        # 실행 중 발생하는 파일/설정 모순(예: 영상을 열 수 없음, fps 정보 없음,
        # 배경 추정 샘플 부족)은 _cmd_multiway의 설정 오류(3)와 같은 계열로 취급.
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 3
    return 0


def _cmd_multiway(args: argparse.Namespace) -> int:
    if not args.phase_name:
        print("오류: --phase-name이 필요합니다", file=sys.stderr)
        return 3
    try:
        config = _build_multiway_config(args)
        videos = _resolve_videos(args)
    except _CliError as exc:
        print(f"오류: {exc.msg}", file=sys.stderr)
        return exc.code
    except (ValueError, TypeError) as exc:
        # ValueError: MultiwayConfig/PointSpec 값 위반. TypeError: 손상된 프로파일이
        # profile_to_config(**layout["bubble"] 등)에서 필드 불일치로 던짐 — 둘 다 설정 모순.
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 3
    return _run_multiway(config, videos, args.phase_name, args.out)


def _add_multiway_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("multiway", help="멀티웨이(1~9 way) 기포 계수 — 챕터=영상, 페이즈=묶음")
    p.add_argument("videos", nargs="*", help="입력 영상(챕터 순서 = 인자 순서)")
    p.add_argument("--profile", default=None, help="저장된 ROI 프로파일 이름")
    p.add_argument("--recording", default=None,
                   help="녹화 설정 WxH@fps:mode (프로파일 대신 직접 지정)")
    p.add_argument("--points", default=None,
                   help="PointSpec 객체의 JSON 배열 파일(프로파일 대신 직접 지정)")
    p.add_argument("--bubble", default=None,
                   help="기포 크기 shape:major[xminor] (예: circle:50, ellipse:60x45)")
    p.add_argument("--count-half-bubbles", action=argparse.BooleanOptionalAction, default=None,
                   help="쪼개진 반쪽 기포도 각각 1개로 계수 (미지정 시: 프로파일 override면 "
                        "프로파일 값 유지, 직접 지정 경로면 False)")
    p.add_argument("--phase-name", default=None, help="페이즈 이름(출력 폴더·요약에 사용)")
    p.add_argument("-o", "--out", default=".", help="출력 루트 폴더(기본: 현재 폴더)")
    p.set_defaults(func=_cmd_multiway)


# --------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------


def _cmd_gui(args: argparse.Namespace) -> int:
    # lazy: tkinter may be absent; bootstrap owns error.log + dialog
    from bubble_counter.bootstrap import run_gui

    return run_gui()


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------


def _add_count_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("count", help="영상에서 기포를 계수합니다.")
    p.add_argument("video", help="입력 영상 파일 경로")
    p.add_argument("-o", "--out", required=True, help="산출물 출력 폴더")
    p.add_argument("--roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   default=None, help="관심영역 (프레임 대비 비율 0~1), 기본 전체")
    p.add_argument("--line-axis", choices=("h", "v"), default=_L.axis,
                   help="계수선 방향: h(수평)/v(수직)")
    p.add_argument("--line-pos", type=float, default=_L.pos, help="계수선 위치 (0~1)")
    p.add_argument("--band", type=int, default=_L.band_px,
                   help="밴드 폭(px, 처리 스케일). 빠른 기포일수록 크게")
    p.add_argument("--scale", type=float, default=_C.scale, help="다운스케일 배율 (0~1)")
    p.add_argument("--bucket", type=float, default=_C.bucket_seconds,
                   help="시간대별 카운트 버킷 크기(초)")
    p.add_argument("--min-width", type=int, default=_C.min_width_px, help="마크 최소 폭(px)")
    p.add_argument("--min-area", type=int, default=_C.min_area_px, help="마크 최소 면적(px)")
    p.add_argument("--merge-gap", type=int, default=_C.merge_gap_frames,
                   help="시간축 병합 허용 간격(프레임)")
    p.add_argument("--row-close", type=int, default=_C.row_close_px,
                   help="밴드 행 1D closing 크기(px)")
    p.add_argument("--var-threshold", type=float, default=_C.var_threshold,
                   help="MOG2 varThreshold")
    p.add_argument("--history", type=int, default=_C.history, help="MOG2 history")
    p.add_argument("--warmup", type=int, default=_C.warmup_frames,
                   help="배경 학습용 워밍업 프레임 수")
    p.add_argument("--fps", type=float, default=None,
                   help="fps 강제 지정 (메타데이터가 없거나 잘못된 경우)")
    p.add_argument("--annotate", action="store_true", help="주석 영상(annotated.avi) 저장")
    p.add_argument("--save-rhythm", action="store_true", help="rhythm 스트립 PNG 저장")
    p.add_argument("--max-frames", type=int, default=None, help="최대 처리 프레임 수")
    p.add_argument("--resmon-interval", type=float, default=0.5,
                   help="자원 샘플링 주기(초). 0이면 끔")
    p.set_defaults(func=_cmd_count)


def _add_synth_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("synth", help="정답이 알려진 합성 기포 영상을 생성합니다.")
    p.add_argument("out", help="출력 영상 경로 (GT는 <out>.gt.json)")
    p.add_argument("--width", type=int, default=_S.width)
    p.add_argument("--height", type=int, default=_S.height)
    p.add_argument("--fps", type=float, default=_S.fps)
    p.add_argument("--duration", type=float, default=_S.duration_s, help="영상 길이(초)")
    p.add_argument("--rate", type=float, default=_S.rate_per_s, help="초당 기포 생성 수")
    p.add_argument("--speed-min", type=float, default=_S.speed_min)
    p.add_argument("--speed-max", type=float, default=_S.speed_max)
    p.add_argument("--radius-min", type=int, default=_S.radius_min)
    p.add_argument("--radius-max", type=int, default=_S.radius_max)
    p.add_argument("--brightness", type=float, default=_S.brightness)
    p.add_argument("--alpha", type=float, default=_S.alpha)
    p.add_argument("--noise", type=float, default=_S.noise_sigma, help="프레임별 가우시안 노이즈 sigma")
    p.add_argument("--lead-in", type=float, default=_S.lead_in_s, help="기포 없는 도입부(초)")
    p.add_argument("--line-pos", type=float, default=_S.line_pos, help="계수선 위치(높이 비율)")
    p.add_argument("--seed", type=int, default=_S.seed)
    p.set_defaults(func=_cmd_synth)


def _add_bench_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("bench", help="디코드 전용 대 fast 파이프라인 처리 속도를 측정합니다.")
    p.add_argument("video", help="입력 영상 파일 경로")
    p.add_argument("--seconds", type=float, default=30.0, help="측정에 사용할 영상 분량(초)")
    p.add_argument("--roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"), default=None)
    p.add_argument("--line-axis", choices=("h", "v"), default=_L.axis)
    p.add_argument("--line-pos", type=float, default=_L.pos)
    p.add_argument("--band", type=int, default=_L.band_px)
    p.add_argument("--scale", type=float, default=_C.scale)
    p.add_argument("--var-threshold", type=float, default=_C.var_threshold)
    p.add_argument("--fps", type=float, default=None, help="fps 강제 지정")
    p.set_defaults(func=_cmd_bench)


def _add_gui_parser(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("gui", help="tkinter GUI를 실행합니다.")
    p.set_defaults(func=_cmd_gui)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bubble-counter",
        description="기포 계수 툴 — 계수선을 통과하는 기포를 셉니다 (프레임 스키핑 없음).",
    )
    sub = parser.add_subparsers(dest="command")
    _add_count_parser(sub)
    _add_synth_parser(sub)
    _add_bench_parser(sub)
    _add_multiway_parser(sub)
    _add_gui_parser(sub)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 1
    return args.func(args)
