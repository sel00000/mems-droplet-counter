"""GUI 설정 영속화 + 밴드 비율 환산 — tkinter를 import하지 않는 순수 stdlib 모듈.

"시작"을 누른 설정을 홈 폴더의 JSON에 저장했다가 다음 실행 때 복원한다. 밴드
폭은 픽셀과 함께 "진행축 처리 길이 대비 비율(band_frac)"로 기억해, 해상도가
다른 영상에서 픽셀값을 자동 환산한다 (설계 스펙
docs/superpowers/specs/2026-07-04-gui-settings-autoband-design.md).

tkinter가 없는 환경(예: 이 WSL dev 머신)에서도 단위 테스트할 수 있도록
gui.py와 분리되어 있다 (gui_params.py와 같은 패턴). fps 강제 값은 스키마에
의도적으로 없다 — 영상마다 다른 값이라 이월되면 시간축이 조용히 틀어진다.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from bubble_counter.gui_params import _RANGES

DEFAULT_PATH = Path.home() / ".bubble_counter_gui.json"

_VERSION = 1

# 숫자 키 → (하한, 상한). Spinbox 범위는 gui_params._RANGES와 단일 원천으로
# 공유하고, 슬라이더 값(line_pos/scale)의 범위만 여기서 직접 정의한다.
_NUMERIC_KEYS = {
    "band_px": (_RANGES["band_px"][1], _RANGES["band_px"][2]),
    "bucket_seconds": (_RANGES["bucket_seconds"][1], _RANGES["bucket_seconds"][2]),
    "var_threshold": (_RANGES["var_threshold"][1], _RANGES["var_threshold"][2]),
    "merge_gap_frames": (_RANGES["merge_gap_frames"][1], _RANGES["merge_gap_frames"][2]),
    "line_pos": (0.0, 1.0),
    "scale": (0.1, 1.0),
}
_INT_KEYS = ("band_px", "merge_gap_frames")
_BOOL_KEYS = ("band_auto", "annotate", "save_rhythm")


def load_settings(path: "str | Path" = DEFAULT_PATH) -> dict:
    """저장된 설정을 읽어 sanitize 통과분만 반환.

    파일 없음/JSON 손상/최상위가 dict 아님 → {} (예외를 밖으로 내지 않는다 —
    설정 문제가 GUI 실행을 막으면 안 된다).
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return _sanitize(raw)


def save_settings(values: dict, path: "str | Path" = DEFAULT_PATH) -> bool:
    """설정을 JSON으로 저장. 실패(OSError)는 삼키고 False 반환 —
    저장 실패가 계수 시작을 막으면 안 된다.

    같은 폴더의 임시 파일에 먼저 쓰고 os.replace로 원자적으로 교체한다 —
    저장 도중 크래시가 대상 파일을 반쯤 쓰인 상태로 손상시키지 않게 한다.
    """
    payload = {"version": _VERSION, **values}
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
    return True


def _sanitize(raw: dict) -> dict:
    """알려진 키만, 타입·범위 검사를 통과한 값만 남긴다.

    범위 밖 값은 클램프하지 않고 버린다 — 그 항목만 GUI 기본값으로 돌아가는
    쪽이, 조용히 다른 값으로 바뀌는 것보다 예측 가능하다.
    """
    out: dict = {}
    if raw.get("axis") in ("h", "v"):
        out["axis"] = raw["axis"]
    for key, (lo, hi) in _NUMERIC_KEYS.items():
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not (lo <= value <= hi):
            continue
        if key in _INT_KEYS:
            if float(value).is_integer():
                out[key] = int(value)
        else:
            out[key] = float(value)
    for key in _BOOL_KEYS:
        if isinstance(raw.get(key), bool):
            out[key] = raw[key]
    frac = raw.get("band_frac")
    if frac is None and "band_frac" in raw:
        out["band_frac"] = None
    elif (isinstance(frac, (int, float)) and not isinstance(frac, bool)
          and 0 < frac <= 1):
        out["band_frac"] = float(frac)
    if isinstance(raw.get("last_video_dir"), str):
        out["last_video_dir"] = raw["last_video_dir"]
    return out


def _processing_len(axis_len: int, scale: float) -> int:
    # VideoSource.out_size와 동일한 최소 1픽셀 보장 규칙
    return max(1, round(axis_len * scale))


def band_px_for(frac: float, axis_len: int, scale: float) -> int:
    """비율 → 처리 스케일 밴드 px. 결과는 Spinbox 범위 [1, 512]로 클램프."""
    lo, hi = _NUMERIC_KEYS["band_px"]
    px = round(frac * _processing_len(axis_len, scale))
    return int(max(lo, min(hi, px)))


def frac_for(band_px: int, axis_len: int, scale: float) -> float:
    """처리 스케일 밴드 px → 비율 (band_px_for의 역방향).

    band_px가 처리 길이를 넘는 병리 케이스는 1.0으로 클램프해, 저장값이 항상
    (0, 1] 범위라 다음 로드의 sanitize가 드롭하지 않게 한다.
    """
    return min(1.0, band_px / _processing_len(axis_len, scale))
