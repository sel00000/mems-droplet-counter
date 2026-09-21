"""실행 중 자원 사용률(CPU/RAM/GPU) 측정 — 표준 라이브러리만.

측정(make_reader), 누적/표시(ResourceSampler, format_line)를 분리한다.
run_count 단일 스레드에서만 sample()이 호출되므로 락은 필요 없다.

- CPU: time.process_time()/perf_counter() 델타 (양 OS 동일 코드, 첫 호출 baseline→None).
- RAM: Linux /proc · Windows ctypes (플랫폼 분기).
- GPU: 시작 시 1회 탐지된 백엔드(nvidia|None)에 따라 값 또는 None (best-effort).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Snapshot:
    cpu_pct: float | None       # 정규화 0~100 (표시·요약 기본)
    cpu_cores: float | None     # 코어 환산 (raw%/100). 1.0 = 한 코어치
    rss_mb: float | None        # 프로세스 RSS (MB)
    sys_ram_pct: float | None   # 시스템 RAM load % (0~100)
    gpu_pct: float | None = None  # 장치 전체 GPU% (미측정=None)


ReadFn = Callable[[], "Snapshot"]

_METRICS = ("cpu_pct", "cpu_cores", "rss_mb", "sys_ram_pct", "gpu_pct")


class ResourceSampler:
    """인라인 샘플러. run_count 루프(단일 스레드)에서만 호출되므로 락 불필요."""

    def __init__(self, read_fn: ReadFn | None = None, *, enabled: bool = True) -> None:
        self._read = read_fn
        self._enabled = bool(enabled) and read_fn is not None
        self._latest: Snapshot | None = None
        self._peak: dict[str, float] = {}
        self._sum: dict[str, float] = {}
        self._cnt: dict[str, int] = {}
        self._samples = 0
        self._ncpu = os.cpu_count() or 1
        self._gpu_seen = False

    def sample(self) -> Snapshot | None:
        """read_fn 1회 호출 → 예외는 삼켜 None → 유효 지표만 peak/sum/n 누적 → 최신 Snapshot 반환.

        비활성(enabled=False)이면 항상 None. 측정 실패가 count를 절대 중단시키지 않는다.
        """
        if not self._enabled:
            return None
        try:
            snap = self._read()  # type: ignore[misc]
        except Exception:
            return None
        if snap is None:
            return None
        self._latest = snap
        self._samples += 1
        for m in _METRICS:
            v = getattr(snap, m)
            if v is None:
                continue
            if m == "gpu_pct":
                self._gpu_seen = True
            self._peak[m] = v if m not in self._peak else max(self._peak[m], v)
            self._sum[m] = self._sum.get(m, 0.0) + v
            self._cnt[m] = self._cnt.get(m, 0) + 1
        return snap

    def latest(self) -> Snapshot | None:
        return self._latest

    def _avg(self, m: str) -> float | None:
        n = self._cnt.get(m, 0)
        return (self._sum[m] / n) if n else None

    def result(self) -> dict:
        """summary["resources"] 블록. 샘플 0개면 값들 None, samples=0."""
        return {
            "cpu_pct_peak": self._peak.get("cpu_pct"),
            "cpu_pct_avg": self._avg("cpu_pct"),
            "cpu_cores_peak": self._peak.get("cpu_cores"),
            "cpu_cores_avg": self._avg("cpu_cores"),
            "cpu_count": self._ncpu,
            "cpu_pct_normalized": True,
            "proc_rss_mb_peak": self._peak.get("rss_mb"),
            "proc_rss_mb_avg": self._avg("rss_mb"),
            "sys_ram_pct_peak": self._peak.get("sys_ram_pct"),
            "sys_ram_pct_avg": self._avg("sys_ram_pct"),
            "gpu_available": self._gpu_seen,
            "gpu_backend": getattr(self._read, "backend", None) if self._read else None,
            "gpu_util_pct_peak": self._peak.get("gpu_pct"),
            "gpu_util_pct_avg": self._avg("gpu_pct"),
            "gpu_device_wide": True,
            "gpu_note": None if self._gpu_seen else "no NVIDIA GPU detected / tool is CPU-only",
            "samples": self._samples,
        }

    def close(self) -> None:
        """GPU 백엔드 자원 정리(있으면). 멱등. 기본 경로(Intel/CPU-only)에선 no-op."""
        closer = getattr(self._read, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def format_line(snap: Snapshot | None, ncpu: int) -> str:
    """진행 줄/라벨용 한국어 문자열.

    예: 'CPU 25% (~1.0/4코어) · RAM 210MB (sys 41%) · GPU 해당없음'.
    지표가 None이면 그 항목은 'N/A'/'해당없음'.
    """
    if snap is None:
        return ""
    parts = []
    if snap.cpu_pct is not None and snap.cpu_cores is not None:
        parts.append(f"CPU {snap.cpu_pct:.0f}% (~{snap.cpu_cores:.1f}/{ncpu}코어)")
    else:
        parts.append("CPU N/A")
    if snap.rss_mb is not None:
        sysr = f" (sys {snap.sys_ram_pct:.0f}%)" if snap.sys_ram_pct is not None else ""
        parts.append(f"RAM {snap.rss_mb:.0f}MB{sysr}")
    else:
        parts.append("RAM N/A")
    parts.append(f"GPU {snap.gpu_pct:.0f}%" if snap.gpu_pct is not None else "GPU 해당없음")
    return " · ".join(parts)


# --- RAM 리더 (플랫폼 분기) --------------------------------------------------

_PROC_STATM = "/proc/self/statm"
_PROC_MEMINFO = "/proc/meminfo"


def _rss_mb_linux() -> float | None:
    try:
        with open(_PROC_STATM) as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return None


def _sys_ram_pct_linux() -> float | None:
    try:
        total = avail = None
        with open(_PROC_MEMINFO) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail = float(line.split()[1])
                if total is not None and avail is not None:
                    break
        if not total:
            return None
        return 100.0 * (total - avail) / total
    except Exception:
        return None


def _win_ram_readers():
    """Windows: (rss_mb, sys_ram_pct) 리더 2개. WSL/Linux에선 호출 안 됨."""
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    class _MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    # argtypes 명시 필수 — 없으면 64-bit HANDLE이 기본 C int(32-bit)로 잘려 호출 실패→RAM N/A (스펙 §4.2)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL

    def rss_mb() -> float | None:
        try:
            c = _PMC(); c.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 1e6
        except Exception:
            pass
        return None

    def sys_ram_pct() -> float | None:
        try:
            m = _MEMSTAT(); m.dwLength = ctypes.sizeof(_MEMSTAT)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return float(m.dwMemoryLoad)
        except Exception:
            pass
        return None

    return rss_mb, sys_ram_pct


# --- GPU best-effort (nvidia 탐지 + throttled one-shot) ----------------------

_GPU_THROTTLE_S = 2.0


def _detect_gpu_backend() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=2.0, check=True)
        return "nvidia"
    except Exception:
        return None


def _make_gpu_read(backend: str | None, now):
    if backend != "nvidia":
        fn = lambda: None
        fn.close = lambda: None  # type: ignore[attr-defined]
        return fn
    state = {"last_t": None, "val": None}

    def read() -> float | None:
        t = now()
        if state["last_t"] is not None and (t - state["last_t"]) < _GPU_THROTTLE_S:
            return state["val"]
        state["last_t"] = t
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=True).stdout
            state["val"] = float(out.strip().splitlines()[0].split(",")[0])
        except Exception:
            state["val"] = None
        return state["val"]

    read.close = lambda: None  # type: ignore[attr-defined]  # one-shot=회수 불필요, 누수 0
    return read


# --- 측정 함수 팩토리 --------------------------------------------------------

def make_reader(*, now=None, cpu_time=None, gpu_backend="auto") -> ReadFn:
    """플랫폼·백엔드를 캡슐화한 측정 함수(read_fn) 생성.

    주입 가능한 now/cpu_time 시계로 결정론 테스트가 가능하다.
    반환 read_fn은 `.backend`·`.close` 속성을 가지며, 첫 호출은 CPU baseline(cpu None).
    """
    now = now or time.perf_counter
    cpu_time = cpu_time or time.process_time
    ncpu = os.cpu_count() or 1
    if sys.platform == "win32":
        rss_fn, sysram_fn = _win_ram_readers()
    else:
        rss_fn, sysram_fn = _rss_mb_linux, _sys_ram_pct_linux
    backend = _detect_gpu_backend() if gpu_backend == "auto" else gpu_backend
    gpu_read = _make_gpu_read(backend, now) if backend else (lambda: None)
    prev = {"wc": None, "ct": None}

    def read() -> Snapshot:
        wc, ct = now(), cpu_time()
        cpu_pct = cpu_cores = None
        if prev["wc"] is not None:
            dwc = wc - prev["wc"]
            if dwc > 0:
                raw = (ct - prev["ct"]) / dwc * 100.0
                cpu_pct, cpu_cores = raw / ncpu, raw / 100.0
        prev["wc"], prev["ct"] = wc, ct
        return Snapshot(cpu_pct=cpu_pct, cpu_cores=cpu_cores,
                        rss_mb=rss_fn(), sys_ram_pct=sysram_fn(), gpu_pct=gpu_read())

    read.backend = backend      # type: ignore[attr-defined]
    read.close = getattr(gpu_read, "close", lambda: None)  # type: ignore[attr-defined]
    return read
