# -*- mode: python ; coding: utf-8 -*-
# Windows portable GUI — onedir + windowed.
# 빌드: packaging/build_windows.ps1 (Windows에서만).
# OpenCV DLL 누락 시 Windows 실빌드 후 datas/binaries 보강 (D-P2).

import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH)  # type: ignore[name-defined]
ROOT = SPEC_DIR.parent
ENTRY = SPEC_DIR / "entry_gui.py"

block_cipher = None

a = Analysis(  # type: ignore[name-defined]
    [str(ENTRY)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "bubble_counter",
        "bubble_counter.bootstrap",
        "bubble_counter.gui",
        "bubble_counter.gui_multiway",
        "bubble_counter.gui_live",
        "bubble_counter.gui_settings",
        "bubble_counter.gui_params",
        "bubble_counter.pipeline",
        "bubble_counter.rhythm",
        "bubble_counter.multiway",
        "bubble_counter.multiway.gui_review",
        "cv2",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "unittest",
        "tkinter.test",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # type: ignore[name-defined]

exe = EXE(  # type: ignore[name-defined]
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="기포계수툴",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # type: ignore[name-defined]
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="기포계수툴",
)
