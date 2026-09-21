# Windows only — portable one-folder build for 기포계수툴 GUI.
# Usage (from repo root OR packaging/):
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This script must run on Windows (not WSL/Linux). See README.md."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

Write-Host "== bubble-counter portable build =="
Write-Host "Root: $Root"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "python not found on PATH. Install Python 3.10+ (64-bit) from python.org."
}

Write-Host "Python: $(python --version)"
# Native stderr + $ErrorActionPreference=Stop 조합이 PS 5.1에서 스크립트를 죽임 (D-P4)
cmd /c "python -c ""import cv2, numpy; print('cv2', cv2.__version__, 'numpy', numpy.__version__)"""
if ($LASTEXITCODE -ne 0) {
    Write-Error "cv2/numpy import failed. Run: python -m pip install -r requirements.txt"
}

cmd /c "python -c ""import PyInstaller"" >nul 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller (user site)..."
    python -m pip install --user "pyinstaller>=6.0"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install pyinstaller failed."
    }
}

$distName = "기포계수툴"
$spec = Join-Path $ScriptDir "bubble_counter_gui.spec"
$distDir = Join-Path $Root "dist\$distName"

Write-Host "Running PyInstaller..."
python -m PyInstaller --noconfirm --clean $spec
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed (exit $LASTEXITCODE)."
}

if (-not (Test-Path (Join-Path $distDir "기포계수툴.exe"))) {
    Write-Error "Expected exe not found under $distDir"
}

# Colleague-facing docs
Copy-Item (Join-Path $ScriptDir "사용법_30초.txt") $distDir -Force
Copy-Item (Join-Path $ScriptDir "배포_5분_체크.txt") $distDir -Force

# Optional sample clip (may be absent — gitignores *.avi)  D-P3
$sampleSrc = Join-Path $Root "스모크_클립\스모크_480p.avi"
$sampleDir = Join-Path $distDir "샘플"
if (Test-Path $sampleSrc) {
    New-Item -ItemType Directory -Force -Path $sampleDir | Out-Null
    Copy-Item $sampleSrc $sampleDir -Force
    Write-Host "Sample clip copied to 샘플\"
} else {
    Write-Host "No 스모크_480p.avi found — skipping 샘플\ (ok)"
}

$stamp = Get-Date -Format "yyyyMMdd"
$zipPath = Join-Path $Root "dist\기포계수툴_v$stamp.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path $distDir -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "OK"
Write-Host "  Folder: $distDir"
Write-Host "  Zip:    $zipPath"
Write-Host "Next: copy folder to Desktop, run 배포_5분_체크.txt"
