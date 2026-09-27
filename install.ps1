# supercompress installer (Windows PowerShell 5.1+, run with right-click "Run with PowerShell"
# or:  powershell -ExecutionPolicy Bypass -File install.ps1)
#
# Creates a .venv, installs the dev/test dependency (pytest), and runs the
# full test suite to verify the install.
#
# supercompress is a plain source tree (no packaging / no pip install .):
# add the repo root to PYTHONPATH (or run from the repo root) and
# `import sc` / `import scdata` works out of the box.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ── 1. locate a Python 3.9+ launcher (py launcher preferred on Windows) ──
$Python = $null
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    # `py -3` hands back the newest installed 3.x; -3.9..-3.12 pin when needed.
    & py -3 --version 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = "py -3" }
}
if (-not $Python) {
    foreach ($c in @("python3.12","python3.11","python3.10","python3.9","python")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) {
            & $c -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { $Python = $c; break }
        }
    }
}
if (-not $Python) {
    Write-Error "No Python 3.9+ interpreter found. Install Python 3.9+ (https://www.python.org/) and re-run."
    exit 1
}
Write-Host "==> Using: $(& $Python --version 2>&1)"

# ── 2. create virtualenv ─────────────────────────────────────────────
$VenvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "==> Creating .venv"
    & $Python -m venv .venv
} else {
    Write-Host "==> .venv already present, reusing"
}

# ── 3. install dev deps (pytest) ────────────────────────────────────
Write-Host "==> Installing requirements (pytest)"
& $VenvPy -m pip install --upgrade pip *> $null
& $VenvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ── 4. verify: run the full test suite from the repo root ───────────
Write-Host "==> Running test suite (sc + scdata + bench import smoke)"
$env:PYTHONPATH = (Resolve-Path .).Path
Push-Location $PSScriptRoot
try {
    & $VenvPy -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest reported failures — install not verified" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Install complete. Quick start:"
Write-Host "  \$env:PYTHONPATH = \"<repo root>\"      # or run python from the repo root"
Write-Host "  .venv\Scripts\python.exe -c \"from sc import Segment, JsonlStore; print('supercompress import OK')\""
Write-Host ""
Write-Host "Run tests any time:  .venv\Scripts\python.exe -m pytest"
