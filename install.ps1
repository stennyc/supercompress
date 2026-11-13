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

# ── 5. 配齐 config.yaml（supercompress 配置面；与 install.sh 第 5 步同语义）──
# 定位顺序：$env:HERMES_HOME > ~/.hermes/profiles/$env:HERMES_ACTIVE_PROFILE > ~/.hermes\config.yaml
# 缺失 → 写入 config.default.yaml 完整配置块；已有 → 不覆盖，仅报告缺键。
$HermesCfg = $null
if ($env:HERMES_HOME) {
    # profile 安装形态：HERMES_HOME 已设 = 数据根锚定（缺失也写这里，不 fallback）
    if (-not (Test-Path $env:HERMES_HOME)) { New-Item -ItemType Directory -Path $env:HERMES_HOME | Out-Null }
    $HermesCfg = Join-Path $env:HERMES_HOME "config.yaml"
} elseif ($env:HERMES_ACTIVE_PROFILE -and (Test-Path (Join-Path "$HOME\.hermes\profiles\$($env:HERMES_ACTIVE_PROFILE)" "config.yaml"))) {
    $HermesCfg = Join-Path "$HOME\.hermes\profiles\$($env:HERMES_ACTIVE_PROFILE)" "config.yaml"
} elseif (Test-Path "$HOME\.hermes\config.yaml") {
    $HermesCfg = "$HOME\.hermes\config.yaml"
} elseif (Test-Path "$HOME\.hermes") {
    $HermesCfg = "$HOME\.hermes\config.yaml"   # 不存在也写入（bundled 形态新建）
}

$DefaultBlock = Join-Path $PSScriptRoot "config.default.yaml"
# 键清单 = config.default.yaml 里 4 空格缩进的超压缩子键（排除顶层 context/engine）
$DefaultKeys = Select-String -Path $DefaultBlock -Pattern '^\s{4}[a-z_]+:' |
    ForEach-Object { ($_.Line -replace '^\s+([a-z_]+):.*','$1') } |
    Where-Object { $_ -ne "context" -and $_ -ne "engine" }

if (-not $HermesCfg) {
    Write-Host "WARN: 未定位 Hermes config.yaml（设 \$env:HERMES_HOME）"
} elseif (-not (Test-Path $HermesCfg)) {
    Write-Host "==> 写入完整默认配置块 → $HermesCfg"
    $block = Select-String -Path $DefaultBlock -Pattern '# ── 配置块开始|# ── 配置块结束'
    if ($block.Count -eq 2) {
        $lines = Get-Content $DefaultBlock
        $start = [int]$block[0].LineNumber + 1
        $end   = [int]$block[1].LineNumber
        # 区段内的注释行（# 开头非标记）保留为配置说明
        $lines[($start-1)..($end-1)] | Where-Object { $_ -notmatch '^#' -or $_ -match '配置块' } |
            Out-File -FilePath $HermesCfg -Encoding utf8
        Write-Host "    已写入（context.engine + compression.supercompress 全键默认值）"
    } else {
        throw "config.default.yaml 配置块标记缺失（# ── 配置块开始/结束）"
    }
} else {
    Write-Host "==> 检查现有配置 → $HermesCfg（不覆盖）"
    $cfgText = Get-Content $HermesCfg -Raw
    $missing = @()
    foreach ($k in $DefaultKeys) {
        if ($cfgText -notmatch "(?m)^[[:space:]]+$k:") { $missing += $k }
    }
    if ($missing.Count -gt 0) {
        Write-Host "    缺失键（当前走 engine 内置默认，如需显式配齐请并入 $DefaultBlock 配置块）:"
        $missing | ForEach-Object { Write-Host "      $_" }
    } else {
        Write-Host "    全部键已就位"
    }
}

Write-Host ""
Write-Host "Install complete. Quick start:"
Write-Host "  \$env:PYTHONPATH = `"<repo root>`"      # or run python from the repo root"
Write-Host "  .venv\Scripts\python.exe -c `"from sc import Segment, JsonlStore; print('supercompress import OK')`""
Write-Host ""
Write-Host "Run tests any time:  .venv\Scripts\python.exe -m pytest"
Write-Host ""
Write-Host "Config:  默认块见 config.default.yaml（缺失键 = 走 engine 内置默认，可整段省略）"
