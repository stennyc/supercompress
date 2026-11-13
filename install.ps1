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

# ── 5. 配齐 config.yaml（supercompress 配置面；多 profile 走 -AllProfiles）──
# 定位顺序：$env:HERMES_HOME > ~/.hermes/profiles/$env:HERMES_ACTIVE_PROFILE > ~/.hermes\config.yaml
# 缺失 → 写入 config.default.yaml 完整配置块；已有 → 不覆盖，仅报告缺键。
#
# 多 profile / multiplex mode（宿主实码）：multiplex gateway 每个 profile turn 把
# HERMES_HOME 切到该 profile 目录，load_config() 读"当前 profile 自己的 config.yaml"。
# 各 profile 要走 supercompress 须各自配齐；-AllProfiles 扫 profiles\* 逐个处理（已有不覆盖）。
$AllProfiles = $false
foreach ($a in $args) { if ($a -eq "-AllProfiles" -or $a -eq "--all-profiles") { $AllProfiles = $true } }

function Process-Config {
    param([string]$Cfg)
    if (-not (Test-Path $Cfg)) {
        Write-Host "==> 写入完整默认配置块 → $Cfg"
        $block = Select-String -Path $DefaultBlock -Pattern '# ── 配置块开始|# ── 配置块结束'
        if ($block.Count -eq 2) {
            $lines = Get-Content $DefaultBlock
            $start = [int]$block[0].LineNumber + 1
            $end   = [int]$block[1].LineNumber
            $lines[($start-1)..($end-1)] | Where-Object { $_ -notmatch '^#' -or $_ -match '配置块' } |
                Out-File -FilePath $Cfg -Encoding utf8
            Write-Host "    已写入（context.engine + compression.supercompress 全键默认值）"
        } else {
            throw "config.default.yaml 配置块标记缺失（# ── 配置块开始/结束）"
        }
        return 2
    }
    Write-Host "==> 检查现有配置 → $Cfg（不覆盖）"
    $cfgText = Get-Content $Cfg -Raw
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
    if ($missing.Count -gt 0) { return 1 } else { return 0 }
}

$HermesCfg = $null
# --all-profiles：强制从 $HOME\.hermes 数据根起步（不读 HERMES_HOME——gateway 运行时
# 注入的 per-profile env 会带偏定位；多 profile 面要的是"所有 profile"而非当前这一个）
if ($AllProfiles) {
    if ((Test-Path "$HOME\.hermes\config.yaml") -or (Test-Path "$HOME\.hermes")) {
        $HermesCfg = "$HOME\.hermes\config.yaml"
    }
} elseif ($env:HERMES_HOME) {
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
} else {
    Process-Config -Cfg $HermesCfg

    # ── 多 profile 面：-AllProfiles 时扫 profiles\*（含 multiplex 各 profile）──
    if ($AllProfiles) {
        Write-Host ""
        Write-Host "==> -AllProfiles：扫描多 profile 目录"
        $RootDir = "$HOME\.hermes"   # 固定数据根（-AllProfiles 语义 = 全 profile，不随 HERMES_HOME 带偏）
        $ProfDir = Join-Path $RootDir "profiles"
        $n = 0
        if (Test-Path $ProfDir) {
            foreach ($pdir in Get-ChildItem -Path $ProfDir -Directory) {
                $n++
                Write-Host "── profile: $($pdir.Name)"
                if ((Resolve-Path $pdir).Path -eq (Resolve-Path (Split-Path $HermesCfg)).Path) {
                    Write-Host "   （= 上方已定位的份，跳过）"
                    continue
                }
                Process-Config -Cfg (Join-Path $pdir.FullName "config.yaml")
            }
            if ($n -eq 0) {
                Write-Host "   $ProfDir 下无 profile 子目录（单机单 profile 形态，无事可做）"
            } else {
                Write-Host "   共处理 $n 个 profile（缺失项见上方逐 profile 报告）"
            }
        } else {
            Write-Host "   未找到 $ProfDir（非多 profile 安装形态，无事可做）"
        }
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
