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
        # 注：$k 不能直接拼进字符串（"$k:" 在 PS 里是非法变量引用）→ 用连接构造正则
        if ($cfgText -notmatch "(?m)^[[:space:]]+" + [regex]::Escape($k) + ":") { $missing += $k }
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

# ── 6. 注册可发现布局（宿主实码发现根；嵌套 plugins\context_engine\<name> 是死布局）──
# 宿主实码（plugins/context_engine/__init__.py::_iter_engine_dirs +
# plugin_loader.iter_plugin_dirs）——context engine 只有两条合法发现根：
#   根 A（bundled）= <宿主仓>\plugins\context_engine\<name>\   （直接子目录）
#   根 B（user）   = $HERMES_HOME\plugins\<name>\             （直接子目录）
# iter_plugin_dirs 只列直接子目录、绝不 descend：
#   $HERMES_HOME\plugins\context_engine\<name>\（多套一层 context_engine\ 包裹）永远不可见
#   → dashboard plugins 界面下拉菜单无该选项 + 运行时静默回退内置 compressor
#   （2026-11-14 Windows host 事故根因：部署时误放了嵌套布局）。
#   另：根 A 若落在宿主 git 仓的 untracked 目录里，hermes update 的 autostash 会把它清掉
#   （残留空壳 → "loaded but no engine instance found"）→ 自动更新的宿主优先用根 B。
# 本步：对每个目标 home 保证 直接子目录 plugins\supercompress 可发现
#   （symlink → 无权限则 junction → 均不可用则复制交付面；已存在则绝不覆盖）。
$regTargets = @()
if ($AllProfiles) {
    $regTargets += "$HOME\.hermes"
    if (Test-Path "$HOME\.hermes\profiles") {
        Get-ChildItem -Path "$HOME\.hermes\profiles" -Directory | ForEach-Object { $regTargets += $_.FullName }
    }
} elseif ($env:HERMES_HOME) {
    $regTargets += $env:HERMES_HOME
} elseif ($HermesCfg) {
    $regTargets += (Resolve-Path (Split-Path $HermesCfg)).Path
}
$RepoRoot = (Resolve-Path .).Path
if (-not $regTargets -or $regTargets.Count -eq 0) {
    Write-Host "   WARN: 未定位目标 home（设 \$env:HERMES_HOME 或 --all-profiles），可发现布局未注册"
} else {
Write-Host ""
Write-Host "==> 第 6 步：注册可发现布局（user 根 直接子目录 plugins\supercompress）"
foreach ($H in $regTargets) {
    if (-not $H) { continue }
    $plug = Join-Path $H "plugins"
    $link = Join-Path $plug "supercompress"
    New-Item -ItemType Directory -Path $plug -Force | Out-Null
    if (Test-Path (Join-Path $plug "context_engine")) {
        Write-Host "   WARN: $plug\context_engine\ 是 inert 嵌套布局（宿主 loader 不 descend，引擎不会被发现）"
        Write-Host "        正确布局是 直接子目录 $link（下方处理）；嵌套目录可安全删除"
    }
    if (Test-Path $link -PathType Container) {
        $target = (Resolve-Path $link).Path
        if ($target -eq $RepoRoot) {
            Write-Host "   $H : 可发现布局已就位（$link -> 本 repo）"
        } else {
            Write-Host "   $H : $link 已存在（指向 $target），不覆盖；如需指向本 repo 请手工重建"
        }
        continue
    }
    # 死 link（Test-Path 为假但条目在）：重建
    $dead = Get-Item $link -ErrorAction SilentlyContinue
    if ($dead) { Remove-Item $link -Force }
    # 顺序：junction（目录 reparse，无需特权，本类 host 首选）→ SymbolicLink（需 Dev Mode/管理员）
    #      → 复制交付面。每级都做 reparse/落地验证——MSYS 侧 GNU ln -s 曾实测 rc=0 却静默
    #      降级为"整目录复制"（连 .git 一起），不验证 = 代码分叉。
    try {
        New-Item -ItemType Junction -Path $link -Target $RepoRoot | Out-Null
        $v = Get-Item -LiteralPath $link
        if ($v.LinkType -ne 'Junction' -or -not (Test-Path (Join-Path $link '__init__.py'))) {
            throw "junction 验证失败（LinkType=$($v.LinkType)）"
        }
        Write-Host "   $H : 已建 junction $link -> $RepoRoot（reparse 已验证，user 根直接子目录，可发现）"
    } catch {
        # junction 失败/验证失败：清掉残留再试下一级（symlink / 复制）
        if (Test-Path $link) {
            $cur = Get-Item -LiteralPath $link
            if ($cur.LinkType) { Remove-Item -LiteralPath $link -Force }
            else { Remove-Item -LiteralPath $link -Recurse -Force }
        }
        try {
            New-Item -ItemType SymbolicLink -Path $link -Target $RepoRoot | Out-Null
            $v = Get-Item -LiteralPath $link
            if ($v.LinkType -ne 'SymbolicLink' -or -not (Test-Path (Join-Path $link '__init__.py'))) {
                throw "symlink 验证失败"
            }
            Write-Host "   $H : 已建 symlink $link -> $RepoRoot（reparse 已验证）"
        } catch {
            # 最后手段：复制交付面（壳 + sc 组件 + 数据层 + 清单；排除 .git/.venv/测试脚手架）
            $keep = @("__init__.py", "engine.py", "plugin.yaml", "scdata.py", "sc",
                      "config.default.yaml", "requirements.txt", "LICENSE", "LICENSE-COMMERCIAL.md", "README.md")
            if (Test-Path $link) { Remove-Item -LiteralPath $link -Recurse -Force }
            New-Item -ItemType Directory -Path $link | Out-Null
            foreach ($f in $keep) {
                if (Test-Path $f) { Copy-Item -Path $f -Destination $link -Recurse -Force }
            }
            Write-Host "   WARN $H : junction/symlink 均不可用，已复制交付面到 $link；后续代码升级需重新执行本步"
        }
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
Write-Host ""
Write-Host "发现面验证（宿主实码链，离线不重启；宿主仓根入 sys.path + HERMES_HOME=<目标 home>）:"
Write-Host "  python -c 'from plugins.context_engine import discover_context_engines; print(discover_context_engines())'"
Write-Host "  期望出现 ('supercompress', '<desc>', True)。注意：宿主若经 launcher 注入 venv generation"
Write-Host "  （如 pm-runtime），探针须把该 venv 的 site-packages 一并挂上 sys.path——"
Write-Host "  宿主 hermes_yaml 硬依赖 ruamel.yaml，裸解释器缺它会误报 engine unavailable"
