#!/usr/bin/env bash
# supercompress installer (POSIX: macOS / Linux / WSL / MSYS git-bash)
#
# Installs a Python 3.9+ virtualenv, installs the dev/test dependency
# (pytest), and runs the test suite to verify the install.
#
# Usage:
#   bash install.sh            # creates .venv, installs pytest, runs tests
#   python3 -m venv .venv && \
#     .venv/bin/pip install -r requirements.txt   # manual equivalent
#
# supercompress is a plain source tree (no packaging / no pip install .):
# add the repo root to PYTHONPATH (or run from the repo root) and
# `import sc` / `import scdata` works out of the box.

set -euo pipefail

cd "$(dirname "$0")"

# ── 1. locate a Python 3.9+ interpreter ─────────────────────────────
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for c in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        PYTHON="$c"; break
      fi
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "ERROR: no Python 3.9+ interpreter found (tried python3.x / python)." >&2
  echo "       Install Python 3.9+ and re-run, or set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "==> Using $($PYTHON --version 2>&1) at $(command -v "$PYTHON")"

# ── 2. create virtualenv ────────────────────────────────────────────
if [ ! -x ".venv/bin/python" ]; then
  echo "==> Creating .venv"
  "$PYTHON" -m venv .venv
else
  echo "==> .venv already present, reusing"
fi
VENV_PY=".venv/bin/python"

# ── 3. install dev deps (pytest) ────────────────────────────────────
echo "==> Installing requirements (pytest)"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PY" -m pip install -r requirements.txt

# ── 4. verify: run the full test suite from the repo root ──────────
echo "==> Running test suite (sc + scdata + bench import smoke)"
( export PYTHONPATH="$(pwd)"
  cd "$PWD" && "$VENV_PY" -m pytest -q )

# ── 5. 配齐 config.yaml（supercompress 配置面；多 profile 走 --all-profiles）──
# 定位 Hermes config.yaml：
#   1) $HERMES_HOME 已设 → 该目录下（profile 安装形态）
#   2) 否则 ~/.hermes/profiles/<HERMES_ACTIVE_PROFILE>/config.yaml（存在才用）
#   3) 否则 ~/.hermes/config.yaml（bundled 安装形态）
# config.yaml 缺失 → 写入 config.default.yaml 的完整配置块（引擎全默认值）。
# config.yaml 已有 → 不覆盖，仅逐键报告缺失项（写操作 0，安全默认）。
#
# 多 profile / multiplex mode 说明（宿主实码）：
#   multiplex gateway 每个 profile turn 会把 HERMES_HOME 切到该 profile 目录
#   （hermes_constants ContextVar 链），load_config() 读的是"当前 profile 自己的
#   config.yaml"。所以各 profile 若要走 supercompress，各自的 config.yaml 都要有
#   context.engine + compression.supercompress；数据面（supercompress_store/）
#   随 HERMES_HOME per-profile 落盘，天然隔离。一次 install.sh 只配定位到的那一份，
#   配齐全部 profile 用：bash install.sh --all-profiles
ALL_PROFILES=0
for a in "$@"; do case "$a" in --all-profiles) ALL_PROFILES=1;; esac; done

process_config() {   # $1 = config.yaml 路径；返回 0=全键就位 1=有缺失 2=新写入
  local cfg="$1"
  if [ ! -f "$cfg" ]; then
    echo "==> 写入完整默认配置块 → $cfg"
    # 取「配置块」区段（标记行之间、标记行本身除外），避免把本文件头注释写进 config
    awk '/^# ── 配置块开始/{f=1;next} /^# ── 配置块结束/{f=0} f' "$DEFAULT_BLOCK" > "$cfg"
    echo "    已写入（context.engine + compression.supercompress 全键默认值）"
    return 2
  fi
  echo "==> 检查现有配置 → $cfg（不覆盖）"
  local MISSING="" k
  for k in $DEFAULT_KEYS; do
    grep -qE "^[[:space:]]+${k}:" "$cfg" || MISSING="$MISSING $k"
  done
  if [ -n "$MISSING" ]; then
    echo "    缺失键（当前走 engine 内置默认，如需显式配齐请并入 $DEFAULT_BLOCK 配置块）:"
    # shellcheck disable=SC2086
    for k in $MISSING; do echo "      $k"; done
  else
    echo "    全部键已就位"
  fi
  [ -n "$MISSING" ] && return 1
  return 0
}

HERMES_CFG=""
# --all-profiles：强制从 $HOME/.hermes 数据根起步（不读 HERMES_HOME——gateway 运行时
# 注入的 per-profile env 会带偏定位；多 profile 面要的是"所有 profile"而非当前这一个）
if [ "$ALL_PROFILES" -eq 1 ]; then
  if [ -f "$HOME/.hermes/config.yaml" ]; then
    HERMES_CFG="$HOME/.hermes/config.yaml"
  elif [ -d "$HOME/.hermes" ]; then
    HERMES_CFG="$HOME/.hermes/config.yaml"   # 不存在也写入（bundled 形态新建）
  fi
elif [ -n "${HERMES_HOME:-}" ]; then
  # profile 安装形态：HERMES_HOME 已设 = 数据根锚定，config.yaml 在该目录下（缺失也写这里，不 fallback）
  mkdir -p "$HERMES_HOME" 2>/dev/null
  HERMES_CFG="$HERMES_HOME/config.yaml"
elif [ -n "${HERMES_ACTIVE_PROFILE:-}" ] && [ -f "$HOME/.hermes/profiles/$HERMES_ACTIVE_PROFILE/config.yaml" ]; then
  HERMES_CFG="$HOME/.hermes/profiles/$HERMES_ACTIVE_PROFILE/config.yaml"
elif [ -f "$HOME/.hermes/config.yaml" ]; then
  HERMES_CFG="$HOME/.hermes/config.yaml"
elif [ -d "$HOME/.hermes" ]; then
  HERMES_CFG="$HOME/.hermes/config.yaml"   # 不存在也写入（bundled 形态新建）
fi

DEFAULT_BLOCK="$PWD/config.default.yaml"
DEFAULT_KEYS=$(grep -E '^\s{4}[a-z_]+:' "$DEFAULT_BLOCK" | sed -E 's/^\s+([a-z_]+):.*/\1/' | grep -vE '^(context|engine)$' | tr '\n' ' ')

if [ -z "$HERMES_CFG" ]; then
  echo "WARN: 未定位 Hermes config.yaml（设 HERMES_HOME 或用 --hermes-home <dir>）"
else
  process_config "$HERMES_CFG" || true   # 缺键 = 报告语义（非致命），set -e 下不炸脚本

  # ── 多 profile 面：--all-profiles 时扫 ~/.hermes/profiles/*/（含 multiplex 各 profile）──
  # 宿主实码：multiplex gateway 每个 profile turn 读"自己的" config.yaml（HERMES_HOME
  # per-turn ContextVar 切换）；这里逐个配齐，已有的仍不覆盖。
  if [ "$ALL_PROFILES" -eq 1 ]; then
    echo
    echo "==> --all-profiles：扫描多 profile 目录"
    # 数据根固定 $HOME/.hermes（--all-profiles 语义 = 全 profile，不随 HERMES_HOME 带偏）
    ROOT_DIR="$HOME/.hermes"
    PROF_DIR="$ROOT_DIR/profiles"
    N=0
    if [ -d "$PROF_DIR" ]; then
      for pdir in "$PROF_DIR"/*/; do
        [ -d "$pdir" ] || continue
        p="$(basename "$pdir")"
        N=$((N+1))
        echo "── profile: $p"
        # 当前定位到的那份（HERMES_HOME 指向的 profile 自身）跳过，避免重复处理
        if [ "$(cd "$pdir" && pwd)" = "$(cd "$(dirname "$HERMES_CFG")" && pwd) 2>/dev/null" ]; then
          echo "   （= 上方已定位的份，跳过）"
          continue
        fi
        process_config "$pdir/config.yaml" || true   # 同上：报告语义，非致命
      done
      if [ "$N" -eq 0 ]; then
        echo "   $PROF_DIR 下无 profile 子目录（单机单 profile 形态，无事可做）"
      else
        echo "   共处理 $N 个 profile（缺失项见上方逐 profile 报告）"
      fi
    else
      echo "   未找到 $PROF_DIR（非多 profile 安装形态，无事可做）"
    fi
  fi
fi

echo
echo "Install complete. Quick start:"
echo "  export PYTHONPATH=\"$(pwd)\""
echo "  $VENV_PY -c 'from sc import Segment, JsonlStore; print(\"supercompress import OK\")'"
echo
echo "Run tests any time:  .venv/bin/python -m pytest"
echo
echo "Config:  默认块见 config.default.yaml（缺失键 = 走 engine 内置默认，可整段省略）"
