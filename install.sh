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

# ── 5. 配齐 config.yaml（supercompress 配置面）──────────────────────
# 定位 Hermes config.yaml：
#   1) $HERMES_HOME 已设 → 该目录下（profile 安装形态）
#   2) 否则 ~/.hermes/profiles/<HERMES_ACTIVE_PROFILE>/config.yaml（存在才用）
#   3) 否则 ~/.hermes/config.yaml（bundled 安装形态）
# config.yaml 缺失 → 写入 config.default.yaml 的完整配置块（引擎全默认值）。
# config.yaml 已有 → 不覆盖，仅逐键报告缺失项（写操作 0，安全默认）。
HERMES_CFG=""
if [ -n "${HERMES_HOME:-}" ]; then
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
elif [ ! -f "$HERMES_CFG" ]; then
  echo "==> 写入完整默认配置块 → $HERMES_CFG"
  # 取「配置块」区段（标记行之间、标记行本身除外），避免把本文件头注释写进 config
  awk '/^# ── 配置块开始/{f=1;next} /^# ── 配置块结束/{f=0} f' "$DEFAULT_BLOCK" > "$HERMES_CFG"
  echo "    已写入（context.engine + compression.supercompress 全键默认值）"
else
  echo "==> 检查现有配置 → $HERMES_CFG（不覆盖）"
  MISSING=""
  for k in $DEFAULT_KEYS; do
    grep -qE "^[[:space:]]+${k}:" "$HERMES_CFG" || MISSING="$MISSING $k"
  done
  if [ -n "$MISSING" ]; then
    echo "    缺失键（当前走 engine 内置默认，如需显式配齐请并入 $DEFAULT_BLOCK 配置块）:"
    # shellcheck disable=SC2086
    for k in $MISSING; do echo "      $k"; done
  else
    echo "    全部键已就位"
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
