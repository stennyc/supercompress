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

echo
echo "Install complete. Quick start:"
echo "  export PYTHONPATH=\"$(pwd)\""
echo "  $VENV_PY -c 'from sc import Segment, JsonlStore; print(\"supercompress import OK\")'"
echo
echo "Run tests any time:  .venv/bin/python -m pytest"
