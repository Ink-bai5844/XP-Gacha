#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
BUILD_PYTHON="${XP_GACHA_BUILD_PYTHON:-python3}"
if ! command -v "$BUILD_PYTHON" >/dev/null 2>&1; then
  echo "Build requires Python 3.11+. Set XP_GACHA_BUILD_PYTHON to its executable." >&2
  exit 1
fi
unset PYTHONHOME PYTHONPATH __PYVENV_LAUNCHER__
exec "$BUILD_PYTHON" -E -s -B "$SCRIPT_DIR/build_portable_release_macos.py" "$@"
