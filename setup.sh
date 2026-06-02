#!/usr/bin/env bash
# nascence — one-time setup for macOS.
# Open the Terminal app, drag this project folder in, and run:  bash setup.sh
# It creates a private "venv" folder with everything the app needs. Safe to re-run.

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "  Setting up nascence ..."
echo "=============================================="

# 1. Make sure Python 3 is installed.
if ! command -v python3 >/dev/null 2>&1; then
  echo
  echo "  Python 3 is not installed."
  echo "  Please install it (the easy way): https://www.python.org/downloads/"
  echo "  Then run  bash setup.sh  again."
  exit 1
fi

PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "  Found Python $PYVER"

# Warn if running x86 Python under Rosetta on Apple Silicon (slow / wrong wheels).
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  PYARCH="$(python3 -c 'import platform; print(platform.machine())')"
  if [ "$PYARCH" != "arm64" ]; then
    echo "  WARNING: your Python is not native Apple-Silicon (arm64). This may be slow."
    echo "           Consider reinstalling Python from python.org."
  fi
fi

# 2. Create the virtual environment (skip if it already exists).
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment (venv) ..."
  python3 -m venv venv
fi

# 3. Install dependencies.
echo "  Installing libraries (this can take a few minutes the first time) ..."
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/python -m pip install -r requirements.txt

echo
echo "=============================================="
echo "  Setup complete!  Now run:  bash run.sh"
echo "=============================================="
