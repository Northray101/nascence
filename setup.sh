#!/usr/bin/env bash
# nascence — one-time setup for macOS.
# Open the Terminal app, drag this project folder in, and run:  bash setup.sh
# It creates a private "venv" folder with everything the app needs. Safe to re-run.

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "  Setting up nascence ..."
echo "=============================================="

# nascence needs a Python version that has ready-to-install packages.
# 3.10 - 3.12 all work. Newer versions (3.13/3.14) are too new: their packages
# would have to be compiled from source and usually fail. We look for a good
# one automatically.
SUPPORTED="3.12 3.11 3.10"

is_supported() {  # $1 = "3.12" etc.
  case "$1" in
    3.10 | 3.11 | 3.12) return 0 ;;
    *) return 1 ;;
  esac
}

ver_of() {  # echo the X.Y version of a python executable
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

# 1. Find a compatible Python interpreter.
PYBIN=""
for v in $SUPPORTED; do
  if command -v "python$v" >/dev/null 2>&1; then
    PYBIN="python$v"
    break
  fi
done
# Fall back to a plain `python3` only if it happens to be a supported version.
if [ -z "$PYBIN" ] && command -v python3 >/dev/null 2>&1; then
  if is_supported "$(ver_of python3)"; then
    PYBIN="python3"
  fi
fi

if [ -z "$PYBIN" ]; then
  FOUND="(none found)"
  command -v python3 >/dev/null 2>&1 && FOUND="$(ver_of python3)"
  echo
  echo "  nascence needs Python 3.10, 3.11, or 3.12."
  echo "  The Python on your Mac is: $FOUND  (too new or missing)."
  echo
  echo "  Please install Python 3.12:"
  echo "    1. Go to  https://www.python.org/downloads/macos/"
  echo "    2. Download the latest 'Python 3.12.x' macOS installer and run it."
  echo "    3. Then run  bash setup.sh  again."
  echo
  exit 1
fi

echo "  Using $PYBIN (Python $(ver_of "$PYBIN"))"

# Warn if running x86 Python under Rosetta on Apple Silicon (slow / wrong wheels).
if [ "$(uname -m)" = "arm64" ]; then
  PYARCH="$("$PYBIN" -c 'import platform; print(platform.machine())')"
  if [ "$PYARCH" != "arm64" ]; then
    echo "  WARNING: this Python is not native Apple-Silicon (arm64); it may be slow."
  fi
fi

# 2. Create the virtual environment. Recreate it if an old/incompatible one
#    (e.g. built with Python 3.14) is already there.
if [ -d "venv" ]; then
  if ! is_supported "$(ver_of ./venv/bin/python)"; then
    echo "  Existing venv uses an unsupported Python; rebuilding it ..."
    rm -rf venv
  fi
fi
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment (venv) ..."
  "$PYBIN" -m venv venv
fi

# 3. Install dependencies.
echo "  Installing libraries (this can take a few minutes the first time) ..."
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/python -m pip install -r requirements.txt

echo
echo "=============================================="
echo "  Setup complete!  Now run:  bash run.sh"
echo "=============================================="
