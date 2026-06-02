#!/usr/bin/env bash
# nascence — start the app.  In Terminal, run:  bash run.sh
# (If setup hasn't run yet, this runs it for you automatically.)

set -e
cd "$(dirname "$0")"

# Auto-setup on first run so there is only ever one command to remember.
if [ ! -x "venv/bin/python" ]; then
  echo "First run — setting things up (this happens once)..."
  bash setup.sh
fi

exec ./venv/bin/python -m nascence
