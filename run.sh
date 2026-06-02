#!/usr/bin/env bash
# nascence — start the app.  In Terminal, run:  bash run.sh
# (Run  bash setup.sh  once first.)

set -e
cd "$(dirname "$0")"

if [ ! -x "venv/bin/python" ]; then
  echo "It looks like setup hasn't run yet."
  echo "Please run:  bash setup.sh"
  exit 1
fi

exec ./venv/bin/python -m nascence
