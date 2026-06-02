#!/usr/bin/env bash
# nascence — double-click launcher for macOS.
#
# In Finder, double-click this file ("nascence.command") to set up (first time
# only) and launch the app — no Terminal commands needed.
#
# First time only: macOS may say it "cannot verify the developer". If so,
# right-click this file -> Open -> Open. You only have to do that once.

cd "$(dirname "$0")"

# Auto-setup on first launch, then run. Keep the window open on error so you
# can read any message.
if ! bash run.sh; then
  echo
  echo "Something went wrong above. Please copy the message and send it over."
  echo "Press Return to close this window."
  read -r _
fi
