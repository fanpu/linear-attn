#!/usr/bin/env bash
# Run a command while holding one of $GPU_SLOTS theory-project GPU slots (default 3).
# The GB10 is shared with the art/ agents (their own 8-slot pool), so keep this small.
#
#   _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/sweep.py
#
# Blocks until a slot is free, so launch long jobs with run_in_background.
set -u
SLOTS=${GPU_SLOTS:-3}
DIR="$(cd "$(dirname "$0")" && pwd)/.gpu_slots"
mkdir -p "$DIR"
while true; do
  for i in $(seq 1 "$SLOTS"); do
    exec {fd}>"$DIR/slot$i"
    if flock -n "$fd"; then
      echo "[gpu_run] slot $i acquired: $*" >&2
      "$@"
      exit $?
    fi
    exec {fd}>&-
  done
  sleep 10
done
