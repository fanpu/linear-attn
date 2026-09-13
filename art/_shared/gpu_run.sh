#!/usr/bin/env bash
# Run a command while holding one of $GPU_SLOTS shared GPU slots (default 8).
# Many art projects share one GB10; this caps how many heavy GPU jobs run at once.
#
#   art/_shared/gpu_run.sh .venv/bin/python sweep.py --res 1024
#
# Blocks until a slot is free, so launch long jobs with run_in_background.
set -u
SLOTS=${GPU_SLOTS:-8}
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
