#!/usr/bin/env bash
# Run one of Claude's GPU jobs politely on the shared GB10.
#
#   setsid nohup autonomous/tools/gpu_run.sh <cmd...> > <log> 2>&1 < /dev/null &
#
# What it does, in order:
#   1. Takes an exclusive lock, so only one of my GPU jobs runs at a time (others queue).
#   2. Waits while autonomous/GPU_PAUSE exists (Fan Pu wants the GPU).
#   3. Waits until no other process is using the GPU (two jobs at once -> driver OOM on the GB10).
#   4. Runs the command. If it exits with code 75 (PAUSE_EXIT, see pause.py), the job checkpointed
#      because of a pause: go back to step 2 and rerun the same command, which resumes from checkpoint.
#
# Env: GPU_RUN_IGNORE_OTHERS=1 skips step 3 (e.g. for a tiny smoke test).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PAUSE_FILE="$ROOT/GPU_PAUSE"
export AUTONOMOUS_PAUSE_FILE="$PAUSE_FILE"
PAUSE_EXIT=75
ts() { date '+%F %T'; }

exec {lockfd}>"$HERE/.gpu.lock"
if ! flock -n "$lockfd"; then
  echo "[gpu_run $(ts)] another autonomous GPU job holds the lock; queueing" >&2
  flock "$lockfd"
fi

other_gpu_users() {
  [ "${GPU_RUN_IGNORE_OTHERS:-0}" = 1 ] && return 1
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^\s*$')
  [ -n "$apps" ] && { echo "$apps"; return 0; }
  return 1
}

while true; do
  waited=0
  while [ -e "$PAUSE_FILE" ] || other_gpu_users > /tmp/gpu_run_others.$$; do
    if (( waited % 300 == 0 )); then
      if [ -e "$PAUSE_FILE" ]; then echo "[gpu_run $(ts)] paused (GPU_PAUSE present)" >&2
      else echo "[gpu_run $(ts)] waiting for other GPU processes: $(tr '\n' ';' < /tmp/gpu_run_others.$$)" >&2; fi
    fi
    sleep 15; waited=$((waited + 15))
  done
  rm -f /tmp/gpu_run_others.$$
  echo "[gpu_run $(ts)] start: $*" >&2
  "$@" &
  child=$!
  echo "$child" > "$HERE/.running.pid"
  wait "$child"; rc=$?
  rm -f "$HERE/.running.pid"
  if [ "$rc" = "$PAUSE_EXIT" ]; then
    echo "[gpu_run $(ts)] job checkpointed for pause; will resume when GPU_PAUSE is removed" >&2
    continue
  fi
  echo "[gpu_run $(ts)] exit $rc: $*" >&2
  exit "$rc"
done
