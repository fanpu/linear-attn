#!/usr/bin/env bash
# Serial GPU queue for the 3D art campaign (art/ml-art-3d.md §12).
#
#   setsid nohup art/_shared/gpu1.sh <cmd...> > <log> 2>&1 < /dev/null &
#
# 1. Takes art/_shared/.gpu1.lock: one art GPU job at a time, the rest queue.
# 2. Takes autonomous/tools/.gpu.lock (append mode, never truncated), so an autonomous/ job and an art job
#    never overlap. If autonomous/GPU_PAUSE exists its jobs cannot start, so we stop waiting for that lock.
# 3. Waits until nvidia-smi lists no compute process (two jobs at once -> driver OOM on the GB10).
# 4. Runs the command and exits with its status. Jobs must checkpoint so a rerun resumes.
#
# Env: GPU1_IGNORE_OTHERS=1 skips step 3 (tests). GPU1_AUTONOMOUS_LOCK overrides the step-2 lock path.
#      GPU1_POLL_S sets the poll interval (default 15 s).
set -u
ROOT=/home/fzeng/ml/research
POLL=${GPU1_POLL_S:-15}
ts() { date '+%F %T'; }

exec {artfd}>>"$ROOT/art/_shared/.gpu1.lock"
if ! flock -n "$artfd"; then
  echo "[gpu1 $(ts)] queued behind another art GPU job" >&2
  flock "$artfd"
fi

ALOCK=${GPU1_AUTONOMOUS_LOCK:-$ROOT/autonomous/tools/.gpu.lock}
APAUSE="$(dirname "$(dirname "$ALOCK")")/GPU_PAUSE"
if [ -d "$(dirname "$ALOCK")" ]; then
  exec {autfd}>>"$ALOCK"
  n=0
  until flock -n "$autfd"; do
    if [ -e "$APAUSE" ]; then
      echo "[gpu1 $(ts)] autonomous/ is paused; not waiting for its lock" >&2
      break
    fi
    (( n % 20 == 0 )) && echo "[gpu1 $(ts)] waiting for an autonomous/ GPU job to finish" >&2
    sleep "$POLL"; n=$((n + 1))
  done
fi

other_gpu_users() {
  [ "${GPU1_IGNORE_OTHERS:-0}" = 1 ] && return 1
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^\s*$')
  [ -n "$apps" ] && { echo "$apps"; return 0; }
  return 1
}
n=0
while other_gpu_users > "/tmp/gpu1_others.$$"; do
  (( n % 20 == 0 )) && echo "[gpu1 $(ts)] waiting for other GPU processes: $(tr '\n' ';' < "/tmp/gpu1_others.$$")" >&2
  sleep "$POLL"; n=$((n + 1))
done
rm -f "/tmp/gpu1_others.$$"

echo "[gpu1 $(ts)] start: $*" >&2
"$@"
rc=$?
echo "[gpu1 $(ts)] exit $rc: $*" >&2
exit "$rc"
