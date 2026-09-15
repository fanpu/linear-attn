#!/usr/bin/env bash
# Serial GPU queue for the 3D art campaign (art/ml-art-3d.md §12).
#
#   setsid nohup art/_shared/gpu1.sh <cmd...> > <log> 2>&1 < /dev/null &
#
# Two modes:
#   Share mode (default, 2026-09-15): art jobs may run while autonomous/ jobs are
#     running too. After the art-only lock, the only safeguard is waiting for
#     enough free memory (this GB10 has ~120 GB unified CPU+GPU memory; two
#     concurrent GPU processes once exhausted the driver's pool and both died
#     silently, so we keep a light guard rather than none).
#   Exclusive mode (GPU1_EXCLUSIVE=1): the original, stricter behaviour -- also
#     takes the autonomous/ lock so an autonomous/ job and an art job never
#     overlap, and waits until nvidia-smi shows no other compute process.
#
# Steps:
# 1. Takes GPU1_ART_LOCK (default art/_shared/.gpu1.lock): one art GPU job at a time, the rest queue.
# 2. [exclusive only] Takes GPU1_AUTONOMOUS_LOCK (default autonomous/tools/.gpu.lock, append mode,
#    never truncated), so an autonomous/ job and an art job never overlap. If autonomous/GPU_PAUSE
#    exists its jobs cannot start, so we stop waiting for that lock.
# 3. [exclusive only] Waits until nvidia-smi lists no compute process (two jobs at once -> driver OOM).
# 4. Waits until MemAvailable in /proc/meminfo (or GPU1_MEMINFO) is >= GPU1_MIN_FREE_GB.
# 5. Exports PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless the caller already set it.
# 6. Runs the command and exits with its status. Jobs must checkpoint so a rerun resumes.
#
# Env:
#   GPU1_EXCLUSIVE=1            use exclusive mode (steps 2-3) instead of the default share mode
#   GPU1_ART_LOCK=<path>        overrides the step-1 art lock path (default art/_shared/.gpu1.lock)
#   GPU1_AUTONOMOUS_LOCK=<path> overrides the step-2 lock path (exclusive mode only)
#   GPU1_IGNORE_OTHERS=1        skips step 3, the nvidia-smi wait (tests; exclusive mode only)
#   GPU1_MEMINFO=<path>         overrides /proc/meminfo for the step-4 memory wait (tests)
#   GPU1_MIN_FREE_GB=<n>        minimum MemAvailable required to proceed, in GB (default 30)
#   GPU1_POLL_S=<secs>          poll interval for all waits (default 15 s), logged every 20 polls
set -u
ROOT=/home/fzeng/ml/research
POLL=${GPU1_POLL_S:-15}
ts() { date '+%F %T'; }

ARTLOCK=${GPU1_ART_LOCK:-$ROOT/art/_shared/.gpu1.lock}
exec {artfd}>>"$ARTLOCK"
if ! flock -n "$artfd"; then
  echo "[gpu1 $(ts)] queued behind another art GPU job" >&2
  flock "$artfd"
fi

if [ "${GPU1_EXCLUSIVE:-0}" = 1 ]; then
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
else
  echo "[gpu1 $(ts)] share mode: may run alongside autonomous/ jobs (set GPU1_EXCLUSIVE=1 for the old serialised behaviour)" >&2
fi

MEMINFO=${GPU1_MEMINFO:-/proc/meminfo}
MIN_FREE_GB=${GPU1_MIN_FREE_GB:-30}
MIN_FREE_KB=$(( MIN_FREE_GB * 1024 * 1024 ))
n=0
while :; do
  avail_kb=$(awk '/^MemAvailable:/ { print $2 }' "$MEMINFO" 2>/dev/null)
  [ -n "$avail_kb" ] && [ "$avail_kb" -ge "$MIN_FREE_KB" ] && break
  (( n % 20 == 0 )) && echo "[gpu1 $(ts)] waiting for ${MIN_FREE_GB}G free memory (MemAvailable ${avail_kb:-?} kB)" >&2
  sleep "$POLL"; n=$((n + 1))
done

: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

echo "[gpu1 $(ts)] start: $*" >&2
"$@"
rc=$?
echo "[gpu1 $(ts)] exit $rc: $*" >&2
exit "$rc"
