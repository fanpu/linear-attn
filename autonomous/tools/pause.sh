#!/usr/bin/env bash
# Ask Claude's GPU jobs to checkpoint and get off the GPU. Waits (up to 3 min) until they have.
HERE="$(cd "$(dirname "$0")" && pwd)"
touch "$HERE/../GPU_PAUSE"
echo "GPU_PAUSE set at $(date '+%F %T')"
for _ in $(seq 36); do
  if [ ! -e "$HERE/.running.pid" ] || ! kill -0 "$(cat "$HERE/.running.pid" 2>/dev/null)" 2>/dev/null; then
    echo "no autonomous GPU job running"; exit 0
  fi
  sleep 5
done
echo "WARNING: job $(cat "$HERE/.running.pid") still running after 3 min; kill it with: kill $(cat "$HERE/.running.pid")"
exit 1
