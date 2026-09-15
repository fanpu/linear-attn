#!/usr/bin/env bash
# Let Claude's GPU jobs run again (queued/paused jobs in gpu_run.sh restart within ~15 s).
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -f "$HERE/../GPU_PAUSE" && echo "GPU_PAUSE removed at $(date '+%F %T')"
