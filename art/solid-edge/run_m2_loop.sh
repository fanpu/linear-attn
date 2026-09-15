#!/usr/bin/env bash
# M2 remainder as <= ~18-minute GPU segments (controller rule: no queued job > ~20 min).
# Each segment is its own gpu1.sh job; vol_run.py exits 3 when it stopped at a chunk checkpoint with work left.
#   setsid nohup bash run_m2_loop.sh >> logs/m2.log 2>&1 < /dev/null &
cd /home/fzeng/ml/research/art/solid-edge
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
GPU1=/home/fzeng/ml/research/art/_shared/gpu1.sh
SEG=${SEG_SECONDS:-1080}
run_seg() {
  local name=$1; shift
  local n=0
  while :; do
    n=$((n + 1))
    echo "[loop $(date '+%F %T')] $name segment $n"
    "$GPU1" "$PY" vol_run.py --name "$name" "$@" --max_seconds "$SEG"
    rc=$?
    [ $rc -eq 3 ] && continue
    [ $rc -eq 0 ] && return 0
    echo "[loop $(date '+%F %T')] $name segment $n FAILED rc=$rc"; exit $rc
  done
}
run_seg B256 --grid json --grid_json cache/vol/B_window.json --res 256 --stages f32,shell,audit,final,plane
run_seg B256_sub64 --grid sub --parent B256 --res 64 --stages f32,shell,audit,final
echo ALL_DONE
