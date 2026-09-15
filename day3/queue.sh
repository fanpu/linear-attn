#!/usr/bin/env bash
# Day 3 queue: six transformer baselines, sequential (one GPU; concurrent runs would share bandwidth
# and thermal state, which confounds tok/s). Idempotent: a run with summary.json is skipped, and a
# run with checkpoints resumes. Launch:  nohup ./queue.sh > queue.log 2>&1 &
set -u
DATA=${DATA:-data/fineweb_edu}
declare -A BUDGET=( [30M]=300000000 [60M]=600000000 [125M]=1200000000 )   # edit after Day 2 tok/s check
for size in 30M 60M 125M; do
  for seed in 0 1; do
    out=runs/${size}_s${seed}
    if [ -f "$out/summary.json" ]; then echo "skip $out (done)"; continue; fi
    echo ">> $(date)  $size seed $seed  budget ${BUDGET[$size]}"
    python train.py run --size $size --seed $seed --tokens ${BUDGET[$size]} --data $DATA --out $out --resume \
      > $out.log 2>&1 || echo "!! $out failed (rc=$?), continuing"
  done
done
echo "queue done $(date)"
