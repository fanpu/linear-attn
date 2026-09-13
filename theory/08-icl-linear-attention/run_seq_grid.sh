#!/usr/bin/env bash
# Train the build-on grid. Usage: run_seq_grid.sh "softmax linear" > logs/x.log
cd "$(dirname "$0")"
for k in $1; do
  for L in 1 2 4; do
    [ -f cache/seq_${k}_L${L}_w64_d10_N40_s0.0_seed0.pt ] || ../.venv/bin/python train_seq.py --kind $k --layers $L --sigma 0 --steps 8000
  done
done
for k in $1; do
  [ -f cache/seq_${k}_L2_w64_d10_N40_s0.5_seed0.pt ] || ../.venv/bin/python train_seq.py --kind $k --layers 2 --sigma 0.5 --steps 8000
done
