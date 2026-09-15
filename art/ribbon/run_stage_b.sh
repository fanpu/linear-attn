#!/usr/bin/env bash
# Stage B: 2/eta = 80, 6000 steps, main4 protocol + fixed eigenvector bank + theta every 10 + replay.
# Rerunning resumes from cache/stageB/ckpt.pt (training) and replay_partial.npz (replay).
set -e
cd /home/fzeng/ml/research/art/ribbon
export OMP_NUM_THREADS=4
/home/fzeng/ml/research/art/.venv/bin/python stage_b.py --inv 80 --steps 6000 --check-every 500 \
  --check-iters 60 --bank-every 250 --bank-iters 30 --theta-every 10 --ckpt-every 500 \
  --out-dir cache/stageB --replay
