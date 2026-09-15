#!/usr/bin/env bash
# M2: two more width-4096 256^3 draws (seeds 8, 9) for draw statistics. Run through gpu1.sh.
set -e
cd /home/fzeng/ml/research/art/rough-skin
for s in 8 9; do
  OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python compute_fields.py 4096 1 $s
done
