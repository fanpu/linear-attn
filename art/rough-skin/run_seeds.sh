#!/usr/bin/env bash
# Extra width-1024 128^3 draws (seeds 8-11) for draw-to-draw scatter of D. Run through gpu1.sh.
set -e
cd /home/fzeng/ml/research/art/rough-skin
for s in 8 9 10 11; do
  OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python compute_fields.py 1024 2 $s
done
