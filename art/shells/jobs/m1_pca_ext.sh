#!/usr/bin/env bash
# Extend the PCA-direction volume past the +a and +c faces, where its loss <= 0.5 .. 2.3 shells were clipped.
# Same spacing and origin; the 27^3 points are prefilled from the base volume. Resumes per slab.
#   setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh bash jobs/m1_pca_ext.sh > logs/m1_pca_ext.log 2>&1 < /dev/null &
set -euo pipefail
cd /home/fzeng/ml/research/art/shells
export OMP_NUM_THREADS=4
/home/fzeng/ml/research/art/.venv/bin/python volume.py --dirs pca --K 1 --img-batch 1000 \
  --extend 0 9 0 0 0 5 --reuse cache/vol/resnet20_final_pca_g27.npz
