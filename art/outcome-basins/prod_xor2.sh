#!/usr/bin/env bash
# Piece B follow-ups, part 1: null uncertainty + eta series (1024^2). Run via gpu_run.sh.
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_uncert.py xor --seed 4 --eta 0.3 --win -3 3 -3 3 --M 20000 --eps -1 -11 21 --tag xor_e0.3_null
$PY compute_map.py xor --seed 4 --eta 0.3 0.8 1.0 1.1 1.15 1.25 --win -3 3 -3 3 --res 1024 --tag eta_xor
echo PROD_XOR2_DONE
