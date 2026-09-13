#!/usr/bin/env bash
# Production compute for piece B (XOR 2-2-1 tanh net, seed-4 slice). Run via gpu_run.sh.
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_map.py xor --seed 4 --eta 1.2 --win -3 3 -3 3 --res 2048 --tag hero_xor_s4_e1.2 --save_theta
$PY compute_uncert.py xor --seed 4 --eta 1.2 --win -3 3 -3 3 --M 20000 --eps -1 -11 21 --tag xor_e1.2
$PY compute_uncert.py xor --seed 4 --eta 1.2 --win -3 3 -3 3 --M 20000 --eps -1 -11 21 --key canon --tag xor_e1.2_canon
$PY compute_uncert.py xor --seed 4 --eta 0.3 --win -3 3 -3 3 --M 20000 --eps -1 -11 21 --tag xor_e0.3_null
$PY compute_map.py xor --seed 4 --eta 0.3 0.8 1.0 1.1 1.15 1.25 --win -3 3 -3 3 --res 768 --tag eta_xor
$PY compute_zoom.py xor --seed 4 --eta 1.2 --win -3 3 -3 3 --levels 6 --factor 6 --res 768 --tag zX
$PY compute_frames.py xor eta --seed 4 --eta0 0.6 --eta1 1.26 --n 60 --win -3 3 -3 3 --res 360 --tag xor_etafilm
echo PROD_XOR_DONE
