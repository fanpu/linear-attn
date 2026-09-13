#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
export OMP_NUM_THREADS=6
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_multiscale.py heaviside_L6 1 --store
$PY compute_multiscale.py heaviside_L6 2,3
$PY compute_multiscale.py heaviside_L1 4 --store
