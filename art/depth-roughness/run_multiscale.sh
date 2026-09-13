#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
export OMP_NUM_THREADS=8
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_multiscale.py heaviside_L2 1 --store
$PY compute_multiscale.py heaviside_L2 2,3
$PY compute_multiscale.py heaviside_L1 1,2,3
$PY compute_multiscale.py heaviside_L3 1,2,3
$PY compute_multiscale.py heaviside_L4 1,2
$PY compute_multiscale.py relu_L2 1,2
$PY compute_multiscale.py rbf 1,2
