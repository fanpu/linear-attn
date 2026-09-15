#!/usr/bin/env bash
# M2 GPU chain (run through gpu1.sh): full-resolution casts, Spectral fog, plate reproduction. Resumable (skips outputs).
set -e
cd /home/fzeng/ml/research/art/rough-skin
PY=/home/fzeng/ml/research/art/.venv/bin/python
export OMP_NUM_THREADS=4
$PY verify_plate.py
$PY render_casts.py --size 2048 --device cuda
$PY render_fog.py --size 2048 --device cuda
