#!/usr/bin/env bash
# M1 GPU chain (queued once via gpu1.sh). Each step resumes from per-chunk checkpoints.
set -e
cd /home/fzeng/ml/research/art/solid-edge
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv || true
$PY toys_compute.py --toy a
$PY toys_compute.py --toy null
$PY toys_compute.py --toy a --dtype float64 --sigma_plane_only
$PY toys_compute.py --toy b
echo ALL_DONE
