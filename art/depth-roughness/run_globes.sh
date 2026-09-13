#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY render_globes.py hero heaviside_L2 dark
$PY render_globes.py hero heaviside_L2 plotter
$PY render_globes.py series plotter
$PY render_globes.py spin heaviside_L2 dark 480
$PY render_globes.py series riso
$PY render_globes.py spin heaviside_L3 plotter 480
