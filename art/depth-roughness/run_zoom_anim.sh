#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
export OMP_NUM_THREADS=6
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY render_zoom_anim.py gp heaviside_L2 1
