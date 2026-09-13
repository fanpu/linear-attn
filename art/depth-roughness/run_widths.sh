#!/bin/bash
cd /home/fzeng/ml/research/art/depth-roughness
export OMP_NUM_THREADS=6
/home/fzeng/ml/research/art/.venv/bin/python render_zoom_anim.py widths
