#!/usr/bin/env bash
# Piece B: nested zoom into the divergence fan (level-0 centre = most boundary-dense point of the hero map).
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_zoom.py xor --seed 4 --eta 1.2 --win -2.559110893991207 -1.559110893991207 -0.69491939423546656 0.30508060576453344 --levels 6 --factor 6 --res 1024 --tag zX
echo PROD_XOR_ZOOM_DONE
