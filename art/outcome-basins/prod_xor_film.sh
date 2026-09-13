#!/usr/bin/env bash
# Piece B eta film, 72 frames at 480^2, one of 3 parallel parts: ./prod_xor_film.sh <part>
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_frames.py xor eta --seed 4 --eta0 0.6 --eta1 1.26 --n 72 --win -3 3 -3 3 --res 480 --tag xor_etafilm --part $1 --parts 3
echo PROD_XOR_FILM_$1_DONE
