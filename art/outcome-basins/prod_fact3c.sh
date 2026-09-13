#!/usr/bin/env bash
# high-statistics small-eps tail of the uncertainty exponent (riddling test) for piece A
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_uncert.py fact3 --s 0.5 --eta 1.1 --win -3.5 3.5 -3.5 3.5 --M 1000000 --eps -4 -14 11 --tag f3_e1.1_full_bigM
$PY compute_uncert.py fact3 --s 0.5 --eta 1.1 --win -1.6361 -0.7611 -2.1430 -1.2680 --M 1000000 --eps -4 -14 11 --tag f3_e1.1_L1_bigM
echo PROD_FACT3C_DONE
