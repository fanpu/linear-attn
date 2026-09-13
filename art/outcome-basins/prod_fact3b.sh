#!/usr/bin/env bash
# follow-up plates for piece A
set -u
cd /home/fzeng/ml/research/art/outcome-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python
$PY compute_map.py fact3 --s 0.0 --eta 0.3 --win -5 5 -5 5 --res 4096 --tag hero_f3_s0_e0.3_wide
$PY compute_map.py fact3 --s 0.5 --eta 1.1 --win -1.6361301369863015 -0.7611301369863015 -2.1429794520547945 -1.2679794520547945 --res 4096 --tag hero_f3_s0.5_e1.1_tassel
$PY compute_map.py fact3 --s 0.5 --eta 1.1 --win -1.1976669520547947 -1.0882919520547947 -1.8483518835616437 -1.7389768835616437 --res 4096 --tag hero_f3_s0.5_e1.1_tasselzoom
$PY compute_map.py fact3 --s 1.5 --eta 1.2 --win -3.5 3.5 -3.5 3.5 --res 4096 --tag hero_f3_s1.5_e1.2
echo PROD_FACT3B_DONE
