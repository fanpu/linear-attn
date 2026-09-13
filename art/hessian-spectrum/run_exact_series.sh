#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
for C in 2 3 4 5 7 10; do
  until [ -f cache/runs/s_mlps_C$C/meta.json ]; do sleep 20; done
  OMP_NUM_THREADS=4 $PY exact_analyze.py --run s_mlps_C$C --E
done
