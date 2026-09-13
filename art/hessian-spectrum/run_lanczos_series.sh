#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
for C in 2 3 4 5 7 10; do
  until [ -f cache/runs/s_mlp_C$C/meta.json ]; do sleep 20; done
  OMP_NUM_THREADS=4 $PY analyze.py --run s_mlp_C$C --m_top 200 --m_slq 80 --nv 4 --per_class 500
done
