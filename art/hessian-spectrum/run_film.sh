#!/usr/bin/env bash
# exact spectra at every checkpoint of the film run (waits for the exact class-count series to finish)
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
while pgrep -f run_exact_series.sh > /dev/null; do sleep 30; done
until [ -f cache/runs/film_mlps_C10/meta.json ]; do sleep 20; done
OMP_NUM_THREADS=4 $PY exact_analyze.py --run film_mlps_C10 --steps all --per_class 200 --noG --kvec 12
