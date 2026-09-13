#!/usr/bin/env bash
# Launch both terminal-phase runs through the shared GPU slot limiter.
cd "$(dirname "$0")"; mkdir -p logs
PY=/home/fzeng/ml/research/art/.venv/bin/python
G=/home/fzeng/ml/research/art/_shared/gpu_run.sh
OMP_NUM_THREADS=4 $G $PY train_nc.py --classes 0 1 2 3 4 5 6 7 8 9 --tag c10 --width 32 --epochs 200 --n_sub_train 1000 > logs/c10.log 2>&1 &
OMP_NUM_THREADS=4 $G $PY train_nc.py --classes 0 1 2 3 --tag c4 --width 32 --epochs 250 --n_sub_train 2500 > logs/c4.log 2>&1 &
wait
