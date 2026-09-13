#!/usr/bin/env bash
# class-count series: MNIST, classes 0..C-1, identical hyper-parameters
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
for C in 2 3 4 5 7 10; do
  OMP_NUM_THREADS=4 $PY train.py --ds MNIST --arch mlps --down 10 --C $C --epochs 10 --lr 0.05 --name s_mlps_C$C --threads 4
  OMP_NUM_THREADS=4 $PY train.py --ds MNIST --arch mlp --C $C --epochs 10 --lr 0.02 --name s_mlp_C$C --threads 4
done
# spectrograph run: many log-spaced checkpoints
OMP_NUM_THREADS=4 $PY train.py --ds MNIST --arch mlps --down 10 --C 10 --epochs 10 --lr 0.05 --n_ckpt 48 --name film_mlps_C10 --threads 4
