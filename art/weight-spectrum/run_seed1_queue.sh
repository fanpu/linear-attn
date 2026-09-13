#!/usr/bin/env bash
# Replicate seed 1 of the batch-size series; keeps at most 6 single-thread train.py processes alive.
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
for bs in 1024 512 256 128 64 32; do
  while [ "$(pgrep -fc 'python train.py --name mlp_bs')" -ge 6 ]; do sleep 30; done
  OMP_NUM_THREADS=1 nohup $PY train.py --name mlp_bs${bs}_s1 --data fmnist --widths 1024,1024,1024 --bs $bs --lr 0.01 \
    --momentum 0.9 --epochs 30 --n_ckpt 40 --n_full 2 --k_vec 8 --seed 1 --device cpu --threads 1 > logs/mlp_bs${bs}_s1.log 2>&1 &
  sleep 60
done
wait
