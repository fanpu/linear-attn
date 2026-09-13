#!/usr/bin/env bash
# CPU batch-size series (GPU was saturated). Each run is single-threaded; 5 in parallel.
# 3-hidden-layer width-1024 ReLU MLP on FashionMNIST, SGD lr=0.01 momentum=0.9, 30 epochs, Glorot-normal init.
cd "$(dirname "$0")"
PY=/home/fzeng/ml/research/art/.venv/bin/python
SEED=${SEED:-0}
run() {
  bs=$1; nck=$2; nfull=$3
  OMP_NUM_THREADS=1 $PY train.py --name mlp_bs${bs}_s${SEED} --data fmnist --widths 1024,1024,1024 --bs $bs \
    --lr 0.01 --momentum 0.9 --epochs 30 --n_ckpt $nck --n_full $nfull --k_vec 32 --seed $SEED --device cpu --threads 1 \
    > logs/mlp_bs${bs}_s${SEED}.log 2>&1
}
export -f run; export PY SEED
printf '%s\n' "16 120 24" "32 60 6" "64 60 6" "128 60 6" "256 60 6" "512 60 6" "1024 60 6" \
  | xargs -P ${PAR:-5} -L 1 bash -c 'run $0 $1 $2'
