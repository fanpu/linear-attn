#!/bin/bash
# one pasar job per run. usage: submit.sh <arch> <seed> <labels> [epochs] [task] [time]
arch=$1; seed=$2; labels=$3; epochs=${4:-8}; task=${5:-mnist}; t=${6:-10m}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pasar submit --json --time $t --mem 4G \
  --name fl-${task}-${arch}${EXTRA:+-bnb}-${labels}-s${seed} --tag art-first-learned \
  --note "First Learned: per-example learning time for all 60k ${task} train digits (${arch}, ${labels} labels, seed ${seed})" \
  --by art-firstlearned --cwd /home/fzeng/ml/research/art \
  -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python first-learned/train.py \
  --arch $arch --seed $seed --labels $labels --epochs $epochs --task $task ${EXTRA}
