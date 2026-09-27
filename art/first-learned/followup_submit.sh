#!/bin/bash
# FOLLOW-UP: one pasar job per retuned run. usage: followup_submit.sh <arch> <seed> <cfg> <time> -- <extra train args>
arch=$1; seed=$2; cfg=$3; t=$4; shift 4; [ "$1" = "--" ] && shift
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pasar submit --json --time $t --mem 4G \
  --name fl-fu-${arch}-${cfg}-s${seed} --tag art-first-learned \
  --note "First Learned follow-up: lanes vs learning speed; ${arch} retuned (${cfg}) to reach ~99.5% train acc, seed ${seed}" \
  --by art-lanes --cwd /home/fzeng/ml/research/art \
  -- /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python first-learned/followup_train.py \
  --arch $arch --seed $seed --cfg $cfg "$@"
