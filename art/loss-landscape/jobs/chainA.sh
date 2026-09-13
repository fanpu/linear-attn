#!/bin/bash
# 51^2 ResNet-56 pair, then hero 101^2 shard 0/2 (reusing the 51^2 points)
cd /home/fzeng/ml/research/art/loss-landscape
L="/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 1000"
$L --model resnet56_noshort --res 51 --tag g51
$L --model resnet56 --res 51 --tag g51
$L --model resnet56_noshort --res 101 --tag g101 --shard 0 2 --reuse cache/surf/resnet56_noshort_final_g51.npz
$L --model resnet56 --res 101 --tag g101 --shard 0 2 --reuse cache/surf/resnet56_final_g51.npz
