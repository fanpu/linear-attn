#!/bin/bash
# checkpoint film: 1D slices at every saved checkpoint (dirs re-normalized to each checkpoint)
cd /home/fzeng/ml/research/art/loss-landscape
L="/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 1000"
for e in 1 2 3 4 5 6 8 10 13 15 16 20 25 30 35 40; do
  for m in resnet56 resnet56_noshort; do $L --model $m --epoch $e --line --res 101 --tag ckline; done
done
