#!/bin/bash
# PCA planes rerun (chain B failed on torch.load weights_only)
cd /home/fzeng/ml/research/art/loss-landscape
L="/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 1000"
$L --model resnet56 --dirs cache/dirs/resnet56_pca.pt --res 41 --xlim -65 8 --ylim -20 24 --tag pca41 &
$L --model resnet56_noshort --dirs cache/dirs/resnet56_noshort_pca_from10.pt --res 41 --xlim -72 6 --ylim -14 14  --tag pcaf10_41 &
wait
