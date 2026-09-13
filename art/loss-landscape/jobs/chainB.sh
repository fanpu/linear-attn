#!/bin/bash
# 51^2 ResNet-20 pair, 1D slices, subset check, zoom lines, PCA planes, hero shard 1/2
cd /home/fzeng/ml/research/art/loss-landscape
L="/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 1000"
$L --model resnet20 --res 51 --tag g51
$L --model resnet20_noshort --res 51 --tag g51
for m in resnet20 resnet20_noshort resnet56 resnet56_noshort; do $L --model $m --line --res 401 --tag line; done
/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 5000 --model resnet56_noshort --line --res 101 --tag line_n5000
/home/fzeng/ml/research/art/_shared/gpu_run.sh /home/fzeng/ml/research/art/.venv/bin/python landscape.py --n 1000 --model resnet56_noshort --line --res 101 --tag line_n1000
for m in resnet56_noshort resnet56; do
  $L --model $m --line --res 201 --xlim -0.5 0.5 --center 0.5 0 --tag zoom0
  $L --model $m --line --res 201 --xlim -0.05 0.05 --center 0.5 0 --tag zoom1
  $L --model $m --line --res 201 --xlim -0.005 0.005 --center 0.5 0 --tag zoom2 --float64
  $L --model $m --line --res 201 --xlim -0.0005 0.0005 --center 0.5 0 --tag zoom3 --float64
done
$L --model resnet56 --dirs cache/dirs/resnet56_pca.pt --res 41 --xlim -65 8 --ylim -20 24 --tag pca41
$L --model resnet56_noshort --dirs cache/dirs/resnet56_noshort_pca_from10.pt --res 41 --xlim -72 6 --ylim -14 14  --tag pcaf10_41
until [ -f cache/surf/resnet56_noshort_final_g51.npz ]; do sleep 60; done
$L --model resnet56_noshort --res 101 --tag g101 --shard 1 2 --reuse cache/surf/resnet56_noshort_final_g51.npz
until [ -f cache/surf/resnet56_final_g51.npz ]; do sleep 60; done
$L --model resnet56 --res 101 --tag g101 --shard 1 2 --reuse cache/surf/resnet56_final_g51.npz
