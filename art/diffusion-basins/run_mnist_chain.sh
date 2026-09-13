#!/usr/bin/env bash
# Waits for MNIST training to finish, then runs the GPU compute tasks through gpu_run.sh (one slot, sequential).
cd /home/fzeng/ml/research/art/diffusion-basins
PY=/home/fzeng/ml/research/art/.venv/bin/python; G=/home/fzeng/ml/research/art/_shared/gpu_run.sh
until grep -q "\[memo\] it 11999" logs/mnist_memo.log && grep -q "\[ddpm\] it 14999" logs/mnist_ddpm.log; do sleep 60; done
sleep 30
$G sh -c "$PY mnist_compute.py map ddpm hero 192 50; $PY mnist_compute.py map memo memo_hero 192 50; \
$PY mnist_compute.py map memo_exact memo_exact 192 50; $PY mnist_compute.py stepsweep 128; \
$PY mnist_compute.py zoom 128 4; $PY mnist_compute.py uncert; $PY mnist_compute.py f64check"
