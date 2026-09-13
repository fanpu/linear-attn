#!/usr/bin/env bash
# Model-wise x epoch-wise deep double descent sweep: 4 worker processes sharing ONE theory GPU slot.
#   ../_shared/gpu_run.sh bash run_cnn.sh <tag> <n> <noise> <epochs>
set -u
cd "$(dirname "$0")"
TAG=${1:-main}; N=${2:-10000}; P=${3:-0.2}; E=${4:-500}
PY=../.venv/bin/python
export OMP_NUM_THREADS=2
mkdir -p cache/cnn/logs
$PY train_cnn.py --tag $TAG --n $N --noise $P --epochs $E --widths 64            > cache/cnn/logs/${TAG}_n${N}_a.log 2>&1 &
$PY train_cnn.py --tag $TAG --n $N --noise $P --epochs $E --widths 48 12         > cache/cnn/logs/${TAG}_n${N}_b.log 2>&1 &
$PY train_cnn.py --tag $TAG --n $N --noise $P --epochs $E --widths 32 24 5       > cache/cnn/logs/${TAG}_n${N}_c.log 2>&1 &
$PY train_cnn.py --tag $TAG --n $N --noise $P --epochs $E --widths 1 2 3 4 6 8 10 16 > cache/cnn/logs/${TAG}_n${N}_d.log 2>&1 &
wait
echo "all workers finished"
