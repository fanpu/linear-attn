#!/usr/bin/env bash
# M2: ResNet-56 and ResNet-56-noshort 27^3 random-direction volumes, as N slab shards, one gpu1.sh job per shard
# (controller rule: each queued job <= ~20 min). CPU-side loop; rerun resumes (finished slabs are skipped).
#   setsid nohup bash jobs/m2_gpu_loop.sh > logs/m2_gpu_loop.log 2>&1 < /dev/null &
set -u
cd /home/fzeng/ml/research/art/shells
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
G=/home/fzeng/ml/research/art/_shared/gpu1.sh
N=${N:-4}
ts() { date '+%F %T'; }
for m in resnet56 resnet56_noshort; do
  name=${m}_final_random_g27; out=cache/vol/$name.npz
  if [ ! -f "$out" ]; then
    for k in $(seq 0 $((N - 1))); do
      for attempt in 1 2 3; do
        miss=$($PY jobs/shard_missing.py cache/vol/$name.slabs 27 $k $N)
        [ "$miss" = 0 ] && break
        echo "[loop $(ts)] $m shard $k/$N attempt $attempt ($miss slabs missing): queueing"
        $G $PY volume.py --model $m --dirs random --K 1 --slab-shard $k $N >> logs/m2_$m.log 2>&1
        echo "[loop $(ts)] $m shard $k/$N attempt $attempt exit $?"
      done
    done
    $PY volume.py --model $m --dirs random --device cpu --assemble-only >> logs/m2_$m.log 2>&1
  fi
  if [ -f "$out" ]; then
    $PY analyze.py "$out" > logs/analyze_$m.log 2>&1 && $PY preview.py "$out" >> logs/analyze_$m.log 2>&1
    echo "[loop $(ts)] $m done"
  else
    echo "[loop $(ts)] $m INCOMPLETE"
  fi
done
echo "[loop $(ts)] all done"
