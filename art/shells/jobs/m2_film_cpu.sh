#!/usr/bin/env bash
# M2 epoch film: ResNet-20 17^3 random-direction volumes (h = 0.16, [-1.28, 1.28]) at epochs 40 0 1 2 4 8 16,
# directions filter-normalised per checkpoint.  CPU (declared: GPU queue congested), 3 slab-shard workers x 4 threads.
#   setsid nohup bash jobs/m2_film_cpu.sh > logs/m2_film_cpu.log 2>&1 < /dev/null &
set -u
cd /home/fzeng/ml/research/art/shells
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
ts() { date '+%F %T'; }
for e in 40 0 1 2 4 8 16; do
  E=$(printf %03d $e); name=resnet20_ep${E}_random_g17; out=cache/vol/$name.npz
  if [ ! -f "$out" ]; then
    echo "[film $(ts)] epoch $e start"
    for k in 0 1 2; do
      $PY volume.py --device cpu --epoch $e --res 17 --h 0.16 --K 1 --slab-shard $k 3 > logs/film_ep${E}_$k.log 2>&1 &
    done
    wait
    $PY volume.py --device cpu --epoch $e --res 17 --h 0.16 --assemble-only >> logs/film_ep${E}_0.log 2>&1
  fi
  extra=""; [ "$e" = 40 ] && extra="--compare-vol cache/vol/resnet20_final_random_g27.npz"
  [ -f "$out" ] && $PY analyze.py "$out" $extra > logs/analyze_$name.log 2>&1 && $PY preview.py "$out" >> logs/analyze_$name.log 2>&1
  echo "[film $(ts)] epoch $e $( [ -f "$out" ] && echo done || echo INCOMPLETE )"
done
echo "[film $(ts)] all done"
