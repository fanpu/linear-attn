#!/usr/bin/env bash
# All 20 pre-registered runs, sequentially. Finished runs are skipped; exit 75 (pause) propagates to gpu_run.sh,
# which reruns this script after resume (the paused run continues from its checkpoint).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
run() { OMP_NUM_THREADS=4 $PY train.py "$@"; rc=$?; [ $rc -eq 75 ] && exit 75; [ $rc -ne 0 ] && echo "FAILED rc=$rc: $*"; }
for init in I sqrt; do
  for T in 32 128 512; do for qd in 0 1; do run --model delta --T $T --qd $qd --init $init; done; done
  for T in 128 512; do run --model linattn --T $T --init $init; done
done
for seed in 1 2; do
  run --model delta --T 512 --init I --seed $seed
  run --model linattn --T 512 --init I --seed $seed
done
echo ALL_DONE
