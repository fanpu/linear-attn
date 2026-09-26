#!/bin/bash
# CPU post-processing of the pasar runs: summaries, the README table, the random-kick control and
# the tangent-kernel gains. Run from art/palimpsest.
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
PY=../.venv/bin/python
R=cache/runs
$PY analyze.py $R > $R/analyze.txt
$PY table.py $R > $R/table.txt
for f in $R/*_AB_*.npz; do
  n=$(basename $f .npz)
  case $n in *_kick|*_kernel*) continue;; esac
  [ -f $R/${n}_kick.npz ] || $PY kick.py $R $n 3 > $R/${n}_kick.txt
  [ -f $R/${n}_kernel.npz ] || $PY kernel.py $R $n > $R/${n}_kernel.txt
  [ -f $R/${n}_kernel_init.npz ] || $PY kernel.py $R $n init > $R/${n}_kernel_init.txt
done
