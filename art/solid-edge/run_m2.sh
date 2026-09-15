#!/usr/bin/env bash
# M2 GPU chain (queued once via gpu1.sh). Every stage resumes from per-chunk checkpoints.
# B's window is read from cache/vol/B_window.json when B starts (written by choose_windows.py B
# unless it already exists, so it can be overridden by hand while A runs).
set -e
cd /home/fzeng/ml/research/art/solid-edge
export OMP_NUM_THREADS=4
PY=/home/fzeng/ml/research/art/.venv/bin/python
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv || true
$PY toys_compute.py --toy a --tag _fix          # M1 toy (a) recomputed with the non-finite-loss measure fix
$PY choose_windows.py candidates
$PY vol_run.py --name nullA128 --kind quad2 --grid toyA --res 128 --stages f32,shell,audit,final
$PY choose_windows.py null
$PY vol_run.py --name probe_null --kind quad2 --grid json --grid_json cache/vol/null_window.json --res 64 --stages f32
for c in c1 c2 c3; do $PY vol_run.py --name probe_$c --grid json --grid_json cache/vol/cand_$c.json --res 64 --stages f32; done
$PY choose_windows.py B
echo PROBES_DONE
$PY vol_run.py --name A128 --grid toyA --res 128 --stages f32,shell,audit,final
echo A_DONE
$PY vol_run.py --name nullB256 --kind quad2 --grid json --grid_json cache/vol/null_window.json --res 256 --stages f32,shell,audit,final,plane
$PY vol_run.py --name nullB256_sub64 --kind quad2 --grid sub --parent nullB256 --res 64 --stages f32,shell,audit,final
echo NULLB_DONE
$PY vol_run.py --name B256 --grid json --grid_json cache/vol/B_window.json --res 256 --stages f32,shell,audit,final,plane
$PY vol_run.py --name B256_sub64 --grid sub --parent B256 --res 64 --stages f32,shell,audit,final
echo ALL_DONE
