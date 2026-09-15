#!/usr/bin/env bash
# GPU smoke test of stage_b.py: kill-and-resume plus replay on a 40-step toy.
set -e
cd /home/fzeng/ml/research/art/ribbon
export OMP_NUM_THREADS=4
P=/home/fzeng/ml/research/art/.venv/bin/python
rm -rf cache/smoke
A="--steps 40 --check-every 20 --check-iters 5 --bank-every 20 --bank-iters 5 --ckpt-every 20 --out-dir cache/smoke"
$P stage_b.py $A --stop-after 25
$P stage_b.py $A --replay
$P - <<'PY'
import numpy as np
L=np.load('cache/smoke/log.npz'); R=np.load('cache/smoke/replay.npz')
print('steps_done',L['steps_done'],'final',L['final'])
print('lam1*eta/2',L['evals'][[0,20,39],0]*L['eta']/2)
print('bank_t',L['bank_t'],'bank_evals',L['bank_evals'][:,0])
print('anchor_err',R['anchor_err'])
th=np.load('cache/smoke/theta_f32.npy',mmap_mode='r'); B=np.load('cache/smoke/bank_vecs.npy',mmap_mode='r')
print('bank norms',np.linalg.norm(B.reshape(-1,B.shape[-1]),axis=1))
print('proj_moving vs proj_bank at bank steps', L['proj_moving'][20], R['proj_bank'][20,3:6])
print('proj nan', np.isnan(R['proj_bank']).sum())
PY
