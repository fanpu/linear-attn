"""1-ulp precision floor on probe windows (CPU, float64): recompute N shell voxels with both learning
rates multiplied by (1 + 2^-52) and report the label flip fraction (the source's 'seeing' test).
  OMP_NUM_THREADS=4 ../.venv/bin/python ulp_check.py probe_c1 probe_c2 probe_c3 probe_null"""
import json, sys, time
import numpy as np, torch
import se_engine as se
from vol_run import mixed3

out = {}
for name in sys.argv[1:]:
    d = f'cache/vol/{name}/'
    g = json.load(open(d + 'grid.json')); M = np.load(d + 'f32.npy'); R = M.shape[0]
    kind = 'quad2' if 'null' in name else 'net2'
    sh = np.flatnonzero(mixed3(M < 0))
    idx = np.sort(np.random.default_rng(2).choice(sh, min(400, len(sh)), replace=False))
    k, i, j = np.unravel_index(idx, (R, R, R))
    ax = [torch.tensor(g[a], dtype=torch.float64) for a in ('eta0', 'eta1', 'sigma')]
    pr = se.make_problem(kind, device='cpu', dtype=torch.float64)
    t = time.time()
    res = []
    for f in (1.0, 1.0 + 2.0 ** -52):
        m = se.train_chunk(pr, [ax[0][j] * f, ax[1][i] * f], sigma=ax[2][k], compiled=False)['measure'].numpy()
        res.append(np.sign(m))
    out[name] = dict(n=len(idx), ulp_flip_frac=float((res[0] != res[1]).mean()),
                     f32_vs_f64_shell_agree=float((res[0] == np.sign(M.ravel()[idx])).mean()), shell_frac_64=float(len(sh) / R ** 3))
    print(name, out[name], f'{time.time()-t:.0f}s', flush=True)
json.dump(out, open('cache/vol/ulp_probes.json', 'w'), indent=1)
