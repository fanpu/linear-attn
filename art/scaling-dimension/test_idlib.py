"""Sanity test of the ID estimators on manifolds of known dimension."""
import numpy as np, torch, time
torch.cuda.set_per_process_memory_fraction(0.10)
from idlib import estimate_all
rng = np.random.default_rng(0)
n = 12000
print(f"{'manifold':>14} {'d':>3} {'TwoNN':>7} {'TwoNN-ML':>8} {'MLE5':>6} {'MLE10':>6} {'MLE20':>6}")
for d in [2, 3, 4, 6, 8, 12, 16]:
    for name in ["cube", "gauss", "torus"]:
        if name == "cube":
            Z = rng.uniform(-.5, .5, (n, d)); D = 24
        elif name == "gauss":
            Z = rng.standard_normal((n, d)); D = 24
        else:
            th = rng.uniform(0, 2*np.pi, (n, d)); Z = np.concatenate([np.cos(th), np.sin(th)], 1); D = max(24, 2*d)
        Q, _ = np.linalg.qr(rng.standard_normal((D, Z.shape[1])))
        X = Z @ Q.T
        t = time.time()
        o, _, _ = estimate_all(X, bootstrap=False)
        print(f"{name:>14} {d:>3} {o['twonn']:7.2f} {o['twonn_mle']:8.2f} {o['mle5']:6.2f} {o['mle10']:6.2f} {o['mle20']:6.2f}  ({time.time()-t:.1f}s)")
