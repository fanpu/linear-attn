"""Initialization basin maps for discrete MWU learning.

System: two agents, three parallel links with linear costs c_e(l) = c_e * l, exponential MWU
with step eta, both agents starting from the SAME mixed strategy x0 (the symmetric subspace is
invariant, so the map is x -> x * exp(-eta c (1 + x)) / Z on the 2-simplex).
Each pixel of the simplex is one initial condition, run T0 steps; the attractor reached is
identified by (period, time-averaged strategy over T1 steps, Lyapunov exponent).

python compute_basins.py toy | plates | boxcount
"""
import sys
import time

import numpy as np
import torch

from gamelib import DT, gpu_setup, dev


def simplex_grid(R, lo=(0.0, 0.0), width=1.0):
    """Pixel (i,j): x_P = lo0 + width*j/(R-1), x_S = lo1 + width*i/(R-1); x_R = 1 - x_P - x_S."""
    a = torch.linspace(0, 1, R, dtype=DT) * width
    S, P = torch.meshgrid(a + lo[1], a + lo[0], indexing='ij')
    X = torch.stack([1 - P - S, P, S], -1)
    mask = (X > 0).all(-1)
    return X.clamp_min(1e-15), mask


@torch.no_grad()
def attractors(X0, c, eta, T0=4000, T1=2048, chunk=1 << 21):
    """Return period (0 = none <= 64), mean strategy over T1, lyapunov exponent (tangent in
    the 2-D logit subspace), per initial condition."""
    shp = X0.shape[:-1]
    X0 = X0.reshape(-1, 3)
    out_per = torch.empty(X0.shape[0], dtype=torch.int16)
    out_mean = torch.empty(X0.shape[0], 3, dtype=torch.float32)
    out_lam = torch.empty(X0.shape[0], dtype=torch.float32)
    c = torch.as_tensor(c, dtype=DT, device=dev())
    for i in range(0, X0.shape[0], chunk):
        Q = torch.log(X0[i:i + chunk].to(dev()))
        n = Q.shape[0]
        for _ in range(T0):
            x = torch.softmax(Q, -1)
            Q = Q - eta * c * (1 + x)
            Q = Q - Q.max(-1, keepdim=True).values
        g = torch.Generator(device=dev()).manual_seed(0)
        dQ = torch.randn(n, 3, dtype=DT, device=dev(), generator=g); dQ = dQ - dQ.mean(-1, keepdim=True)
        dQ = dQ / dQ.norm(dim=-1, keepdim=True)
        L = torch.zeros(n, dtype=DT, device=dev()); mean = torch.zeros(n, 3, dtype=DT, device=dev())
        hist = []
        for t in range(T1):
            x = torch.softmax(Q, -1)
            dx = x * (dQ - (x * dQ).sum(-1, keepdim=True))
            Q = Q - eta * c * (1 + x); Q = Q - Q.max(-1, keepdim=True).values
            dQ = dQ - eta * c * dx; dQ = dQ - dQ.mean(-1, keepdim=True)
            nr = dQ.norm(dim=-1); dQ = dQ / nr[:, None]; L += torch.log(nr)
            mean += x
            if t >= T1 - 65:
                hist.append(x)
        H = torch.stack(hist)  # 65, n, 3
        per = torch.zeros(n, dtype=torch.int16, device=dev())
        for p in range(1, 65):
            d = (H[-1] - H[-1 - p]).abs().amax(-1)
            per = torch.where((per == 0) & (d < 1e-8), torch.tensor(p, dtype=torch.int16, device=dev()), per)
        out_per[i:i + chunk] = per.cpu(); out_mean[i:i + chunk] = (mean / T1).float().cpu()
        out_lam[i:i + chunk] = (L / T1).float().cpu()
    return out_per.reshape(shp), out_mean.reshape(shp + (3,)), out_lam.reshape(shp)


def label(per, mean, lam, tol=2e-3):
    """Attractor identity: cycles by (period, mean rounded); chaotic (per==0) by mean rounded coarser."""
    key = np.where(per[..., None] > 0, np.round(mean / tol), np.round(mean / (5 * tol)))
    code = per.astype(np.int64) * 10 ** 12 + key[..., 0].astype(np.int64) * 10 ** 6 + key[..., 1].astype(np.int64)
    u, inv = np.unique(code, return_inverse=True)
    counts = np.bincount(inv.ravel())
    order = np.argsort(-counts)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    return rank[inv].reshape(per.shape), u[order], counts[order]


def toy():
    c = [1.0, 1.2, 1.5]
    for eta in (16, 24, 32, 40, 56):
        X, m = simplex_grid(384)
        t = time.time()
        per, mean, lam = attractors(X, c, eta, T0=3000, T1=1024)
        lab, codes, counts = label(per.numpy(), mean.numpy(), lam.numpy())
        mm = m.numpy()
        vals, cnt = np.unique(lab[mm], return_counts=True)
        top = [(int(v), int(k), int(codes[v] // 10 ** 12)) for v, k in zip(vals[:8], cnt[:8])]
        print(f'eta={eta}: {time.time()-t:.0f}s  #labels={len(vals)}  top(label,count,period)={top}  '
              f'chaotic frac={np.mean(per.numpy()[mm]==0):.3f}', flush=True)
        np.savez_compressed(f'cache/basin_toy_eta{eta}.npz', per=per.numpy(), mean=mean.numpy(), lam=lam.numpy(), mask=mm)


if __name__ == '__main__':
    gpu_setup()
    globals()[sys.argv[1]]()
