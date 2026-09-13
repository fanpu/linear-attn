"""Initialization basin maps for discrete MWU learning.

System: two agents, three parallel links with linear costs c_e(l) = c_e * l, exponential MWU
with step eta, both agents starting from the SAME mixed strategy x0 (the symmetric subspace is
invariant, so the map is x -> x * exp(-eta c (1 + x)) / Z on the 2-simplex).
Each pixel of the simplex is one initial condition, run T0 steps; the attractor reached is
identified by (period, time-averaged strategy over T1 steps, Lyapunov exponent).

python compute_basins.py toy | atlas
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


@torch.no_grad()
def basin_2x2(sign, s, q, R, T=3000):
    """MWU on a symmetric 2x2 coordination (sign=+1) or anti-coordination (sign=-1) game,
    u' = u + sign*s*(sigma(v)-q), v' = v + sign*s*(sigma(u)-q); pixel = (x0, y0) in (0,1)^2.
    Label: 0/1/2/3 = pure profile (x->1?, y->1?) reached (|logit| > 30), 4 = not converged."""
    g = torch.linspace(0, 1, R + 2, dtype=DT, device=dev())[1:-1]
    Y, X = torch.meshgrid(g, g, indexing='ij')
    u = torch.log(X / (1 - X)); v = torch.log(Y / (1 - Y))
    for _ in range(T):
        u, v = u + sign * s * (torch.sigmoid(v) - q), v + sign * s * (torch.sigmoid(u) - q)
        u = u.clamp(-60, 60); v = v.clamp(-60, 60)
    conv = (u.abs() > 30) & (v.abs() > 30)
    lab = torch.where(conv, 2 * (u > 0).long() + (v > 0).long(), torch.full_like(u, 4, dtype=torch.long))
    return lab.cpu().numpy()


@torch.no_grad()
def basin_3link(c, eta, R, T=4000, lo=(0.0, 0.0), width=1.0):
    """Two agents, three links, exponential MWU, full (asymmetric) game. Pixel = player 1's start x0 on
    the simplex; player 2 starts at the P<->R swap of x0. Label = 3*argmax(x)+argmax(y) once both
    strategies are within 1e-3 of pure, 9 = not converged, -1 = outside the simplex."""
    X, m = simplex_grid(R, lo, width)
    X = X.to(dev()); c = torch.as_tensor(c, dtype=DT, device=dev())
    Qx = torch.log(X); Qy = torch.log(X[..., [1, 0, 2]])
    for _ in range(T):
        x = torch.softmax(Qx, -1); y = torch.softmax(Qy, -1)
        Qx, Qy = Qx - eta * c * (1 + y), Qy - eta * c * (1 + x)
        Qx = (Qx - Qx.max(-1, keepdim=True).values).clamp_min(-700); Qy = (Qy - Qy.max(-1, keepdim=True).values).clamp_min(-700)
    x = torch.softmax(Qx, -1); y = torch.softmax(Qy, -1)
    conv = (x.amax(-1) > 0.999) & (y.amax(-1) > 0.999)
    lab = torch.where(conv, 3 * x.argmax(-1) + y.argmax(-1), torch.full(x.shape[:-1], 9, device=dev()))
    lab = torch.where(m.to(dev()), lab, torch.full_like(lab, -1))
    return lab.cpu().numpy()


ATLAS = [  # name, kind, params (large step) , null params (small step)
    ('coord2x2', 'c2', dict(sign=1, s=20.0, q=0.45), dict(sign=1, s=0.5, q=0.45)),
    ('anti2x2', 'c2', dict(sign=-1, s=30.0, q=0.7), dict(sign=-1, s=0.5, q=0.7)),
    ('link3_a', 'l3', dict(c=[1.0, 1.2, 1.5], eta=60.0), dict(c=[1.0, 1.2, 1.5], eta=1.0)),
    ('link3_b', 'l3', dict(c=[1.0, 1.3, 1.1], eta=40.0), dict(c=[1.0, 1.3, 1.1], eta=1.0)),
]


def atlas():
    out = {}
    for name, kind, big, small in ATLAS:
        for tag, prm in (('big', big), ('null', small)):
            for R in (256, 512, 1024):
                t = time.time()
                lab = basin_2x2(R=R, **prm) if kind == 'c2' else basin_3link(R=R, **prm)
                out[f'{name}_{tag}_R{R}'] = lab.astype(np.int8)
                print(f'{name} {tag} R={R}: {time.time()-t:.0f}s labels={np.unique(lab).tolist()} '
                      f'nonconv={np.mean(lab == (4 if kind == "c2" else 9)):.4f}', flush=True)
    np.savez_compressed('cache/basin_atlas.npz', **out, meta=str(ATLAS))


def island_zoom():
    """5x zoom on the only non-trivial basin feature (link3_b, eta=40), for the resolution check."""
    out = {}
    for R in (512, 1024):
        out[f'R{R}'] = basin_3link([1.0, 1.3, 1.1], 40.0, R, lo=(0.02, 0.52), width=0.25).astype(np.int8)
        print('island zoom', R, np.unique(out[f'R{R}']).tolist(), flush=True)
    np.savez_compressed('cache/basin_island_zoom.npz', **out, lo=(0.02, 0.52), width=0.25)


if __name__ == '__main__':
    gpu_setup()
    globals()[sys.argv[1]]()
