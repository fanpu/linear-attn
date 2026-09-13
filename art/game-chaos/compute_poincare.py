"""Continuous-time learning in zero-sum generalised RPS (Sato-Akiyama-Farmer 2002).
Computes and caches:
  cache/saf_repro.npz      SAF Fig.1 / Table I reproduction: 25 initial conditions x eps
  cache/kam_eps{e}.npz     single-energy Poincare sections seeded on a grid (true 2-D
                           area-preserving return map), with per-orbit Lyapunov exponents
  cache/traj.npz           long trajectories for simplex portraits and the plotter drawing
Usage: python compute_poincare.py [repro|kam|traj|all]
"""
import sys
import time

import numpy as np
from scipy.optimize import brentq

import replicator_c as rc

CACHE = 'cache'


def saf_ic(k):
    k = np.asarray(k, float)
    x = np.stack([np.full_like(k, 0.5), 0.01 * k, 0.5 - 0.01 * k], -1)
    y = np.broadcast_to([0.5, 0.25, 0.25], x.shape).copy()
    return x, y


def repro(T=100000.0):
    out = {}
    k = np.arange(1, 26)
    x, y = saf_ic(k)
    s0 = rc.probs_to_logits(x, y)
    for eps in (0.0, 0.25, 0.5):
        t = time.time()
        r = rc.integrate(s0, eps, -eps, h=0.01, T=T, every=100, maxsec=60000,
                         hist_every=10000, nhist=int(T / 100))
        tag = f'{eps:.2f}'
        out[f'sec_{tag}'] = r['sec'].astype(np.float32); out[f'nsec_{tag}'] = r['nsec']
        out[f'lyap_{tag}'] = r['lyap']; out[f'hist_{tag}'] = r['lyap_hist']; out[f'Hd_{tag}'] = r['Hdrift']
        print(f'repro eps={eps}: {time.time()-t:.1f}s  lambda1*1e3 (k=1..8) =',
              np.round(r['lyap'][:8] * 1e3, 2), 'max H drift', r['Hdrift'].max(), flush=True)
    out['k'] = k; out['T'] = T
    np.savez_compressed(f'{CACHE}/saf_repro.npz', **out)


def section_seed(p, q, H0, eps):
    """Point on section g=0 with x0=p, y1=q, energy H0 and g increasing, or None."""
    def parts(a):
        return np.array([p, a, 1 - p - a]), np.array([a - p + q, q, 1 - a + p - 2 * q])

    lo = max(0.0, p - q); hi = min(1 - p, 1 + p - 2 * q)
    if hi - lo < 1e-9:
        return None
    H = lambda a: rc.energy(*parts(a)) - H0
    grid = np.linspace(lo, hi, 402)[1:-1]
    vals = np.array([H(a) for a in grid])
    i = np.argmin(vals)
    if vals[i] > 0:
        return None
    A = rc_matrix(eps)
    # H -> +inf at both ends of the admissible interval and is convex: one root on each side
    for a_lo, a_hi in ((lo + 1e-12, grid[i]), (grid[i], hi - 1e-12)):
        try:
            a = brentq(H, a_lo, a_hi, xtol=1e-14)
        except ValueError:
            continue
        x, y = parts(a)
        if np.any(x <= 0) or np.any(y <= 0):
            continue
        xd = x * (A @ y - x @ A @ y); B = rc_matrix(-eps); yd = y * (B @ x - y @ B @ x)
        if xd[1] - xd[0] + yd[1] - yd[0] > 0:
            return x, y
    return None


def rc_matrix(e):
    return np.array([[e, -1, 1], [1, e, -1], [-1, 1, e]], float)


def kam(eps, H0, n=26, T=30000.0, name=None):
    """Seed a grid on the section at one energy; integrate; store crossings + lyap."""
    seeds = []
    for p in np.linspace(0.02, 0.98, n):
        for q in np.linspace(0.02, 0.98, n):
            s = section_seed(p, q, H0, eps)
            if s is not None:
                seeds.append(s)
    xs = np.array([s[0] for s in seeds]); ys = np.array([s[1] for s in seeds])
    t = time.time()
    r = rc.integrate(rc.probs_to_logits(xs, ys), eps, -eps, h=0.01, T=T, every=100,
                     maxsec=int(T / 3), hist_every=5000, nhist=int(T / 50))
    print(f'kam eps={eps} H0={H0:.4f}: {len(seeds)} orbits, {time.time()-t:.1f}s, crossings/orbit '
          f'{r["nsec"].mean():.0f}, chaotic(l>5e-3) {np.mean(r["lyap"] > 5e-3):.2f}, '
          f'max lyap {r["lyap"].max():.4f}, Hdrift {r["Hdrift"].max():.2e}', flush=True)
    np.savez_compressed(f'{CACHE}/{name or f"kam_eps{eps:.2f}"}.npz', sec=r['sec'].astype(np.float32),
                        nsec=r['nsec'], lyap=r['lyap'], hist=r['lyap_hist'], x0=xs, y0=ys, eps=eps, H0=H0,
                        T=T, Hdrift=r['Hdrift'])
    return r


def traj():
    out = {}
    # chaotic and regular orbits from SAF's initial conditions, eps=0.5
    x, y = saf_ic([1, 2, 5, 20])
    for eps in (0.0, 0.5):
        r = rc.integrate(rc.probs_to_logits(x, y), eps, -eps, h=0.005, T=4000.0, every=200,
                         traj_every=4, ntraj=200000)
        out[f'traj_{eps:.2f}'] = r['traj'].astype(np.float32); out[f'lyap_{eps:.2f}'] = r['lyap']
    # butterfly: 16 orbits started 1e-9 apart (chaotic IC k=1, eps=0.5)
    x, y = saf_ic([1])
    s0 = rc.probs_to_logits(x, y)[0]
    s0s = s0[None] + 1e-9 * np.arange(16)[:, None] * np.array([1, 0, 0, 0])[None]
    r = rc.integrate(s0s, 0.5, -0.5, h=0.005, T=1500.0, every=200, traj_every=10, ntraj=30000)
    out['butterfly'] = r['traj'].astype(np.float32)
    # long single line for the plotter drawing: k=1, eps=0.5
    x, y = saf_ic([1])
    r = rc.integrate(rc.probs_to_logits(x, y), 0.5, -0.5, h=0.005, T=20000.0, every=200,
                     traj_every=10, ntraj=400000)
    out['plotter'] = r['traj'][0].astype(np.float32); out['plotter_lyap'] = r['lyap']
    np.savez_compressed(f'{CACHE}/traj.npz', **out)
    print('traj saved; lyap', out['lyap_0.50'], out['lyap_0.00'], out['plotter_lyap'])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('repro', 'all') and not __import__('os').path.exists(f'{CACHE}/saf_repro.npz'):
        repro()
    if what in ('kam', 'all'):
        # H0 = 2.8 sits between SAF's k=3 (2.807) and k=4 orbits: a mixed phase space at eps=0.5
        import os
        for eps in (0.0, 0.1, 0.25, 0.4, 0.5):
            if not os.path.exists(f'{CACHE}/kam_eps{eps:.2f}.npz'):
                kam(eps, 2.8, n=40, T=40000.0)
        if not os.path.exists(f'{CACHE}/kam_eps0.50_H3.0.npz'):
            kam(0.5, 3.0, n=40, T=40000.0, name='kam_eps0.50_H3.0')
    if what in ('traj', 'all'):
        traj()
    if what == 'energies':
        for k in range(1, 26, 2):
            x, y = saf_ic([k]); print(k, rc.energy(x, y))
