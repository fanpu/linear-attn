"""Lyapunov planes of discrete experience-weighted attraction (Galla & Farmer 2013) on the
Sato-Akiyama-Farmer generalised rock-paper-scissors game.
  Q' = (1-alpha) Q + beta * A(eps) y,   x = softmax(Q)   (and symmetrically for player 2)
Largest Lyapunov exponent via exact tangent propagation, float64. Also stores min strategy
probability on the attractor (orbits hugging the simplex boundary = near-pure best-response cycling).
python compute_lyap_rps.py [name ...]
"""
import sys
import time

import numpy as np
import torch

from gamelib import DT, gpu_setup, lyap_rps_ewa

# name: (x-axis param, x range, y-axis param, y range, fixed dict, R)
PLATES = {
    'rps_beta_eps_a0.30': ('beta', (0.05, 12.0), 'eps', (-1.0, 1.0), dict(alpha=0.3, zerosum=True), 1000),
    'rps_beta_eps_a0.30_zoom': ('beta', (2.5, 9.0), 'eps', (-0.6, 0.6), dict(alpha=0.3, zerosum=True), 1000),
    'rps_beta_alpha_e0.50': ('beta', (0.05, 12.0), 'alpha', (0.0, 0.5), dict(eps=0.5, zerosum=True), 1000),
    'rps_beta_eps_a0.02': ('beta', (0.05, 12.0), 'eps', (-1.0, 1.0), dict(alpha=0.02, zerosum=True), 1000),
    'rps_beta_eps_a0.02_detail': ('beta', (6.0, 8.0), 'eps', (0.2, 0.6), dict(alpha=0.02, zerosum=True), 1000),
}


def run(name, T0=3000, T1=5000):
    xk, xr, yk, yr, fix, R = PLATES[name]
    xs = torch.linspace(*xr, R, dtype=DT); ys = torch.linspace(*yr, R, dtype=DT)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    p = dict(fix); p[xk] = X; p[yk] = Y
    eps = p['eps']; ex, ey = (eps, -eps) if p.pop('zerosum') else (eps, eps)
    t = time.time()
    L, xmin = lyap_rps_ewa(p['beta'], p['alpha'], ex, ey, T0=T0, T1=T1, return_state=True, chunk=1 << 19)
    np.savez_compressed(f'cache/{name}.npz', L=L.astype(np.float32), xmin=xmin.astype(np.float32),
                        x=xs.numpy(), y=ys.numpy(), xk=xk, yk=yk, fixed=str(fix), T0=T0, T1=T1)
    print(f'{name}: {time.time()-t:.0f}s  frac(L>3e-3)={np.mean(L>3e-3):.4f} max={L.max():.3f} '
          f'frac(xmin<1e-6)={np.mean(xmin<1e-6):.3f}', flush=True)


if __name__ == '__main__':
    gpu_setup()
    for n in (sys.argv[1:] or list(PLATES)):
        run(n)
