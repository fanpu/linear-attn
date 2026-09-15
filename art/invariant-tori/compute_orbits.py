"""Orbits for Invariant Tori (float64, C RK4 integrator in logits, h = 0.01), all on H = 2.8.

  cache/orbits_eps0.npz    12 orbits at eps = 0 seeded on a ray of the Poincare section from the
                           elliptic fixed point (the core periodic orbit) to the section fold
  cache/orbits_eps05.npz   eps = 0.5: one chaotic orbit to T = 2e5 and 8 regular island orbits
  cache/sweep/eps_*.npz    26 eps in [0, 0.5], the same 12 starts (one file per eps; resumable)

Each file stores float64 logits L (n, m, 4) sampled every dt (the start point included), the
largest finite-time Lyapunov exponent `lyap` (Benettin, renormalised every 100 steps), its running
history, section crossings `sec` (probabilities, SAF section g = x_P - x_R + y_P - y_R = 0 upward).

python compute_orbits.py [eps0|eps05|sweep|all]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

import replicator_c as rc
from section import section_seed, first_returns

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
H0 = 2.8
H_STEP = 0.01
N_TORI = 12
SWEEP_EPS = np.linspace(0.0, 0.5, 26)
GC_KAM05 = os.path.join(HERE, '..', 'game-chaos', 'cache', 'kam_eps0.50.npz')
# Regular eps = 0.5 orbits: indices into game-chaos kam_eps0.50.npz, chosen by farthest-point sampling
# (1 - IoU of filled section masks) among orbits with lambda <= 2.5e-3 at T = 4e4 and filled
# section area in (0.003, 0.05) of the (x_R, y_P) square [0, 0.8]^2.  See NOTES.md Decision.
REGULAR_KAM05 = [38, 2, 59, 84, 121, 201, 243, 347]
CHAOTIC_KAM05 = 190  # largest lambda (0.0287) of the 395 game-chaos orbits at eps = 0.5, H = 2.8


def run(s0, eps, T, dt, maxsec, nhist=200):
    """Integrate from logits s0 (n, 4); prepend the start so L[:, 0] = s0."""
    every_traj = int(round(dt / H_STEP)); nsteps = int(round(T / H_STEP))
    ntraj = nsteps // every_traj
    hist_every = nsteps // nhist
    r = rc.integrate(s0, eps, -eps, h=H_STEP, T=T, every=100, traj_every=every_traj, ntraj=ntraj,
                     maxsec=maxsec, hist_every=hist_every, nhist=nhist)
    L = np.concatenate([np.asarray(s0, np.float64)[:, None], r['traj']], 1)
    return dict(L=L, lyap=r['lyap'], lyap_hist=r['lyap_hist'], sec=r['sec'], nsec=r['nsec'],
                Hdrift_c=r['Hdrift'], eps=eps, T=T, dt=dt, h=H_STEP, H0=H0)


def core_fixed_point(eps=0.0):
    """Elliptic fixed point of the eps = 0 return map, on the diagonal x = y, section root 1."""
    def obj(v):
        s = section_seed(v[0], v[1], H0, eps, root=1)
        if s is None:
            return 1.0
        ret, n = first_returns(s[0], s[1], eps)
        return 1.0 if n[0] < 1 else np.linalg.norm(ret[0, 0] - np.concatenate(s))
    r = minimize(obj, [0.4557, 0.4557], method='Nelder-Mead',
                 options=dict(xatol=1e-11, fatol=1e-15, maxiter=600))
    return r.x, r.fun


def eps0_seeds():
    c, resid = core_fixed_point()
    d = np.array([1.0, 1.0]) / np.sqrt(2)          # toward the upper-right tip of the plate
    lo, hi = 0.0, 0.8
    for _ in range(50):                             # the section fold along the ray (root 1 ends)
        m = 0.5 * (lo + hi); p, q = c + m * d
        ok = 0 < p < 1 and 0 < q < 1 and section_seed(p, q, H0, 0.0, root=1) is not None
        lo, hi = (m, hi) if ok else (lo, m)
    fold = lo
    radii = fold * np.arange(1, N_TORI + 1) / (N_TORI + 1)
    seeds = [section_seed(*(c + r * d), H0, 0.0, root=1) for r in radii]
    x = np.array([s[0] for s in seeds]); y = np.array([s[1] for s in seeds])
    return dict(centre_pq=c, centre_residual=resid, fold_r=fold, radii=radii, x0=x, y0=y,
                s0=rc.probs_to_logits(x, y))


def do_eps0():
    t = time.time()
    sd = eps0_seeds()
    r = run(sd['s0'], 0.0, T=20000.0, dt=0.1, maxsec=4000)
    np.savez(os.path.join(CACHE, 'orbits_eps0.npz'), **r, **sd)
    print(f'eps0: centre {sd["centre_pq"]} (resid {sd["centre_residual"]:.1e}), fold r {sd["fold_r"]:.4f}; '
          f'lambda*1e3 {np.round(r["lyap"] * 1e3, 3)}; {time.time() - t:.0f}s', flush=True)
    return sd


def do_eps05():
    t = time.time()
    k = np.load(GC_KAM05)
    xs, ys = k['x0'], k['y0']
    idx = [CHAOTIC_KAM05] + REGULAR_KAM05
    s0 = rc.probs_to_logits(xs[idx], ys[idx])
    ch = run(s0[:1], 0.5, T=200000.0, dt=0.05, maxsec=40000, nhist=400)
    print(f'eps05 chaotic: lambda {ch["lyap"][0]:.5f}, {ch["nsec"][0]} crossings, {time.time() - t:.0f}s', flush=True)
    rg = run(s0[1:], 0.5, T=20000.0, dt=0.1, maxsec=4000)
    out = {f'chaotic_{k_}': v for k_, v in ch.items()}
    out.update({f'regular_{k_}': v for k_, v in rg.items()})
    np.savez(os.path.join(CACHE, 'orbits_eps05.npz'), kam_index=np.array(idx), **out)
    print(f'eps05 regular: lambda*1e3 {np.round(rg["lyap"] * 1e3, 3)}; total {time.time() - t:.0f}s', flush=True)


def do_sweep():
    sd = np.load(os.path.join(CACHE, 'orbits_eps0.npz'))
    s0 = sd['s0']
    os.makedirs(os.path.join(CACHE, 'sweep'), exist_ok=True)
    for i, eps in enumerate(SWEEP_EPS):
        f = os.path.join(CACHE, 'sweep', f'eps_{i:02d}.npz')
        if os.path.exists(f):
            continue
        t = time.time()
        r = run(s0, float(eps), T=10000.0, dt=0.2, maxsec=2000)
        np.savez(f + '.tmp.npz', **r); os.replace(f + '.tmp.npz', f)
        print(f'sweep eps={eps:.2f}: chaotic(>5e-3) {int((r["lyap"] > 5e-3).sum())}/12, '
              f'max lambda {r["lyap"].max():.4f}, {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(CACHE, exist_ok=True)
    if what in ('eps0', 'all'):
        do_eps0()
    if what in ('eps05', 'all'):
        do_eps05()
    if what in ('sweep', 'all'):
        do_sweep()
