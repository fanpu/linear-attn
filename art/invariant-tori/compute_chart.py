"""Chart products for Invariant Tori (reads cache/orbits_*.npz and cache/sweep/*.npz).

  cache/pole.json          pole p in S^3 (maximises the minimum angle to every stored orbit point),
                           the angle, the runner-up poles, chart scale statistics
  cache/stereo_eps0.npz    X (12, m, 3) float64 stereographic coordinates, + section dots
  cache/stereo_eps05.npz   chaotic X (1, m, 3), regular X (8, m, 3), + section dots
  cache/stereo_sweep.npz   X (26, 12, m, 3) float32 (render only; the float64 logits stay in sweep/)
  cache/density_eps05.npz  visit density of the chaotic orbit, 256^3, over the per-axis 0.5-99.5
                           percentile box of its chart coordinates
  cache/membrane.npz       g = x_P - x_R + y_P - y_R on a 128^3 cell-centred grid over the same box
                           via the inverse chart at h = 2.8; membrane field (NaN off the level set)

python compute_chart.py [pole|stereo|density|membrane|all]
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import chart as C

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
H0 = 2.8


def load_all_logits(exclude_kam=()):
    """Every stored orbit (list of (name, L (n, m, 4))), optionally without some regular eps=0.5 orbits
    (identified by their game-chaos kam_eps0.50 index)."""
    out = []
    d = np.load(os.path.join(CACHE, 'orbits_eps0.npz')); out.append(('eps0', d['L']))
    d = np.load(os.path.join(CACHE, 'orbits_eps05.npz'))
    keep = [i for i, k in enumerate(d['kam_index'][1:]) if int(k) not in set(exclude_kam)]
    out.append(('eps05_chaotic', d['chaotic_L'])); out.append(('eps05_regular', d['regular_L'][keep]))
    for f in sorted(glob.glob(os.path.join(CACHE, 'sweep', 'eps_*.npz'))):
        out.append(('sweep_' + os.path.basename(f)[4:6], np.load(f)['L']))
    return out


def chord_to_angle(c):
    return 2.0 * np.arcsin(np.clip(c / 2.0, 0, 1))


def do_pole(n_cand=400000, n_refine=24, seed=0, exclude_kam=(), out_name='pole.json'):
    t0 = time.time()
    groups = load_all_logits(exclude_kam)
    Q = []; step = 0.0
    for name, L in groups:
        q = C.to_sphere(C.logits_to_u(L))
        step = max(step, np.linalg.norm(np.diff(q, axis=1), axis=-1).max())
        Q.append(q.reshape(-1, 4))
    Q = np.concatenate(Q)
    print(f'pole: {len(Q)} points, max chord between samples {step:.4f}; building tree', flush=True)
    tree = cKDTree(Q)
    rng = np.random.default_rng(seed)
    cand = C.to_sphere(rng.normal(size=(n_cand, 4)))
    dist, _ = tree.query(cand, k=1, workers=4)
    top = cand[np.argsort(-dist)[:n_refine * 20]]
    # keep distinct candidates (angle > 20 deg apart) so refinement explores separate holes
    keep = []
    for c in top:
        if all(np.dot(c, k) < np.cos(np.radians(20)) for k in keep):
            keep.append(c)
        if len(keep) == n_refine:
            break
    res = []
    for c in keep:
        B = C.pole_basis(c)
        f = lambda v: -tree.query(C.to_sphere(c + B @ v), k=1)[0]
        r = minimize(f, np.zeros(3), method='Nelder-Mead',
                     options=dict(initial_simplex=np.vstack([np.zeros(3), 0.05 * np.eye(3)]),
                                  xatol=1e-7, fatol=1e-10, maxiter=2000))
        p = C.to_sphere(c + B @ r.x)
        res.append((-r.fun, p))
    res.sort(key=lambda z: -z[0])
    best_chord, p = res[0]
    ang = chord_to_angle(best_chord)
    ang_lb = chord_to_angle(max(best_chord - step / 2, 0))
    per_group = {}; seg_chord = np.inf
    for name, L in groups:
        q = C.to_sphere(C.logits_to_u(L))
        per_group[name] = float(np.degrees(np.arccos(np.clip(q @ p, -1, 1)).min()))
        # distance from p to the chords between consecutive samples: the orbit between samples
        a = q[:, :-1]; ab = q[:, 1:] - a
        tt = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300), 0, 1)
        seg_chord = min(seg_chord, np.linalg.norm(a + tt[..., None] * ab - p, axis=-1).min())
    ch = C.Chart(p, H0)
    Xs = [ch.forward_logits(L).reshape(-1, 3) for _, L in groups]
    scale = np.concatenate([C.stereo_scale(X) for X in Xs])
    rad = np.concatenate([np.linalg.norm(X, axis=1) for X in Xs])
    out = dict(pole=p.tolist(), min_angle_deg=float(np.degrees(ang)),
               min_angle_deg_lower_bound_between_samples=float(np.degrees(ang_lb)),
               min_angle_deg_to_sample_chords=float(np.degrees(chord_to_angle(seg_chord))),
               max_sample_chord=float(step), n_points=int(len(Q)),
               per_group_min_angle_deg=per_group,
               chart_scale_min=float(scale.min()), chart_scale_max=float(scale.max()),
               chart_scale_p99=float(np.quantile(scale, 0.99)), chart_radius_max=float(rad.max()),
               runner_up=[dict(pole=q_.tolist(), min_angle_deg=float(np.degrees(chord_to_angle(c_))),
                               angle_to_best_deg=float(np.degrees(np.arccos(np.clip(q_ @ p, -1, 1)))))
                          for c_, q_ in res[1:6]],
               seconds=time.time() - t0)
    json.dump(out, open(os.path.join(CACHE, out_name), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'runner_up'}, indent=1), flush=True)


def get_chart():
    return C.Chart(np.array(json.load(open(os.path.join(CACHE, 'pole.json')))['pole']), H0)


def sec_to_X(ch, sec, nsec):
    pts = [sec[o, :nsec[o]] for o in range(len(nsec))]
    idx = np.concatenate([np.full(len(s), o) for o, s in enumerate(pts)])
    P = np.concatenate(pts)
    return ch.forward_strategies(P[:, :3], P[:, 3:]), idx


def do_stereo():
    t0 = time.time(); ch = get_chart()
    d = np.load(os.path.join(CACHE, 'orbits_eps0.npz'))
    Xs, si = sec_to_X(ch, d['sec'], d['nsec'])
    np.savez(os.path.join(CACHE, 'stereo_eps0.npz'), X=ch.forward_logits(d['L']), sec_X=Xs, sec_orbit=si,
             lyap=d['lyap'], dt=d['dt'], radii=d['radii'])
    d = np.load(os.path.join(CACHE, 'orbits_eps05.npz'))
    cXs, _ = sec_to_X(ch, d['chaotic_sec'], d['chaotic_nsec'])
    rXs, ri = sec_to_X(ch, d['regular_sec'], d['regular_nsec'])
    np.savez(os.path.join(CACHE, 'stereo_eps05.npz'), chaotic_X=ch.forward_logits(d['chaotic_L']),
             regular_X=ch.forward_logits(d['regular_L']), chaotic_sec_X=cXs, regular_sec_X=rXs,
             regular_sec_orbit=ri, chaotic_lyap=d['chaotic_lyap'], regular_lyap=d['regular_lyap'],
             chaotic_dt=d['chaotic_dt'], regular_dt=d['regular_dt'], kam_index=d['kam_index'])
    files = sorted(glob.glob(os.path.join(CACHE, 'sweep', 'eps_*.npz')))
    X = np.stack([ch.forward_logits(np.load(f)['L']).astype(np.float32) for f in files])
    lam = np.stack([np.load(f)['lyap'] for f in files]); eps = np.array([float(np.load(f)['eps']) for f in files])
    np.savez(os.path.join(CACHE, 'stereo_sweep.npz'), X=X, lyap=lam, eps=eps, dt=0.2)
    for tag in ('eps0', 'eps05'):
        f = os.path.join(CACHE, f'orbits_{tag}_fine.npz')
        if os.path.exists(f):
            np.savez(os.path.join(CACHE, f'stereo_{tag}_fine.npz'), X=ch.forward_logits(np.load(f)['L']), dt=0.01)
    print(f'stereo: {time.time() - t0:.0f}s', flush=True)


def box_of(X, pad=0.02):
    """M3 (controller fix 2): the full support of the stored T = 2e5 chaotic segment plus 2% padding per side.
    (M1/M2 used the 0.5-99.5 percentile box, which cut the fog flat at the box faces.)"""
    lo, hi = X.min(0), X.max(0)
    w = hi - lo
    return lo - pad * w, hi + pad * w


def do_density(R=256, n_extra=29):
    """Box from the stored T = 2e5 orbit.  Counts from that orbit plus n_extra bit-identical continuations
    of 2e5 each (the integrator restarts from the exact float64 logits), T_total = 6e6 by default (M2 ruling 3: 1.2e6 left the 256^3 fog Poisson-noisy).
    counts_first = the stored segment only; split-half agreement is reported as a convergence check."""
    import replicator_c as rc
    t0 = time.time(); ch = get_chart()
    s = np.load(os.path.join(CACHE, 'stereo_eps05.npz'))
    X = s['chaotic_X'][0]
    lo, hi = box_of(X)
    rng_ = list(zip(lo, hi))
    H1 = np.histogramdd(X, bins=R, range=rng_)[0]
    tot = H1.copy(); n_tot = len(X); halves = [H1.copy(), np.zeros_like(H1)]
    L = np.load(os.path.join(CACHE, 'orbits_eps05.npz'))['chaotic_L'][:, -1]
    lam_chunks = []
    for k in range(n_extra):
        r = rc.integrate(L, 0.5, -0.5, h=0.01, T=200000.0, every=100, traj_every=5, ntraj=4000000)
        L = r['traj'][:, -1]; lam_chunks.append(float(r['lyap'][0]))
        Hk = np.histogramdd(ch.forward_logits(r['traj'][0]), bins=R, range=rng_)[0]
        tot += Hk; n_tot += r['traj'].shape[1]
        halves[0 if k < (n_extra - 1) // 2 else 1] += Hk
        print(f'  density chunk {k + 1}/{n_extra}: lambda {lam_chunks[-1]:.4f}', flush=True)
    a, b = halves[0].ravel(), halves[1].ravel()
    split_corr = float(np.corrcoef(a, b)[0, 1])
    inside = tot.sum() / n_tot
    np.savez_compressed(os.path.join(CACHE, 'density_eps05.npz'), counts=tot.astype(np.uint32),
                        counts_first=H1.astype(np.uint32), lo=lo, hi=hi, R=R, n_samples=n_tot,
                        T_total=200000.0 * (1 + n_extra), frac_inside=inside, dt=0.05,
                        lyap_chunks=np.array(lam_chunks), split_half_corr=split_corr,
                        note='counts of time-uniform samples (dt=0.05) of the chaotic eps=0.5 orbit per voxel; '
                             'axis order (X0, X1, X2); box = full support (+2% pad) of the stored T=2e5 '
                             'segment; counts include bit-identical continuations to T_total')
    print(f'density: box lo {lo.round(3)} hi {hi.round(3)}, T_total {200000.0 * (1 + n_extra):.0f}, '
          f'{inside:.3f} of samples inside, {(tot > 0).mean():.3f} voxels occupied, mean occupied count '
          f'{tot[tot > 0].mean():.1f}, max {tot.max():.0f}, split-half corr {split_corr:.3f}, '
          f'{time.time() - t0:.0f}s', flush=True)


def do_membrane(R=128):
    t0 = time.time(); ch = get_chart()
    dz = np.load(os.path.join(CACHE, 'density_eps05.npz')); lo, hi = dz['lo'], dz['hi']
    ax = [lo[i] + (np.arange(R) + 0.5) * (hi[i] - lo[i]) / R for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    u = ch.inverse_u(G)
    x, y = C.u_to_strategies(u)
    Herr = np.abs(C.energy_u(u) - H0).max()
    g = C.section_g(x, y).reshape(R, R, R)
    gd05 = C.section_gdot(x, y, 0.5).reshape(R, R, R)
    gd0 = C.section_gdot(x, y, 0.0).reshape(R, R, R)
    spacing = (hi - lo) / R
    grad = np.gradient(g, *spacing)
    gn = np.sqrt(sum(gi ** 2 for gi in grad))
    sdist = g / gn                                   # first-order distance to g = 0 in chart units
    band = np.abs(sdist) <= 0.5 * spacing.max()      # within half a (largest) voxel of the level set
    mem05 = np.where(band & (gd05 > 0), g, np.nan)
    mem_both = np.where(band, g, np.nan)
    # check: section dots of the eps = 0.5 orbits fall in the membrane band
    s = np.load(os.path.join(CACHE, 'stereo_eps05.npz'))
    dots = np.concatenate([s['chaotic_sec_X'], s['regular_sec_X']])
    ijk = np.floor((dots - lo) / (hi - lo) * R).astype(int)
    ok = np.all((ijk >= 0) & (ijk < R), 1)
    hit = np.isfinite(mem05[tuple(ijk[ok].T)])
    near = np.zeros(ok.sum(), bool)                  # or a face neighbour (dot sits on a voxel boundary)
    for dd in np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]):
        j = np.clip(ijk[ok] + dd, 0, R - 1); near |= np.isfinite(mem05[tuple(j.T)])
    stats = dict(H_err_max=float(Herr), band_voxels=int(band.sum()), membrane05_voxels=int(np.isfinite(mem05).sum()),
                 dots_in_box=int(ok.sum()), dots_total=int(len(dots)), dots_hit_voxel=float(hit.mean()),
                 dots_hit_voxel_or_neighbour=float((hit | near).mean()),
                 grid_H_ok=bool(Herr < 1e-9), seconds=time.time() - t0)
    np.savez_compressed(os.path.join(CACHE, 'membrane.npz'), g=g, gdot_eps05=gd05, gdot_eps0=gd0, sdist=sdist,
                        membrane_eps05=mem05, membrane_both=mem_both, lo=lo, hi=hi, R=R, **stats,
                        note='cell-centred grid, axis order (X0,X1,X2). membrane_* = g where |g|/|grad g| <= half '
                             'the largest voxel side (first-order distance to g=0), NaN elsewhere; membrane_eps05 '
                             'also requires dg/dt > 0 at eps=0.5 (the upward section used by the plates).')
    print('membrane:', json.dumps(stats), flush=True)


def do_fields256(R=256, chunk=1 << 21):
    """g, dg/dt (eps=0.5) and first-order distance to g=0 on the density grid itself (256^3, cell-centred), so the
    renderer can put fog and membrane in one multi-channel volume without resampling. float32."""
    t0 = time.time(); ch = get_chart()
    dz = np.load(os.path.join(CACHE, 'density_eps05.npz')); lo, hi = dz['lo'], dz['hi']
    ax = [lo[i] + (np.arange(R) + 0.5) * (hi[i] - lo[i]) / R for i in range(3)]
    g = np.empty(R ** 3); gd = np.empty(R ** 3); Herr = 0.0
    I, J, K = np.meshgrid(np.arange(R), np.arange(R), np.arange(R), indexing='ij')
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    for s0 in range(0, R ** 3, chunk):
        sl = slice(s0, s0 + chunk)
        G = np.stack([ax[0][I[sl]], ax[1][J[sl]], ax[2][K[sl]]], -1)
        u = ch.inverse_u(G); x, y = C.u_to_strategies(u)
        Herr = max(Herr, float(np.abs(C.energy_u(u) - H0).max()))
        g[sl] = C.section_g(x, y); gd[sl] = C.section_gdot(x, y, 0.5)
    g = g.reshape(R, R, R); gd = gd.reshape(R, R, R)
    spacing = (hi - lo) / R
    gn = np.sqrt(sum(gi ** 2 for gi in np.gradient(g, *spacing)))
    np.savez(os.path.join(CACHE, 'fields256.npz'), g=g.astype(np.float32), gdot_eps05=gd.astype(np.float32),
             sdist=(g / gn).astype(np.float32), lo=lo, hi=hi, R=R, H_err_max=Herr,
             note='cell-centred on the density grid, axis order (X0,X1,X2); sdist = g/|grad g| (chart units)')
    print(f'fields256: H err {Herr:.1e}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    what = sys.argv[1:] or ['all']
    for w, fn in (('pole', do_pole), ('stereo', do_stereo), ('density', do_density), ('membrane', do_membrane), ('fields256', do_fields256)):
        if w in what or 'all' in what:
            fn()
