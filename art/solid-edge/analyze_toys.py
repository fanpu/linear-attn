"""M1 structure tests for the 64^3 toys and the null (CPU; reads cache/toys/*.npz).

  OMP_NUM_THREADS=4 ../.venv/bin/python analyze_toys.py

Writes cache/toys_summary.json and cache/preview/toy_<t>_{slices,projections}.png,
cache/preview/toy_a_sigma1_vs_overview.png, cache/preview/toys_boxcount.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

from se_analysis import boundary3d, dimension3d, edges, local_slopes

HERE = os.path.dirname(os.path.abspath(__file__))
SRCW = '/home/fzeng/ml/research/art/trainability-fractal/cache/windows'
PV = os.path.join(HERE, 'cache', 'preview')
os.makedirs(PV, exist_ok=True)
CM = matplotlib.colors.ListedColormap(['#c8324a', '#2b4f8c'])   # diverged, converged (declared)
TOYS = [('a', 'toy_a_64_float32'), ('b', 'toy_b_64_float32'), ('null', 'toy_null_64_float32')]
AX = {'a': ['log10 sigma', 'log10 eta1', 'log10 eta0'], 'b': ['log10 eta2', 'log10 eta1', 'log10 eta0'],
      'null': ['log10 eta2 (block c)', 'log10 eta1 (block b)', 'log10 eta0 (block a)']}

summary = {}
box = {}
for t, name in TOYS:
    fn = os.path.join(HERE, 'cache', 'toys', f'{name}.npz')
    if not os.path.exists(fn):
        print('missing', fn)
        continue
    d = np.load(fn)
    meta = json.loads(str(d['meta']))
    M = d['measure']
    L = (M < 0).astype(np.uint8)
    R = L.shape[0]
    B6 = boundary3d(L)
    fit, s, c, E = dimension3d(L, 1, 16)
    fit2, *_ = dimension3d(L, 2, 16)
    box[t] = (s, c)
    coords = [d['third_log'], d['log10_eta'], d['log10_eta']]
    per_axis_edges = [E.sum(axis=tuple(a for a in range(3) if a != ax)) for ax in range(3)]
    summary[t] = dict(name=name, kind=meta['kind'], dtype=meta['dtype'], seconds=meta['seconds'],
                      px_per_s=meta['px_per_s'], px_per_s_excl_first=meta['px_per_s_excl_first'],
                      chunk=meta['chunk'], chunk_seconds=meta['chunk_seconds'],
                      trainable_frac=float(L.mean()), boundary_voxels_6nbr=int(B6.sum()),
                      boundary_voxel_frac=float(B6.mean()), edge_cells_2x2x2=int(E.sum()),
                      box_sizes=s.tolist(), box_counts=c.tolist(), local_slopes=local_slopes(s, c).tolist(),
                      D3_b1_16=fit, D3_b2_16=fit2,
                      conv_frac_along_axis=[L.mean(axis=tuple(a for a in range(3) if a != ax)).tolist() for ax in range(3)],
                      flag='b = 1-16 is 1.2 decades (< 2): preview only, not a dimension claim')
    # --- per-slice 2D D along each axis (median over slices that hold enough boundary) ---
    d2 = []
    for ax in range(3):
        vals = []
        for k in range(R):
            sl = np.take(M, k, axis=ax)
            if edges(sl).sum() >= 32:
                from se_analysis import dimension_of_measure
                f, _, _ = dimension_of_measure(sl, 1, 16)
                vals.append(f['D'])
        d2.append(dict(n=len(vals), median=float(np.median(vals)) if vals else None))
    summary[t]['D2_axis_slices_b1_16'] = d2
    # --- six axis slices: per axis, the plane with most edge cells and a second plane ---
    planes = []
    for ax in range(3):
        kmax = int(np.argmax(per_axis_edges[ax]))
        k2 = meta['sigma_plane'] if (t == 'a' and ax == 0) else (R // 2 if abs(R // 2 - kmax) > 4 else R // 4)
        planes += [(ax, k2), (ax, kmax)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.8), dpi=100)
    for n_, (ax, k) in enumerate(planes):
        a = axs[n_ % 2, ax]
        other = [q for q in range(3) if q != ax]
        img = np.take(L, k, axis=ax)
        yv, xv = coords[other[0]], coords[other[1]]
        a.imshow(img, origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', aspect='auto',
                 extent=[xv[0], xv[-1], yv[0], yv[-1]])
        a.set_xlabel(AX[t][other[1]]); a.set_ylabel(AX[t][other[0]])
        a.set_title(f'{AX[t][ax]} = {coords[ax][k]:.3f} (plane {k}), conv {100*img.mean():.0f}%', fontsize=9)
    fig.suptitle(f'toy {t} ({meta["kind"]}, {R}^3 {meta["dtype"]}): axis slices, blue = converged, red = diverged; '
                 f'D3(b=1-16) = {fit["D"]:.2f}', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(PV, f'toy_{t}_slices.png')); plt.close(fig)
    summary[t]['preview_planes'] = [dict(axis=AX[t][ax], index=int(k), coord=float(coords[ax][k])) for ax, k in planes]
    # --- projections: converged fraction along each axis (declared: a mean, not a slice) ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), dpi=100)
    for ax in range(3):
        other = [q for q in range(3) if q != ax]
        yv, xv = coords[other[0]], coords[other[1]]
        im = axs[ax].imshow(L.mean(axis=ax), origin='lower', cmap='cividis', vmin=0, vmax=1, interpolation='nearest',
                            aspect='auto', extent=[xv[0], xv[-1], yv[0], yv[-1]])
        axs[ax].set_xlabel(AX[t][other[1]]); axs[ax].set_ylabel(AX[t][other[0]])
        axs[ax].set_title(f'converged fraction along {AX[t][ax]}', fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(os.path.join(PV, f'toy_{t}_projections.png')); plt.close(fig)

    # --- slice reproduction (spec §0.7) for toy a at sigma = 1 ---
    if t == 'a':
        k1 = meta['sigma_plane']
        pix = d['pix']
        toy = M[k1]
        ref64 = np.load(f'{SRCW}/hero_overview_tanh_1024_f64.npz')['measure']
        ref32 = np.load(f'{SRCW}/hero_overview_tanh_1024_f32.npz')['measure']
        E64 = edges(ref64)                                       # (1023,1023) his edge cells
        Epad = np.zeros_like(ref64, bool); Epad[:-1, :-1] = E64
        near = maximum_filter(Epad.astype(np.uint8), size=33)[np.ix_(pix, pix)] > 0   # within 16 px = one toy voxel
        sub64 = ref64[np.ix_(pix, pix)]
        sub32 = ref32[np.ix_(pix, pix)]
        agree = np.sign(toy) == np.sign(sub64)
        res = dict(sigma_plane=int(k1), pixels='overview pixel (16i+8, 16j+8)', n=int(agree.size),
                   agreement=float(agree.mean()), disagreements=int((~agree).sum()),
                   near_boundary_def='an f64 overview edge cell lies within Chebyshev distance 16 px (one toy voxel)',
                   n_near=int(near.sum()), disagree_near=int((~agree & near).sum()),
                   disagree_rate_near=float((~agree & near).sum() / max(1, near.sum())),
                   n_far=int((~near).sum()), disagree_far=int((~agree & ~near).sum()),
                   agreement_vs_f32_overview=float((np.sign(toy) == np.sign(sub32)).mean()),
                   f32_overview_vs_f64_overview_at_these_pixels=float((np.sign(sub32) == np.sign(sub64)).mean()),
                   conv_toy=float((toy < 0).mean()), conv_overview_sub=float((sub64 < 0).mean()))
        fp = os.path.join(HERE, 'cache', 'toys', 'toy_a_64_float64_plane.npz')
        if os.path.exists(fp):
            pl = np.load(fp)['measure']
            res['plane_f64_recompute_vs_overview_f64'] = float((np.sign(pl) == np.sign(sub64)).mean())
            res['plane_f64_recompute_vs_toy_f32'] = float((np.sign(pl) == np.sign(toy)).mean())
            res['plane_f64_disagree_near'] = int(((np.sign(pl) != np.sign(sub64)) & near).sum())
            res['plane_f64_disagree_far'] = int(((np.sign(pl) != np.sign(sub64)) & ~near).sum())
            res['plane_f64_px_per_s'] = json.loads(str(np.load(fp)['meta']))['px_per_s']
        summary[t]['slice_agreement'] = res
        print('slice agreement', json.dumps(res, indent=1))
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), dpi=100)
        ev = d['log10_eta']; ext = [ev[0], ev[-1], ev[0], ev[-1]]
        axs[0].imshow((sub64 < 0), origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', extent=ext)
        axs[0].set_title('float64 1024^2 overview at every 16th pixel', fontsize=9)
        axs[1].imshow(L[k1], origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', extent=ext)
        axs[1].set_title(f'toy a, sigma = 1 plane (float32, 64^3 run)', fontsize=9)
        dis = np.zeros(toy.shape + (3,)); dis[:] = 0.93
        dis[near] = (0.75, 0.75, 0.75); dis[~agree] = (0.1, 0.1, 0.1)
        axs[2].imshow(dis, origin='lower', interpolation='nearest', extent=ext)
        axs[2].set_title(f'disagreements (black) {res["disagreements"]}/{res["n"]}; grey = near boundary', fontsize=9)
        for a in axs:
            a.set_xlabel('log10 eta0'); a.set_ylabel('log10 eta1')
        fig.tight_layout(); fig.savefig(os.path.join(PV, 'toy_a_sigma1_vs_overview.png')); plt.close(fig)

if box:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=100)
    for (t, (s, c)), col in zip(box.items(), ['#2b4f8c', '#2e8b57', '#8a847c']):
        axs[0].loglog(s, c, 'o-', color=col, label=f'{t}: D3(1-16) = {summary[t]["D3_b1_16"]["D"]:.2f}')
        axs[1].semilogx(np.sqrt(s[1:] * s[:-1]), local_slopes(s, c), 'o-', color=col, label=t)
    axs[0].set_xlabel('box side b (voxels)'); axs[0].set_ylabel('occupied boxes'); axs[0].legend(fontsize=8)
    axs[1].axhline(2, color='k', lw=0.5); axs[1].set_xlabel('b'); axs[1].set_ylabel('local slope'); axs[1].legend(fontsize=8)
    fig.suptitle('3D box counting on 2x2x2 edge cells, 64^3 (1.2 decades: preview only)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(PV, 'toys_boxcount.png')); plt.close(fig)

json.dump(summary, open(os.path.join(HERE, 'cache', 'toys_summary.json'), 'w'), indent=1)
for t, v in summary.items():
    print(t, f'conv={v["trainable_frac"]:.4f} B6={v["boundary_voxels_6nbr"]} edge_cells={v["edge_cells_2x2x2"]} '
             f'D3(1-16)={v["D3_b1_16"]["D"]:.3f}±{v["D3_b1_16"]["se"]:.3f} D3(2-16)={v["D3_b2_16"]["D"]:.3f} '
             f'slopes={np.round(v["local_slopes"], 2).tolist()} D2med={[x["median"] for x in v["D2_axis_slices_b1_16"]]} '
             f'{v["px_per_s"]:.0f} px/s')
