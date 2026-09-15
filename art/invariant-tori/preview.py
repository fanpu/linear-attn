"""M1 previews (matplotlib only, cache/preview/; not gallery material).  Reads cache/stereo_*.npz,
cache/density_eps05.npz, cache/membrane.npz.   python preview.py [tori|sea|all]

Declared: colour = torus index (viridis, 0 = innermost seed) or orbit identity (tab10); chaotic
points grey.  Axes are the stereographic chart coordinates X0, X1, X2 (pole in cache/pole.json).
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(CACHE, 'preview')
VIEWS = [(20, -60), (70, 30), (-10, 120)]   # (elev, azim)


def pole_caption():
    P = json.load(open(os.path.join(CACHE, 'pole.json')))
    return f'pole min angle {P["min_angle_deg"]:.2f} deg, chart scale {P["chart_scale_min"]:.2f}-{P["chart_scale_max"]:.1f}'


def set_equal(ax, X):
    c = 0.5 * (X.max(0) + X.min(0)); r = 0.5 * (X.max(0) - X.min(0)).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def tori():
    d = np.load(os.path.join(CACHE, 'stereo_eps0.npz')); X = d['X']; n = len(X)
    cols = plt.cm.viridis(np.linspace(0, 0.95, n))
    stride = 4
    fig = plt.figure(figsize=(18, 6.4))
    for k, (el, az) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        for i in range(n):
            Y = X[i, :60000:stride]
            ax.plot(*Y.T, lw=0.15, color=cols[i], alpha=0.5)
        set_equal(ax, X[:, ::50].reshape(-1, 3)); ax.view_init(el, az)
        ax.set_xlabel('X0'); ax.set_ylabel('X1'); ax.set_zlabel('X2'); ax.set_title(f'elev {el}, azim {az}')
    fig.suptitle('eps = 0, H = 2.8: 12 orbits (t <= 6000) in the stereographic chart; colour = seed index '
                 '(inner -> outer, viridis)\n' + pole_caption())
    fig.savefig(os.path.join(OUT, 'tori_eps0_views.png'), dpi=110, bbox_inches='tight'); plt.close(fig)

    # each torus alone (small multiples, same view and limits)
    fig = plt.figure(figsize=(16, 12))
    lim = X[:, ::50].reshape(-1, 3)
    for i in range(n):
        ax = fig.add_subplot(3, 4, i + 1, projection='3d')
        ax.plot(*X[i, :40000:2].T, lw=0.15, color=cols[i])
        set_equal(ax, lim); ax.view_init(*VIEWS[0]); ax.set_title(f'torus {i}, lambda={d["lyap"][i]:.1e}')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.savefig(os.path.join(OUT, 'tori_eps0_each.png'), dpi=80, bbox_inches='tight'); plt.close(fig)

    # planar slices: all orbit samples within a thin slab around the plane through the core
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))
    c = X[0].mean(0)
    for a in range(3):
        other = [j for j in range(3) if j != a]
        w = 0.004 * (X[:, :, a].max() - X[:, :, a].min())
        ax = axs[a]
        for i in range(n):
            m = np.abs(X[i, :, a] - c[a]) < w
            ax.plot(X[i, m, other[0]], X[i, m, other[1]], '.', ms=0.8, color=cols[i])
        ax.set_aspect('equal'); ax.set_xlabel(f'X{other[0]}'); ax.set_ylabel(f'X{other[1]}')
        ax.set_title(f'slab |X{a} - {c[a]:.3f}| < {w:.4f}')
    fig.suptitle('eps = 0: slab cuts through the innermost torus centre. Nested tori -> nested closed curves. '
                 + pole_caption())
    fig.savefig(os.path.join(OUT, 'tori_eps0_slices.png'), dpi=110, bbox_inches='tight'); plt.close(fig)


def sea():
    s = np.load(os.path.join(CACHE, 'stereo_eps05.npz'))
    Xc = s['chaotic_X'][0]; Xr = s['regular_X']; kam = s['kam_index'][1:]
    dz = np.load(os.path.join(CACHE, 'density_eps05.npz')); lo, hi = dz['lo'], dz['hi']
    rng = np.random.default_rng(0)
    sub = Xc[rng.choice(len(Xc), 80000, replace=False)]
    cols = plt.cm.tab10(np.arange(len(Xr)))
    fig = plt.figure(figsize=(18, 6.6))
    for k, (el, az) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.scatter(*sub[:80000].T, s=0.6, c='0.3', alpha=0.25, lw=0, depthshade=False)
        for i in range(len(Xr)):
            Y = Xr[i, :12000:2]
            Y = np.where(np.all((Y >= lo) & (Y <= hi), 1)[:, None], Y, np.nan)   # clip to the box
            ax.plot(*Y.T, lw=0.25, color=cols[i], alpha=0.9,
                    label=f'kam {kam[i]} lambda={s["regular_lyap"][i]:.1e}')
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect((1, 1, 1))
        ax.view_init(el, az); ax.set_title(f'elev {el}, azim {az}')
        if k == 0:
            ax.legend(fontsize=6, loc='upper left')
    fig.suptitle(f'eps = 0.5, H = 2.8: chaotic orbit (grey, lambda={s["chaotic_lyap"][0]:.4f}, 80k of 4M samples) '
                 f'+ 8 regular orbits (tab10, t <= 1200), clipped to the density box\n' + pole_caption())
    fig.savefig(os.path.join(OUT, 'sea_eps05_views.png'), dpi=110, bbox_inches='tight'); plt.close(fig)

    # slice atlas: density (log1p counts), membrane g = 0 contour, regular orbits in the slab
    mem = np.load(os.path.join(CACHE, 'membrane.npz'))
    D = dz['counts'].astype(float); R = D.shape[0]; g = mem['g']; Rm = g.shape[0]; M05 = mem['membrane_eps05']
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for a in range(3):
        other = [j for j in range(3) if j != a]
        for row, frac in enumerate((0.5, 0.35)):
            ax = axs[row, a]
            iD = int(frac * R); im = int(frac * Rm)
            sl = np.take(D, iD, axis=a)
            ext = [lo[other[0]], hi[other[0]], lo[other[1]], hi[other[1]]]
            ax.imshow(np.log1p(sl).T, origin='lower', extent=ext, cmap='magma', interpolation='nearest')
            gs = np.take(g, im, axis=a); ms = np.isfinite(np.take(M05, im, axis=a))
            ax.contour(np.linspace(ext[0], ext[1], Rm), np.linspace(ext[2], ext[3], Rm), gs.T, levels=[0],
                       colors='cyan', linewidths=0.6)
            yy, xx = np.nonzero(ms.T)
            ax.plot(ext[0] + (xx + 0.5) * (ext[1] - ext[0]) / Rm, ext[2] + (yy + 0.5) * (ext[3] - ext[2]) / Rm,
                    's', ms=1.2, color='cyan', alpha=0.5)
            zc = lo[a] + (iD + 0.5) * (hi[a] - lo[a]) / R; w = (hi[a] - lo[a]) / R * 2
            for i in range(len(Xr)):
                m = np.abs(Xr[i, :, a] - zc) < w
                ax.plot(Xr[i, m, other[0]], Xr[i, m, other[1]], '.', ms=0.6, color=cols[i])
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(f'X{a} = {zc:.3f}: log1p density (magma), g=0 (cyan line), membrane voxels '
                         f'g-dot>0 (cyan squares)', fontsize=8)
    fig.suptitle('eps = 0.5 slice atlas: visit density of the chaotic orbit (256^3, T=1.2e6) with the '
                 'Poincare-section membrane and regular orbits (tab10) in a 2-voxel slab')
    fig.savefig(os.path.join(OUT, 'sea_eps05_slices.png'), dpi=100, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ['all']
    if 'tori' in what or 'all' in what:
        tori()
    if 'sea' in what or 'all' in what:
        sea()
