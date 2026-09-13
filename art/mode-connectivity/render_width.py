"""Width series.  Measured (compute_width.py): train loss along 25-point lerps for widths 32..2048,
3 independent pairs each; naive, weight-matched, matched+REPAIR.  Planes (compute_planes.py width).

  width_ridge_<ds>_{night,paper}.png  one row per width: naive (ghost fill) vs matched (solid) vs matched+REPAIR (thin)
  width_barrier_<ds>.png              barrier vs width (verification plate)
  width_planes_<ds>_{spectral,topo}.png  planes through {A, B, pi(B)} for every width

usage: python render_width.py mnist [--pieces ridge,barrier,planes]
"""
import argparse, textwrap
from scipy.ndimage import zoom
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument('ds')
ap.add_argument('--pieces', default='ridge,barrier,planes')
args = ap.parse_args()
D = load(f'width_{args.ds}.npz')
for extra in sorted(__import__('glob').glob(os.path.join(CACHE, f'width_{args.ds}_*.npz'))):  # e.g. width_mnist_2048.npz from a GPU job
    E_ = dict(np.load(extra, allow_pickle=True))
    D.update({k: v for k, v in E_.items() if k.startswith('w') and k != 'widths_done'})
    D['widths_done'] = np.array(sorted(set(D['widths_done'].tolist()) | set(E_['widths_done'].tolist())))
DSNAME = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}[args.ds]
widths = [int(w) for w in D['widths_done']]
lams = D['lams']


def blin(L):
    return (L - ((1 - lams) * L[0] + lams * L[-1])).max()


def bars(key, col=0):
    return np.array([blin(r[:, col]) for r in D[key]])


def cap(fig, H, y, txt, col, width=150, fs=11):
    fig.text(0.5, 1 - y / H, '\n'.join(textwrap.fill(p, width) for p in txt.split('\n')), ha='center', va='top',
             color=col, fontsize=fs, linespacing=1.6)


if 'ridge' in args.pieces:
    for style in ['night', 'paper']:
        bg, ink, sub = (NIGHT, '#ece5d5', '#8f887c') if style == 'night' else (PAPER, INK, '#5b554d')
        acc = '#f4a259' if style == 'night' else '#b2182b'
        W, H = 2400, 3200
        fig = fig_px(W, H, bg=bg)
        ax = ax_px(fig, 330, 420, 1550, 2300, W, H)
        n = len(widths)
        ymax = max(D[f'w{w}_naive'][:, :, 0].max() for w in widths)
        step = 1.0 / n
        amp = step * 1.9 / ymax
        xs = np.linspace(0, 1, 400)
        for i, w in enumerate(widths):
            base = 1 - (i + 1) * step
            nv = np.array([np.interp(xs, lams, r[:, 0]) for r in D[f'w{w}_naive']])
            mt = np.array([np.interp(xs, lams, r[:, 0]) for r in D[f'w{w}_matched']])
            rp = np.array([np.interp(xs, lams, r[:, 0]) for r in D[f'w{w}_matched_repair']])
            z = 10 * (i + 1)  # lower rows in front (ridgeline convention)
            ax.fill_between(xs, base, base + amp * nv.max(0), color=bg, zorder=z, lw=0)
            ax.fill_between(xs, base + amp * nv.min(0), base + amp * nv.max(0), color=ink, alpha=0.16, zorder=z + 1, lw=0)
            ax.plot(xs, base + amp * nv.mean(0), color=ink, lw=0.8, alpha=0.55, zorder=z + 2, ls=(0, (3, 2)))
            ax.fill_between(xs, base, base + amp * mt.mean(0), color=ink, alpha=0.9, zorder=z + 3, lw=0)
            ax.plot(xs, base + amp * rp.mean(0), color=acc, lw=1.3, zorder=z + 4)
            ax.plot([0, 1], [base, base], color=ink, lw=0.6, zorder=z + 4)
            ax.text(-0.03, base + 0.004, f'{w}', ha='right', va='bottom', color=ink, fontsize=16, zorder=z + 5)
            bm, bn = bars(f'w{w}_matched').mean(), bars(f'w{w}_naive').mean()
            ax.text(1.03, base + 0.004, f'{bn:.2f} → {bm:.3f}', ha='left', va='bottom', color=sub, fontsize=11,
                    family='DejaVu Sans Mono', zorder=z + 5)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1 + 1.0 * step)
        fig.text(310 / W, 1 - 380 / H, 'width', ha='right', color=sub, fontsize=12, style='italic')
        fig.text(1900 / W, 1 - 380 / H, 'barrier: naive → matched', ha='left', color=sub, fontsize=11, style='italic')
        ax.text(0, -0.008, 'A', ha='center', va='top', color=ink, fontsize=14)
        ax.text(1, -0.008, 'B  /  π(B)', ha='center', va='top', color=ink, fontsize=14)
        fig.text(0.5, 1 - 130 / H, 'THE MOUNTAIN IS BOOKKEEPING, EVENTUALLY', ha='center', color=ink, fontsize=26)
        fig.text(0.5, 1 - 210 / H, f'loss along the straight line between two {DSNAME} MLPs, by hidden width', ha='center',
                 color=ink, fontsize=15, style='italic')
        cap(fig, H, 2830, f'Each row: 3 hidden ReLU layers of the given width; 3 independent pairs of networks (Adam, 20 epochs). '
            f'Ghost band and dashed line: train loss along A → B (min–max and mean over pairs). Solid: along A → π(B) after '
            f'weight matching (mean). {"Orange" if style == "night" else "Red"} line: matched + REPAIR. Same vertical scale in every row '
            f'(row height = {ymax / 1.9:.2f} nats); numbers are mean barriers in nats. 25 measured points per path, linearly interpolated.',
            sub, width=125)
        save(fig, f'width_ridge_{args.ds}_{style}.png')
    print('ridge done')

if 'barrier' in args.pieces:
    W, H = 2400, 1600
    fig = fig_px(W, H, bg='#faf7f0')
    for j, (col, nm) in enumerate([(0, 'train'), (2, 'test')]):
        ax = ax_px(fig, 250 + j * 1100, 250, 900, 1000, W, H, off=False)
        ax.set_facecolor('#faf7f0')
        for key, c, lab in [('naive', '#555555', 'naive'), ('naive_repair', '#999999', 'naive + REPAIR'),
                            ('matched', '#2166ac', 'weight matching'), ('matched_repair', '#b2182b', 'matching + REPAIR')]:
            b = np.array([bars(f'w{w}_{key}', col) for w in widths])
            ax.plot(widths, b.mean(1), '-', color=c, lw=1.6, label=lab)
            for k in range(b.shape[1]):
                ax.plot(widths, b[:, k], 'o', color=c, ms=3.5, alpha=0.7)
        ax.set_xscale('log', base=2); ax.set_yscale('symlog', linthresh=0.01)
        ax.set_xticks(widths); ax.set_xticklabels(widths)
        ax.set_xlabel('hidden width'); ax.set_ylabel(f'{nm} loss barrier (nats)')
        ax.axhline(0, color='k', lw=0.5); ax.set_ylim(bottom=-0.001)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=10)
        ax.set_title(nm, fontsize=13, style='italic')
    fig.text(0.5, 1 - 110 / H, f'Barrier vs width, {DSNAME} MLPs (3 pairs per width)', ha='center', fontsize=20, color=INK)
    cap(fig, H, 1380, 'barrier = max over λ of L(λ) − [(1−λ)L(0) + λL(1)], 25 points; dots are individual pairs, lines are means; '
        'symlog y (linear below 0.01).', '#5b554d')
    save(fig, f'width_barrier_{args.ds}.png')
    print('barrier done')

if 'planes' in args.pieces and os.path.exists(os.path.join(CACHE, f'planes_width_{args.ds}.npz')):
    Pd = load(f'planes_width_{args.ds}.npz')
    ws = [w for w in widths if f'w{w}_train' in Pd]
    for style in ['spectral', 'topo']:
        W = 3600
        ncol = min(4, len(ws))
        nrow = int(np.ceil(len(ws) / ncol))
        tile, gap = 780, 70
        H = 330 + nrow * (tile + gap + 60) + 300
        bg, ink = ('#faf7f0', INK) if style == 'spectral' else (PAPER, INK)
        fig = fig_px(W, H, bg=bg)
        x0 = (W - ncol * tile - (ncol - 1) * gap) / 2
        for i, w in enumerate(ws):
            r, c = divmod(i, ncol)
            ax = ax_px(fig, x0 + c * (tile + gap), 300 + r * (tile + gap + 60), tile, tile, W, H)
            Lg = Pd[f'w{w}_train']
            xs, ys, pts = Pd[f'w{w}_xs'], Pd[f'w{w}_ys'], Pd[f'w{w}_pts']
            ext = [xs[0], xs[-1], ys[0], ys[-1]]
            F_ = zoom(np.log(Lg), 700 / Lg.shape[0], order=3)
            from scipy.ndimage import map_coordinates
            seg = pts[0][None] + np.linspace(0, 1, 400)[:, None] * (pts[2] - pts[0])[None]
            cth = float(np.exp(map_coordinates(np.log(Lg), [(seg[:, 1] - ys[0]) / (ys[1] - ys[0]),
                                                            (seg[:, 0] - xs[0]) / (xs[1] - xs[0])], order=3).max()))
            if style == 'spectral':
                ax.imshow(P.render_split(F_ - np.log(cth), 'sd_spectral', near_boundary='small'), origin='lower',
                          extent=ext, interpolation='lanczos')
                mk = INK
            else:
                gx = np.linspace(ext[0], ext[1], F_.shape[1]); gy = np.linspace(ext[2], ext[3], F_.shape[0])
                ax.contour(gx, gy, F_, levels=np.log(np.geomspace(Lg.min() * 1.02, Lg.max(), 32)), colors=INK,
                           linewidths=0.4, negative_linestyles='solid')
                ax.contour(gx, gy, F_, levels=[np.log(cth)], colors='#b2182b', linewidths=1.4, negative_linestyles='solid')
                mk = INK
            ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color=mk, lw=0.7, ls=(0, (3, 2)))
            ax.plot([pts[0, 0], pts[2, 0]], [pts[0, 1], pts[2, 1]], color=mk, lw=0.9)
            for p in pts:
                ax.plot(*p, '^', ms=6, mfc=bg, mec=mk, mew=1)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            fig.text((x0 + c * (tile + gap) + tile / 2) / W, 1 - (300 + r * (tile + gap + 60) + tile + 40) / H,
                     f'width {w}', ha='center', va='center', color=ink, fontsize=15, style='italic')
        fig.text(0.5, 1 - 120 / H, f'ONE BASIN, BY WIDTH  ·  planes through A, B, π(B)  ·  {DSNAME}', ha='center', color=ink, fontsize=24)
        desc = ('Seam (Spectral split): train loss equal to the highest loss on the segment A–π(B) of that width; purple side lower, '
                'red side higher, each side rank-normalised per tile (declared aesthetic).' if style == 'spectral' else
                'Ink: 32 log-spaced loss contours per tile; red: loss equal to the highest loss on the segment A–π(B).')
        cap(fig, H, H - 230, f'Each tile: train loss (5k fixed images) on an affine plane, {Lg.shape[0]}² grid, bicubic-upsampled '
            'log-loss; tiles are scaled independently (each spans its own triangle). Triangles: A (lower left), B (dashed line), '
            'π(B) (solid line). ' + desc, '#5b554d', width=200)
        save(fig, f'width_planes_{args.ds}_{style}.png')
    print('planes done')
