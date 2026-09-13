"""Unknown-Pleasures ridgeline stack: one ridge per checkpoint = ESD of one layer (density of log10 lambda).

Measured: eigenvalues of W^T W / N at log-spaced checkpoints (train.py).
Declared aesthetics: Gaussian KDE in log10(lambda) with bandwidth BW dex; ridge height = (density/peak)^GAMMA
(compresses the bulk so lone outlier eigenvalues remain visible); ridges overlap and occlude.

usage: python render_ridgeline.py [run] [layer] [styles...]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, log_kde, shape_NM, mp_edges
import render_common as R

BW = 0.025
GAMMA = 0.5


def ridge_data(run, layer, n_ridges=80, key='lam', grid=None):
    r = load_run(run)
    lam = r[f'{layer}/{key}']
    T = lam.shape[0]
    idx = np.unique(np.round(np.linspace(0, T - 1, min(n_ridges, T))).astype(int))
    if grid is None:
        allv = np.concatenate([r[f'{layer}/lam'][idx].ravel(), r[f'{layer}/lam_shuf'][idx].ravel()])
        lo = np.log10(np.percentile(allv, 2.0)) - 0.1
        hi = np.log10(allv.max()) + 0.12
        grid = np.linspace(lo, hi, 1400)
    D = np.stack([log_kde(lam[i], grid, BW) for i in idx])
    return grid, D, idx, r


def draw_ridges(ax, grid, D, peak, spacing=1.0, height=7.0, fg='w', bg='k', lw=0.8, colors=None, fill=True,
                z0=0):
    H = (D / peak) ** GAMMA * height
    n = len(D)
    for i in range(n):
        y0 = -i * spacing
        y = y0 + H[i]
        if fill:
            ax.fill_between(grid, y0 - 0.05, y, color=bg, zorder=z0 + 2 * i, lw=0)
        c = fg if colors is None else colors[i]
        ax.plot(grid, y, color=c, lw=lw, zorder=z0 + 2 * i + 1, solid_joinstyle='round')
    ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(-(n - 1) * spacing - 1.5, height + 1.0)
    ax.axis('off')


def caption(fig, r, run, layer, idx, color, y=0.035, size=9, font='DejaVu Sans Mono'):
    N, M = shape_NM(r, layer)
    s0, s1 = int(r['step'][idx[0]]), int(r['step'][idx[-1]])
    meta = r['meta']
    txt = (f'{layer}  {N}×{M}   ESD of WᵀW/N   {len(idx)} checkpoints, step {s0} → {s1:,} (log-spaced, top → bottom)   '
           f'MLP 784-1024-1024-1024-10 · FashionMNIST · SGD bs {meta["bs"]} lr {meta["lr"]}   '
           f'x: log₁₀ λ   height: (KDE density)^{GAMMA}, bw {BW} dex')
    fig.text(0.5, y, txt, ha='center', va='center', color=color, fontsize=size, family=font)


def render(run, layer, style, n_ridges=80, out=None):
    grid, D, idx, r = ridge_data(run, layer, n_ridges)
    peak = D.max()
    W, Hh = 8, 10
    if style in ('joy', 'ink', 'gold', 'spectral'):
        st = {'joy': dict(bg='#000000', fg='#f4f1ea', lw=1.0),
              'ink': dict(bg=R.PAPER, fg=R.INK, lw=0.55),
              'gold': dict(bg='#0b0a08', fg=None, lw=0.9),
              'spectral': dict(bg='#101014', fg=None, lw=1.0)}[style]
        fig = plt.figure(figsize=(W, Hh), facecolor=st['bg'])
        ax = fig.add_axes([0.12, 0.09, 0.76, 0.80]); ax.set_facecolor(st['bg'])
        colors = None
        if style == 'gold':
            cm = plt.get_cmap('art.klimt_gold')
            colors = [cm(0.35 + 0.6 * i / (len(idx) - 1)) for i in range(len(idx))]
        if style == 'spectral':
            cm = plt.get_cmap('Spectral_r')
            colors = [cm(0.04 + 0.92 * i / (len(idx) - 1)) for i in range(len(idx))]
        draw_ridges(ax, grid, D, peak, fg=st['fg'], bg=st['bg'], lw=st['lw'], colors=colors)
        tc = st['fg'] if st['fg'] else '#d9cfb5'
        fig.text(0.5, 0.935, 'UNKNOWN PLEASURES OF SGD' if style == 'joy' else 'A weight matrix learning, eigenvalue by eigenvalue',
                 ha='center', color=tc, fontsize=15, family=R.SANS if style == 'joy' else R.SERIF,
                 fontweight='bold' if style == 'joy' else 'normal')
        caption(fig, r, run, layer, idx, tc if style != 'ink' else '#6d675c', size=5.2)
        return R.save(fig, out or f'ridgeline_{run}_{layer}_{style}.png', dpi=300)
    if style == 'riso':
        # two drums: measured ESD ridges (federal blue) over element-shuffled null ridges (fluo pink)
        _, Dn, _, _ = ridge_data(run, layer, n_ridges, key='lam_shuf', grid=grid)
        covs = []
        for DD in (Dn, D):
            fig = plt.figure(figsize=(W, Hh), facecolor='white', dpi=250)
            ax = fig.add_axes([0.12, 0.09, 0.76, 0.80]); ax.set_facecolor('white')
            draw_ridges(ax, grid, DD, peak, fg='black', bg='white', lw=0.9)
            a = R.fig_to_array(fig)
            covs.append(1 - a.mean(-1) / 255)
        shape = covs[0].shape
        img = R.riso_composite([(covs[0], R.RISO['fluo_pink']), (covs[1], R.RISO['federal_blue'])], shape,
                               offsets=[(0, 0), (3, -4)], grain=0.12)
        fig = plt.figure(figsize=(W, Hh), dpi=250)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img, interpolation='none'); ax.axis('off')
        fig.text(0.5, 0.935, 'measured spectrum  /  shuffled-entries null', ha='center', color=R.RISO['federal_blue'],
                 fontsize=14, family=R.SERIF)
        fig.text(0.5, 0.912, 'blue: eigenvalues of the trained layer    pink: same weights, entries randomly permuted',
                 ha='center', color=R.RISO['fluo_pink'], fontsize=7.5, family=R.MONO)
        caption(fig, r, run, layer, idx, R.RISO['federal_blue'], size=5.2)
        return R.save(fig, out or f'ridgeline_{run}_{layer}_riso.png', dpi=250)


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    layer = sys.argv[2] if len(sys.argv) > 2 else 'FC2'
    styles = sys.argv[3:] or ['joy', 'ink', 'gold', 'spectral', 'riso']
    R.set_rc()
    for s in styles:
        print(render(run, layer, s))
