"""Spectrograph plate: every eigenvalue of one layer as an exposed line, rows = training time (top: init).

Measured: sorted eigenvalues of W^T W/N at log-spaced checkpoints.
Declared: rows between measured checkpoints interpolate the k-th largest eigenvalue linearly in log10(lambda) vs
checkpoint index (rank-order tracks, not eigenvector identity); each eigenvalue is a Gaussian line of SIG px and the
plate darkens as 1 - exp(-density/D0) (photographic exposure model); colour maps.
Optional: the MP upper edge (current sigma^2) as a thin trace, and measured checkpoint rows as ticks at the margin.

usage: python render_plate.py run layer [styles...]   styles: magma bio paper riso spectral
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, shape_NM, mp_edges
import render_common as R

WPX, HPX = 2400, 3000
G_ROWS = {}
SIG = 1.1
D0 = 2.5
LO_PCT = 20


def exposure(run, layer, key='lam', lohi=None):
    r = load_run(run)
    lam = np.sort(r[f'{layer}/{key}'], axis=1)
    T = lam.shape[0]
    L = np.log10(np.maximum(lam, 1e-30))
    if lohi is None:
        lo = np.percentile(np.log10(r[f'{layer}/lam']), LO_PCT) - 0.05
        hi = np.log10(r[f'{layer}/lam'].max()) + 0.12
    else:
        lo, hi = lohi
    st = r['step'].astype(float)
    if r['meta'].get('n_lin'):   # rows linear in SGD step
        rows = np.interp(np.linspace(0, st[-1], HPX), st, np.arange(T))
    else:                        # rows linear in log step, from step 100
        rows = np.interp(np.linspace(2, np.log10(st[-1]), HPX), np.log10(np.maximum(st, 1)), np.arange(T))
    i0 = np.minimum(np.floor(rows).astype(int), T - 2); u = rows - i0
    img = np.zeros((HPX, WPX))
    xs = np.arange(WPX)
    for y in range(HPX):
        l = (1 - u[y]) * L[i0[y]] + u[y] * L[i0[y] + 1]
        px = (l - lo) / (hi - lo) * (WPX - 1)
        px = px[(px > -5) & (px < WPX + 5)]
        dens = np.zeros(WPX)
        base = np.floor(px).astype(int)
        for off in range(-3, 5):
            idx = base + off; ok = (idx >= 0) & (idx < WPX)
            np.add.at(dens, idx[ok], np.exp(-0.5 * ((idx[ok] - px[ok]) / SIG) ** 2))
        img[y] = dens
    G_ROWS['rows'] = rows
    edge = np.log10(mp_edges(np.interp(rows, np.arange(T), r[f'{layer}/elem_var']), shape_NM(r, layer)[0] / shape_NM(r, layer)[1])[1])
    return r, 1 - np.exp(-img / D0), (lo, hi), (edge - lo) / (hi - lo) * (WPX - 1)


def render(run, layer, style):
    r, E, (lo, hi), edge = exposure(run, layer)
    N, M = shape_NM(r, layer)
    if style in ('magma', 'bio', 'spectral'):
        cm = {'magma': plt.get_cmap('magma'), 'bio': plt.get_cmap('art.bioluminescence'), 'spectral': plt.get_cmap('Spectral_r')}[style]
        if style == 'spectral':  # labelled variant: sequential exposure on Spectral (can band)
            rgb = cm(0.02 + 0.96 * E)[..., :3]; rgb = rgb * (E[..., None] > 0.004) + (1 - (E[..., None] > 0.004)) * P_hex('#0b0b10')
        else:
            rgb = cm(E ** 0.8)[..., :3]
        bg, fg = '#07070a', '#e6dfcf'
    elif style == 'paper':
        ink = P_hex('#141824'); paper = P_hex(R.PAPER)
        rgb = paper * (1 - E[..., None]) + ink * E[..., None]
        bg, fg = R.PAPER, R.INK
    elif style == 'riso':
        _, En, _, _ = exposure_null(run, layer, (lo, hi))
        rgb = R.riso_composite([(En * 0.9, R.RISO['fluo_pink']), (E, R.RISO['medium_blue'])], E.shape,
                               offsets=[(0, 0), (0, 3)], grain=0.08)
        bg, fg = R.RISO_PAPER, R.RISO['medium_blue']
    # frame
    fig = plt.figure(figsize=(WPX / 200 + 1.6, HPX / 200 + 2.0), dpi=200, facecolor=bg)
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([0.8 / fw, 1.1 / fh, (WPX / 200) / fw, (HPX / 200) / fh])
    ax.imshow(rgb, interpolation='none', aspect='auto', extent=(lo, hi, HPX, 0))
    ax.set_xlim(lo, hi); ax.set_ylim(HPX, 0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([]); ax.tick_params(colors=fg, labelsize=9, length=3)
    ax.set_xticks(np.arange(np.ceil(lo * 2) / 2, hi, 0.5))
    ax.set_xticklabels([f'{t:.1f}' for t in np.arange(np.ceil(lo * 2) / 2, hi, 0.5)], family='DejaVu Sans Mono')
    T = r['step'].size
    rows = G_ROWS['rows']
    for i in range(T):
        y = np.interp(i, rows, np.arange(HPX), left=np.nan, right=np.nan)
        if np.isfinite(y):
            ax.plot([lo, lo + (hi - lo) * 0.006], [y, y], color=fg, lw=0.4, alpha=0.6, clip_on=False)
    lin = bool(r['meta'].get('n_lin'))
    marks = np.linspace(0, r['step'][-1], 7) if lin else [100, 1000, 10000, 100000, r['step'][-1]]
    for s in marks:
        if s > r['step'][-1]:
            continue
        y = np.interp(np.interp(s, r['step'], np.arange(T)), rows, np.arange(HPX))
        ax.text(lo - (hi - lo) * 0.008, y, f'{int(s):,}', color=fg, ha='right', va='center', fontsize=8, family='DejaVu Sans Mono')
    fig.text(0.8 / fw, 1 - 0.45 / fh, f'{layer} {N}×{M} — every eigenvalue of $W^{{T}}W/N$, init (top) to 30 epochs (bottom), SGD step at left',
             color=fg, fontsize=15, family=R.SERIF)
    fig.text(0.8 / fw, 0.45 / fh, f'x: log₁₀ λ · rows: ' + ('linear' if r['meta'].get('n_lin') else 'log') + f' in step; {T} measured checkpoints (margin ticks), k-th largest eigenvalue '
             f'interpolated in log λ between them (declared) · exposure 1−exp(−density/{D0}), line σ {SIG}px'
             + (' · pink: shuffled-entries null' if style == 'riso' else ''),
             color=fg, fontsize=7.5, family='DejaVu Sans')
    return R.save(fig, f'plate_{run}_{layer}_{style}.png', dpi=200)


def exposure_null(run, layer, lohi):
    return exposure(run, layer, key='lam_shuf', lohi=lohi)


def P_hex(h):
    return np.array(R.P.hex2rgb(h))


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    layer = sys.argv[2] if len(sys.argv) > 2 else 'FC1'
    R.set_rc()
    for s in (sys.argv[3:] or ['magma', 'paper', 'riso', 'bio']):
        print(render(run, layer, s))
