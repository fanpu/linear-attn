"""Departure spectrogram: where along the spectrum, and when, the measured ESD has more (or fewer) eigenvalues
than the same weights with entries shuffled.

Measured field D(t, x) = log10(rho_ESD + eps) - log10(rho_null + eps), rho = Gaussian KDE in x = log10 lambda
(bandwidth BW dex), eps = one third of a single eigenvalue's peak density. Cells where both densities are < eps/2
are empty (ground colour). Declared: Sohl-Dickstein Spectral split at D = 0 (each sign rank-normalised separately;
excess -> purple..pale side, deficit -> red..pale side), plus palettes.py pairings. Columns are checkpoints (log-spaced
in step), nearest-neighbour upsampled.

usage: python render_departure.py run layer
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, log_kde, shape_NM
import render_common as R

BW = 0.06


def field(run, layer, ny=700):
    r = load_run(run)
    lam = r[f'{layer}/lam']; sh = r[f'{layer}/lam_shuf']
    N, M = shape_NM(r, layer)
    lo = np.log10(np.percentile(np.concatenate([lam.ravel(), sh.ravel()]), 0.5)) - 0.05
    hi = np.log10(max(lam.max(), sh.max())) + 0.1
    g = np.linspace(lo, hi, ny)
    eps = (1 / (M * BW * np.sqrt(2 * np.pi))) / 3
    A = np.stack([log_kde(l, g, BW) for l in lam]); B = np.stack([log_kde(l, g, BW) for l in sh])
    D = np.log10(A + eps) - np.log10(B + eps)
    D[(A < eps / 2) & (B < eps / 2)] = np.nan
    return r, g, D.T[::-1]  # rows: high lambda at top


def render(run, layer, pairing='sd_spectral', ground='#0b0b0f', tag=None, norm='rank'):
    r, g, D = field(run, layer)
    if norm == 'rank':
        rgb = R.P.render_split(D, pairing, near_boundary='small', nan_color=ground)
    else:  # linear: |D| / 99th percentile, dark seam at 0 (small departures stay dark)
        from render_texture import lin_split
        rgb = lin_split(np.nan_to_num(D), pairing, q=99)
        rgb[~np.isfinite(D)] = R.P.hex2rgb(ground)
    T = D.shape[1]
    reps = max(1, 2000 // T)
    rgb = np.repeat(rgb, reps, axis=1)
    fig = plt.figure(figsize=(12, 8), dpi=200, facecolor=ground)
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    ax.imshow(rgb, aspect='auto', interpolation='nearest', extent=(-0.5, T - 0.5, g[0], g[-1]))
    fg = '#d8d2c4' if ground < '#8' else R.INK
    steps = r['step']
    marks = np.linspace(0, steps[-1], 6) if r['meta'].get('n_lin') else [1, 10, 100, 1000, 10000, 100000]
    tk = sorted(set([int(np.abs(steps - m).argmin()) for m in marks if m <= steps[-1]] + [T - 1]))
    ax.set_xticks(tk); ax.set_xticklabels([f'{int(steps[i]):,}' for i in tk], color=fg, fontsize=8)
    ax.tick_params(colors=fg, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel('measured checkpoint (label: SGD step; columns equally spaced by checkpoint index)', color=fg, fontsize=9, family='DejaVu Sans')
    ax.set_ylabel('log₁₀ λ', color=fg, fontsize=9, family='DejaVu Sans')
    N, M = shape_NM(r, layer)
    fig.text(0.08, 0.93, f'{layer} {N}×{M}: excess over the shuffled-entries null  (seam: measured = null)', color=fg,
             fontsize=13, family=R.SERIF)
    fig.text(0.08, 0.04, f'colour: log₁₀ρ_ESD − log₁₀ρ_null, KDE bw {BW} dex; {pairing} split at 0, each sign ' + ('rank-normalised' if norm == 'rank' else 'linear ±p99, zero dark') + ' '
             f'(declared). red→pale side: more eigenvalues than null · violet→pale side: fewer', color=fg, fontsize=7.5,
             family='DejaVu Sans')
    return R.save(fig, f'departure_{run}_{layer}_{tag or pairing}.png', dpi=200)


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    layer = sys.argv[2] if len(sys.argv) > 2 else 'FC1'
    R.set_rc()
    print(render(run, layer, 'sd_spectral'))
    print(render(run, layer, 'aurora_ember'))
    print(render(run, layer, 'aurora_ember', tag='aurora_ember_linear', norm='linear'))
