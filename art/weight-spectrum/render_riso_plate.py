"""Two-drum risograph plate: Marchenko-Pastur prediction (ink A) vs measured ESD (ink B).

Measured: eigenvalues of W^T W/N at chosen checkpoints; entry variance sigma^2 -> MP density (no fitting).
Declared: sqrt vertical scale (so the tail is not flattened), histogram with 60 log-spaced bins, ink colours,
2-4 px misregistration and paper grain.

usage: python render_riso_plate.py run [inkA inkB]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import load_run, shape_NM, mp_density_log10, LAYERS
import render_common as R

W_IN, H_IN, DPI = 12, 9, 220


def pick_ckpts(r, n=5):
    st = r['step'].astype(float)
    targets = np.array([0, 0.02, 0.1, 0.35, 1.0]) * st[-1]
    return np.unique([int(np.abs(st - t).argmin()) for t in targets])


def layout(r, which, cols):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor='white')
    nL = len(LAYERS)
    for a, L in enumerate(LAYERS):
        N, M = shape_NM(r, L); Q = N / M
        lam = r[f'{L}/lam']
        lo = np.log10(np.percentile(lam, 4)) - 0.1; hi = np.log10(lam.max()) + 0.15
        bins = np.linspace(lo, hi, 61); w = bins[1] - bins[0]
        g = np.linspace(lo, hi, 800)
        ymax = 0
        for t in cols:
            ymax = max(ymax, (np.histogram(np.log10(lam[t]), bins)[0] / (M * w)).max(),
                       mp_density_log10(10 ** g, r[f'{L}/elem_var'][t], Q).max())
        for b, t in enumerate(cols):
            ax = fig.add_axes([0.07 + b * 0.182, 0.655 - a * 0.28, 0.165, 0.20])
            ax.set_xlim(lo, hi); ax.set_ylim(0, np.sqrt(ymax) * 1.08); ax.axis('off')
            if which == 'mp':
                d = mp_density_log10(10 ** g, r[f'{L}/elem_var'][t], Q)
                ax.fill_between(g, 0, np.sqrt(d), color='k', lw=0)
            elif which == 'esd':
                h = np.histogram(np.log10(lam[t]), bins)[0] / (M * w)
                ax.bar(bins[:-1], np.sqrt(h), width=w * 0.5, align='edge', color='k', lw=0)
                ax.vlines(np.log10(lam[t]), -0.06 * np.sqrt(ymax), -0.01 * np.sqrt(ymax), color='k', lw=0.3)
                ax.set_ylim(-0.07 * np.sqrt(ymax), np.sqrt(ymax) * 1.08)
                ax.plot([lo, hi], [-0.075 * np.sqrt(ymax)] * 2, color='k', lw=0.4)
            elif which == 'text':
                ax.set_ylim(-0.07 * np.sqrt(ymax), np.sqrt(ymax) * 1.08)
                if a == 0:
                    ax.text(0.0, 1.12, f"step {int(r['step'][t]):,}", transform=ax.transAxes, fontsize=10, family=R.SERIF)
                if a == nL - 1:
                    for xt in np.arange(np.ceil(lo), hi, 1.0):
                        ax.text(xt, -0.2 * np.sqrt(ymax), f'$10^{{{int(xt)}}}$', ha='center', fontsize=7.5)
            if b == 0 and which == 'text':
                ax.text(-0.08, 0.5, f'{L}\n{N}×{M}', transform=ax.transAxes, ha='right', va='center', fontsize=10,
                        family=R.SERIF)
    if which == 'text':
        fig.text(0.07, 0.955, 'Random matrix theory, and what training did to it', fontsize=19, family=R.SERIF)
        fig.text(0.07, 0.925, 'solid mass: Marchenko–Pastur law for i.i.d. weights of the same variance   ·   '
                 'bars and ticks: measured eigenvalues of WᵀW/N', fontsize=8.5, family='DejaVu Sans')
        fig.text(0.07, 0.035, f"MLP 784-1024-1024-1024-10 · FashionMNIST · SGD lr {r['meta']['lr']} bs {r['meta']['bs']} · "
                 "x: log₁₀ λ, 60 bins · y: √density (declared) · ticks: every eigenvalue", fontsize=7.5, family='DejaVu Sans Mono')
    a = R.fig_to_array(fig)
    return 1 - a.mean(-1) / 255


def render(run, inkA='sunflower', inkB='federal_blue'):
    r = load_run(run)
    cols = pick_ckpts(r)
    mp = layout(r, 'mp', cols); esd = layout(r, 'esd', cols); txt = layout(r, 'text', cols)
    img = R.riso_composite([(mp * 0.6, R.RISO[inkA]), (np.clip(esd + txt, 0, 1), R.RISO[inkB])], mp.shape,
                           offsets=[(0, 0), (3, -3)], grain=0.10)
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img, interpolation='none'); ax.axis('off')
    return R.save(fig, f'riso_mp_vs_esd_{run}_{inkA}_{inkB}.png', dpi=DPI)


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'mlp_bs16_s0'
    pairs = [('sunflower', 'federal_blue'), ('fluo_pink', 'blue'), ('melon', 'indigo')]
    if len(sys.argv) > 3:
        pairs = [(sys.argv[2], sys.argv[3])]
    R.set_rc()
    for a, b in pairs:
        print(render(run, a, b))
