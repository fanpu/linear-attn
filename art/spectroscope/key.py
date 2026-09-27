"""Spectroscope key: what spectra make what colours (synthetic spectra against the 384x384 MP null),
and an inset of actual measured spectra with the colour each one makes.

    .venv/bin/python spectroscope/key.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import spectral as S
from render import gammafy, PAPER, INK, RUNS, GAL, ROWS

P_ = tuple(c / 255 for c in PAPER); I_ = tuple(c / 255 for c in INK)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': I_, 'axes.edgecolor': I_,
                     'axes.labelcolor': I_, 'xtick.color': I_, 'ytick.color': I_, 'figure.facecolor': P_, 'axes.facecolor': P_})


def colour(lam, null, g):
    return np.clip(S.spectra_to_rgb(gammafy(S.energy_power(lam[None], null[None]), g))[0], 0, 1)


def key():
    null = S.mp_null(384, 384)
    n = len(null)
    r = np.arange(n)
    cases = [
        ('rank-1 spike\n(top mode x20)', np.where(r == 0, null * 20, null)),
        ('top 10 modes\nx3', np.where(r < 10, null * 3, null)),
        ('steeper\n(null^1.5)', null ** 1.5),
        ('slightly steeper\n(null^1.1)', null ** 1.1),
        ('the null\n(shuffled W)', null.copy()),
        ('slightly flatter\n(null^0.9)', null ** 0.9),
        ('flatter\n(null^0.6)', null ** 0.6),
        ('all modes equal\n(orthogonal W)', np.ones(n)),
        ('bottom half\nlifted x3', np.where(r >= n // 2, null * 3, null)),
        ('top modes\nremoved', np.where(r < 20, null * 0.05, null)),
    ]
    fig = plt.figure(figsize=(13, 5.6), dpi=150)
    for j, (lab, lam) in enumerate(cases):
        for k, g in enumerate((1.0, 2.0)):
            ax = fig.add_axes([0.075 + j * 0.091, 0.62 - k * 0.2, 0.082, 0.18])
            ax.set_facecolor(colour(lam, null, g)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.text(-0.08, 0.5, 'gamma 1\n(faithful)' if g == 1 else 'gamma 2\n(declared)', transform=ax.transAxes, ha='right', va='center', fontsize=8)
        ax = fig.add_axes([0.075 + j * 0.091, 0.14, 0.082, 0.24])
        ax.loglog(r + 1, null, color=(0.6, 0.58, 0.55), lw=1.2)
        ax.set_xlabel(lab, fontsize=7.5, labelpad=3)
        ax.loglog(r + 1, lam / lam.sum() * null.sum(), color=I_, lw=1.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(1e-4, 100)
        P = S.energy_power(lam[None], null[None])[0]
        ax2 = fig.add_axes([0.075 + j * 0.091, 0.82 + 0.0, 0.082, 0.1])
        ax2.fill_between(S.WL, 0, P, color=I_, lw=0); ax2.axhline(1, color=(0.6, 0.58, 0.55), lw=0.6)
        ax2.set_xlim(400, 700); ax2.set_ylim(0, 3); ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(False)
    fig.text(0.075, 0.955, 'KEY  -  top: spectral power P(wavelength) = dF_trained/dF_null (400 nm = the top modes; flat = white).\n'
             'middle: the colour it makes.  bottom: eigenvalue vs rank, log-log (grey = shuffled null, ink = the matrix).', fontsize=8.5)
    fig.savefig(os.path.join(GAL, 'key.png'), facecolor=P_)
    print(os.path.join(GAL, 'key.png'))


def inset(runs, mats=('L1.q', 'L3.up', 'L4.o.h2', 'emb', 'head'), g=1.0):
    fig, axs = plt.subplots(len(runs), len(mats), figsize=(2.3 * len(mats), 2.0 * len(runs)), dpi=150)
    axs = np.atleast_2d(axs)
    for i, run in enumerate(runs):
        d = np.load(os.path.join(RUNS, run + '.npz'))
        for j, m in enumerate(mats):
            lam = d[m + '/lam'][-1]
            nl = 0.5 * (np.sort(d[m + '/s1'][-1])[::-1] + np.sort(d[m + '/s2'][-1])[::-1])
            ax = axs[i, j]
            ax.set_facecolor(colour(lam, nl, g))
            rk = np.arange(1, len(lam) + 1)
            ax.loglog(rk, nl, color=(0.45, 0.43, 0.4), lw=1.0)
            ax.loglog(rk, np.maximum(lam, 1e-12), color=I_, lw=1.1)
            ax.set_ylim(max(nl.min() * 0.3, lam.max() * 1e-5), lam.max() * 2)
            ax.tick_params(labelsize=6); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if i == 0:
                ax.set_title(m, fontsize=9)
            if j == 0:
                ax.set_ylabel(run, fontsize=10)
    fig.suptitle(f'Measured spectra at the final checkpoint: eigenvalue vs rank (ink = trained, grey = shuffled null). Panel ground = the colour it makes (gamma {g:g}).', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(GAL, 'inset_spectra.png'), facecolor=P_)
    print(os.path.join(GAL, 'inset_spectra.png'))


if __name__ == '__main__':
    key()
    runs = [r for r in (sys.argv[1:] or ['adamw', 'muon', 'sgd', 'null']) if os.path.exists(os.path.join(RUNS, r + '.npz'))]
    if runs:
        inset(runs)
