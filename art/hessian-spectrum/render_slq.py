"""SLQ smoothing check (verification figure): what Gaussian kernel width does to the outliers and mini-bulk.

784-128-128-C MLP (P ~ 118k), SLQ with m_slq = 80 Lanczos iterations x nv = 4 Rademacher probes (analyze.py),
density rho(lambda) = mean_probe sum_j w_j N(lambda; theta_j, sigma^2) for sigma in {1e-3, 1e-2, 1e-1},
drawn on a symlog axis (tau = 1e-3); top-200 Ritz values (m_top = 200, full reorth) as ticks; outliers
(eigenvectors in span of class-mean gradients) as red ticks.

python render_slq.py -> gallery/verify_slq_kernel.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_common import load_lanczos, slog, n_structural, GAL, set_tau

set_tau(1e-3)
SIG = [1e-3, 1e-2, 1e-1]
Cs = [2, 4, 10]
fig, axs = plt.subplots(len(Cs), 1, figsize=(16, 3.6 * len(Cs)), dpi=150, sharex=True)
grid = np.concatenate([-np.geomspace(0.5, 1e-5, 800), np.linspace(-1e-5, 1e-5, 50), np.geomspace(1e-5, 20, 1600)])
cols = ['#1f3a93', '#5fa052', '#b8860b']
for ax, C in zip(axs, Cs):
    d = load_lanczos(C)
    nodes, w = d['H_slq_nodes'], d['H_slq_w']
    for s, c in zip(SIG, cols):
        rho = np.zeros_like(grid)
        for n_, w_ in zip(nodes, w):
            rho += (w_[None] * np.exp(-0.5 * ((grid[:, None] - n_[None]) / s) ** 2) / (s * np.sqrt(2 * np.pi))).sum(1)
        rho /= len(nodes)
        ax.plot(slog(grid), rho * np.abs(grid).clip(1e-3) * np.log(10), color=c, lw=1.4,
                label=f'SLQ, Gaussian σ = {s:g}')
    ritz = np.sort(d['H_ritz'])[::-1]
    k = n_structural(d)
    ymax = ax.get_ylim()[1]
    for i, r in enumerate(ritz):
        ax.plot([slog(r)] * 2, [0, ymax * (0.25 if i < k else 0.08)], color='#c0392b' if i < k else '#555', lw=1.0)
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_ylim(0, ymax)
    ax.set_title(f'C = {C}   P = {int(d["P"]):,}   m_slq = {int(d["m_slq"])} iterations × nv = {int(d["nv"])} probes   '
                 f'ticks: top {len(ritz)} Ritz values (red: {k} outlier{"s" if k != 1 else ""}; Lanczos gives both spectrum ends)', fontsize=11, loc='left')
    ax.set_ylabel('ρ(λ)·|λ|·ln10  (density per decade)')
tk = [-0.1, -0.01, -1e-3, 0, 1e-3, 0.01, 0.1, 1, 10]
axs[-1].set_xticks(slog(np.array(tk))); axs[-1].set_xticklabels([f'{t:g}' for t in tk])
axs[-1].set_xlabel('Hessian eigenvalue λ (symlog, linear within ±1e-3)')
axs[0].legend(frameon=False, fontsize=10)
fig.suptitle('SLQ density at three kernel widths vs Lanczos Ritz values: wide kernels fuse the outliers into the bulk tail',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{GAL}/verify_slq_kernel.png'); print('wrote')
