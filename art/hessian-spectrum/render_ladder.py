"""Class-count ladder: the top of every spectrum as rungs, both measurement routes side by side.

Each column pair = one class count C (left: exact spectrum of the small MLP, right: Lanczos Ritz values of the
784-128-128 MLP). Each rung = one of the top 2C+6 eigenvalues at its symlog height. Ink of a rung is a measured
mix: the fraction ov = ||proj of its eigenvector onto span{class-mean gradients}||^2 goes to the red ink, 1-ov to
the blue ink (two-spot risograph idiom, multiplicative overprint; misregistration is aesthetic).

python render_ladder.py            -> gallery/ladder_riso.png, gallery/ladder_night.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_common import SERIES, slog, islog, load_exact, load_lanczos, n_structural, GAL, set_tau

set_tau(2e-3)
cols = []
for C in SERIES:
    for src, L, key in (('exact', load_exact, 'H_eig'), ('lanczos', load_lanczos, 'H_ritz')):
        d = L(C)
        lam = np.sort(d[key])[::-1]
        n = min(2 * C + 6, len(d['ov_dc']))
        cols.append(dict(C=C, src=src, lam=lam[:n], ov=np.asarray(d['ov_dc'])[:n], k=n_structural(d)))

STY = {
    'riso': dict(paper='#f2ede1', red='#ff4f5e', blue='#1f5fbf', ink='#2b2a33', faint='#8c8778', mis=(0.05, 0.004)),
    'night': dict(paper='#07080c', red='#ff6a4d', blue='#5aa9ff', ink='#e6e1d4', faint='#6f6a60', mis=(0, 0)),
}


def hexrgb(h):
    return np.array([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)])


for name, st in STY.items():
    plt.rcParams.update({'font.family': 'P052'})
    fig = plt.figure(figsize=(24, 15), dpi=100, facecolor=st['paper'])
    ax = fig.add_axes([0.08, 0.19, 0.84, 0.66], facecolor=st['paper'])
    ylo, yhi = slog(0.03), slog(12)
    xpos = {}
    for i, c in enumerate(cols):
        gx = SERIES.index(c['C']) * 3.0 + (0 if c['src'] == 'exact' else 1.05)
        xpos.setdefault(c['C'], []).append(gx)
        for j, (l, ov) in enumerate(zip(c['lam'], c['ov'])):
            if l <= 0.03:
                continue
            y = slog(l)
            out = j < c['k']
            w = 0.9 if out else 0.55
            lw = 5.5 if out else 2.2
            x_a, x_b = gx - w / 2, gx + w / 2
            if name == 'riso':
                # two inks, each at its coverage, multiply-overprinted
                dx, dy = st['mis']
                ax.plot([x_a, x_b], [y, y], color=st['blue'], lw=lw, alpha=float(np.clip(1 - ov, 0.08, 1)) * 0.85,
                        solid_capstyle='butt')
                ax.plot([x_a + dx, x_b + dx], [y + dy, y + dy], color=st['red'],
                        lw=lw, alpha=float(np.clip(ov, 0.0, 1)) * 0.9, solid_capstyle='butt')
            else:
                col = ov * hexrgb(st['red']) + (1 - ov) * hexrgb(st['blue'])
                ax.plot([x_a, x_b], [y, y], color=col, lw=lw, alpha=0.95 if out else 0.55, solid_capstyle='butt')
    for C in SERIES:
        xm = np.mean(xpos[C])
        k_ex = [c['k'] for c in cols if c['C'] == C]
        ax.text(xm, slog(0.021), f'C = {C}', ha='center', va='top', fontsize=30, color=st['ink'])
        ax.text(xm, slog(0.0125), f'{k_ex[0]}  ·  {k_ex[1]}', ha='center', va='top', fontsize=20, color=st['faint'],
                family='Nimbus Mono PS')
        for gx, lab in zip(xpos[C], ('exact', 'Lanczos')):
            ax.text(gx, slog(0.035), lab, ha='center', va='top', fontsize=11, color=st['faint'],
                    family='Nimbus Mono PS', rotation=0)
    tk = [0.03, 0.1, 0.3, 1, 3, 10]
    ax.set_yticks(slog(np.array(tk)))
    ax.set_yticklabels([f'{t:g}' for t in tk], fontsize=16, color=st['ink'], family='Nimbus Mono PS')
    ax.tick_params(axis='y', colors=st['faint'], length=8)
    ax.set_xticks([])
    for sp in ['top', 'right', 'bottom']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(st['faint'])
    ax.set_xlim(-1.2, 3.0 * len(SERIES) - 0.8); ax.set_ylim(ylo - 0.05, yhi)
    ax.set_ylabel('Hessian eigenvalue λ  (symlog, τ = 2e-3)', fontsize=15, color=st['faint'], family='Nimbus Mono PS')
    fig.text(0.08, 0.95, 'The class-count ladder', fontsize=46, color=st['ink'], va='top')
    fig.text(0.08, 0.895, 'Top Hessian eigenvalues of MNIST MLPs trained on C digit classes.  Wide rungs: eigenvectors '
             'lying in the span of the class-mean gradients.  There are C − 1 of them, not C.',
             fontsize=18, color=st['faint'], va='top', style='italic')
    fig.text(0.08, 0.02, 'rung ink = measured mix: red ∝ |proj of eigenvector v onto span{d_c}|², blue ∝ the rest;  numbers under C: wide rungs '
             '(exact · Lanczos);  top 2C+6 eigenvalues above 0.03 shown', fontsize=13, color=st['faint'],
             family='Nimbus Mono PS')
    out = f'{GAL}/ladder_{name}.png'
    fig.savefig(out, dpi=100, facecolor=st['paper']); plt.close(fig); print('wrote', out)
