"""Initial-condition maps.
  icmap   : continuous replicator, player-1 start swept over the simplex, colour = finite-time lambda
  atlas   : discrete MWU basin atlas (2x2 coordination / anti-coordination, 3-link congestion) + nulls
  selfplay: policy-gradient vs MWU self-play, lambda per unit learning time vs step size
python render_basins.py [icmap|atlas|selfplay]"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from render_lib import GAL, PAPER, INK, RISO_BLUE, RISO_PINK, spectral, save_png, u8

THR_IC = 5e-3


def tri_mask_img(img, L, bg):
    out = img.copy(); out[~np.isfinite(L)] = bg
    return out


def to_triangle(img, bg):
    """Shear the right-angle simplex grid (x_P right, x_S up) into an equilateral triangle (declared geometry)."""
    R = img.shape[0]; H = int(round(R * np.sqrt(3) / 2))
    out = np.ones((H, R, 3)) * np.asarray(bg)
    yy, xx = np.mgrid[0:H, 0:R]
    s = (H - 1 - yy) / (H - 1) / 1.0  # x_S in [0,1] (row 0 = top)
    p = xx / (R - 1) - 0.5 * s
    ok = (p >= 0) & (p + s <= 1)
    i = np.clip(np.round(s * (R - 1)).astype(int), 0, R - 1); j = np.clip(np.round(p * (R - 1)).astype(int), 0, R - 1)
    out[ok] = img[i[ok], j[ok]]
    return out


def icmap():
    f = 'cache/icmap_R400_T2000_eps0.50.npz'
    d = np.load(f); L = d['L'].astype(float)
    M = np.where(np.isfinite(L), L - THR_IC, np.nan)
    for pairing, bg, nm in (('sd_spectral', (0.953, 0.933, 0.886), 'spectral'), ('cyanotype_vandyke', (0.97, 0.96, 0.93), 'cyanotype')):
        img = spectral(np.nan_to_num(M, nan=0.0), ref=M[np.isfinite(M)], pairing=pairing)
        img = tri_mask_img(img, L, bg)
        tri = to_triangle(img, bg)
        big = np.asarray(Image.fromarray(u8(tri)).resize((tri.shape[1] * 5, tri.shape[0] * 5), Image.NEAREST))
        save_png(big, f'{GAL}/icmap_{nm}.png')
    # plate with histogram
    fig = plt.figure(figsize=(16, 10.5), dpi=160, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.12, 0.6, 0.78]); ax.set_axis_off()
    img = tri_mask_img(spectral(np.nan_to_num(M), ref=M[np.isfinite(M)]), L, (0.953, 0.933, 0.886))
    ax.imshow(to_triangle(img, (0.953, 0.933, 0.886)), interpolation='nearest')
    R = L.shape[0]; H = R * np.sqrt(3) / 2
    for lab, (px, py) in zip('RPS', [(-12, H + 14), (R + 2, H + 14), (R / 2 - 5, -8)]):
        ax.text(px, py, lab, fontsize=15, color=INK)
    ax2 = fig.add_axes([0.69, 0.2, 0.28, 0.5], facecolor=PAPER)
    lv = np.log10(L[np.isfinite(L) & (L > 0)])
    ax2.hist(lv, 160, color='#5e4fa2'); ax2.hist(lv[lv > np.log10(THR_IC)], 160, color='#9e0142')
    ax2.axvline(np.log10(THR_IC), color=INK, lw=0.8, ls='--')
    ax2.set_xlabel('log₁₀ finite-time λ₁ (T = 2000)', color=INK); ax2.set_ylabel('initial conditions', color=INK)
    frac = np.mean(L[np.isfinite(L)] > THR_IC)
    ax2.set_title(f'bimodal: {100*frac:.1f}% chaotic', color=INK, fontsize=12)
    fig.text(0.03, 0.94, 'Which first move leads to chaos?  Player 1’s initial mixed strategy, coloured by the Lyapunov exponent of the learning that follows',
             fontsize=15, family='DejaVu Serif', color=INK)
    fig.text(0.03, 0.03, 'Continuous replicator learning, zero-sum RPS ε = 0.5, player 2 starts at (0.5, 0.25, 0.25). 79,800 orbits, RK4 h = 0.01, T = 2000, float64. '
             f'Declared colour: Spectral split at λ = {THR_IC:g} (the gap of the bimodal histogram):\nregular orbits purple→green→pale '
             '(ranked; the ripples are genuine torus-frequency variation of the finite-time λ), chaotic orbits deep red→orange (ranked). '
             'Near the simplex edge energy is high and chaos dominates.', fontsize=10, color=INK, linespacing=1.5)
    fig.savefig(f'{GAL}/plate_icmap.png', facecolor=PAPER); plt.close(fig)
    zf = glob.glob('cache/icmap_R400_T2000_eps0.50_zoom*.npz')
    if zf:
        z = np.load(zf[0]); Lz = z['L'].astype(float); Mz = np.where(np.isfinite(Lz), Lz - THR_IC, np.nan)
        imgz = tri_mask_img(spectral(np.nan_to_num(Mz), ref=Mz[np.isfinite(Mz)]), Lz, (0.953, 0.933, 0.886))
        save_png(np.asarray(Image.fromarray(u8(imgz[::-1])).resize((2000, 2000), Image.NEAREST)), f'{GAL}/icmap_zoom_spectral.png')
    print('icmap done')


ATLAS_TITLES = {'coord2x2': ('2×2 coordination, MWU', 'x₀ (player 1)', 'y₀ (player 2)'),
                'anti2x2': ('2×2 anti-coordination, MWU', 'x₀ (player 1)', 'y₀ (player 2)'),
                'link3_a': ('2 agents, 3 links, costs (1, 1.2, 1.5), MWU', 'x_P', 'x_S'),
                'link3_b': ('2 agents, 3 links, costs (1, 1.3, 1.1), MWU', 'x_P', 'x_S')}
# categorical palette (declared): 9 pure profiles + non-converged, earthy riso-like set
CAT = np.array([[0.114, 0.294, 0.451], [0.839, 0.435, 0.227], [0.525, 0.667, 0.435], [0.878, 0.690, 0.290],
                [0.357, 0.227, 0.420], [0.749, 0.341, 0.408], [0.286, 0.573, 0.596], [0.620, 0.525, 0.380],
                [0.200, 0.200, 0.220], [0.08, 0.08, 0.08]])


def atlas():
    import json
    d = np.load('cache/basin_atlas.npz')
    ver = json.load(open('cache/verify.json')).get('basin_boxcount', {}) if os.path.exists('cache/verify.json') else {}
    names = list(ATLAS_TITLES)
    fig, axs = plt.subplots(2, 4, figsize=(22, 12), dpi=150, facecolor=PAPER)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.1, wspace=0.12, hspace=0.25)
    for j, n in enumerate(names):
        for i, tag in enumerate(('big', 'null')):
            lab = d[f'{n}_{tag}_R1024'].astype(int)
            img = np.ones(lab.shape + (3,)) * np.array([0.953, 0.933, 0.886])
            ok = lab >= 0
            img[ok] = CAT[np.clip(np.where(lab[ok] > 9, 9, lab[ok]) % 10, 0, 9)]
            ax = axs[i, j]; ax.imshow(img, origin='lower', extent=[0, 1, 0, 1], interpolation='nearest')
            ax.tick_params(labelsize=8, colors=INK)
            t, xl, yl = ATLAS_TITLES[n]
            v = ver.get(f'{n}_{tag}', {})
            Dtxt = f'   D = {v["D"]:.2f}' if 'D' in v else ''
            ax.set_title(f'{t}\n{"large step" if tag == "big" else "small step (null)"}{Dtxt}', fontsize=11, color=INK)
            ax.set_xlabel(xl, fontsize=9, color=INK); ax.set_ylabel(yl, fontsize=9, color=INK)
    fig.text(0.03, 0.93, 'Negative result: basins of discrete multiplicative-weights learning are smooth, even at step sizes that make the learning chaotic elsewhere',
             fontsize=15, family='DejaVu Serif', color=INK)
    fig.text(0.03, 0.025, 'Each pixel is one pair of initial mixed strategies; colour = which pure Nash profile the two learners reach (declared categorical palette; black = not converged). '
             '1024² per panel, float64.\nD = box-counting dimension of the basin boundary (fit over box sizes 1–64 px); a smooth curve gives D ≈ 1. Top: large step; bottom: the same game at a small step (null model).',
             fontsize=10, color=INK, linespacing=1.5)
    fig.savefig(f'{GAL}/basin_atlas_negative.png', facecolor=PAPER); plt.close(fig)
    print('atlas done')


def selfplay():
    d = np.load('cache/selfplay.npz'); e = d['etas']
    fig = plt.figure(figsize=(20, 11), dpi=150, facecolor=PAPER)
    ax = fig.add_axes([0.06, 0.58, 0.42, 0.3], facecolor=PAPER)
    for rule, col, lab in (('mwu', RISO_BLUE, 'MWU (logit += η·payoff)'), ('pg', '#c8102e', 'policy gradient (logit += η·∇ payoff)')):
        ax.semilogx(e, d[f'L_{rule}'] / e, color=col, lw=0.9, label=lab)
    ax.axhline(0.0562, color=INK, ls=':', lw=0.8); ax.text(0.5, 0.062, 'continuous-time λ₁ of this start (0.056)', fontsize=9, color=INK)
    ax.set_ylabel('λ per unit learning time  (λ_step / η)', color=INK); ax.legend(fontsize=9, frameon=False)
    ax.set_title('two softmax policies in self-play, zero-sum RPS ε = 0.5, SAF start k = 1', color=INK, fontsize=12)
    ax2 = fig.add_axes([0.06, 0.12, 0.42, 0.34], facecolor=PAPER)
    for rule, col in (('mwu', RISO_BLUE), ('pg', '#c8102e')):
        ax2.loglog(e, np.maximum(d[f'pmin_{rule}'], 1e-300), color=col, lw=0.9)
    ax2.set_ylim(1e-310, 3); ax2.set_xlabel('step size η', color=INK); ax2.set_ylabel('min probability on the orbit (steps 5k–25k)', color=INK)
    ax2.set_title('…while MWU is flung toward the simplex edge (Bailey & Piliouras 2018)', color=INK, fontsize=12)
    show = d['show_etas']
    V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    for r, rule in enumerate(('mwu', 'pg')):
        tr = d[f'traj_{rule}']
        for k in range(len(show)):
            axk = fig.add_axes([0.52 + 0.118 * k, 0.53 - 0.42 * r, 0.11, 0.36]); axk.set_axis_off()
            xy = np.stack([tr[k, :, 1] + 0.5 * tr[k, :, 2], tr[k, :, 2] * np.sqrt(3) / 2], -1)
            axk.plot(V[:, 0], V[:, 1], color=INK, lw=0.5)
            axk.plot(xy[:, 0], xy[:, 1], color=RISO_BLUE if rule == 'mwu' else '#c8102e', lw=0.15, alpha=0.6)
            axk.set_aspect('equal')
            axk.set_title(f'{rule.upper()}  η = {show[k]:g}\nλ/η = {d[f"trajL_{rule}"][k]/show[k]:.3f}', fontsize=9, color=INK)
    fig.text(0.06, 0.94, 'Self-play: the chaos survives discretisation only while the step is small', fontsize=17, family='DejaVu Serif', color=INK)
    fig.text(0.52, 0.05, 'Right: player-1 strategy over 30,000 steps. Exact expected-payoff gradients, float64,\ntangent-vector λ over steps 5k–25k (sweep) or 0–30k (panels). Probabilities below 1e-308 underflow to 0 (plotted at the floor).',
             fontsize=9.5, color=INK)
    fig.savefig(f'{GAL}/selfplay_plate.png', facecolor=PAPER); plt.close(fig)
    print('selfplay done')


if __name__ == '__main__':
    for w in sys.argv[1:]:
        globals()[w]()
