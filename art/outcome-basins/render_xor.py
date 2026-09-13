"""Render piece B ("Modulo Permutation"): XOR 2-2-1 tanh network basin maps, from cache."""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_lib import *
from fractal import boundary
from render_fact3 import downs, BG

G = 'gallery'
NC_LO, NC_HI = np.array(P.hex2rgb('#8a847c')), np.array(P.hex2rgb('#5a554f'))
NAMES = {1: '¬x₁∧¬x₂', 14: 'x₁∨x₂', 7: '¬(x₁∧x₂)', 8: 'x₁∧x₂', 2: '¬x₁∧x₂', 13: 'x₁∨¬x₂', 4: 'x₁∧¬x₂', 11: '¬x₁∨x₂'}


def newton_xor(d, key):
    lab = d[key]
    cols = xor_raw_colors(lab) if key == 'raw' else xor_canon_colors(lab)
    rgb, _ = style_newton(lab, d['status'], d['tconv'], cols, vmin=0.3)
    # not converged = stuck on a plateau; two neutral tones by plateau loss (0.25 vs 0.5), declared
    nc = d['status'] == 1
    hi = d['loss'] > 0.375
    rgb[nc & ~hi] = NC_LO; rgb[nc & hi] = NC_HI
    return rgb


def ink_xor(d):
    """Single ink: every raw-identity boundary hairline; boundaries that survive quotienting by
    permutation+sign drawn heavy."""
    raw = np.where(d['status'] == 0, d['raw'], -1 - d['status'])
    can = np.where(d['status'] == 0, d['canon'], -1 - d['status'])
    b_raw = boundary(raw); b_can = boundary(can)
    from scipy import ndimage
    heavy = ndimage.binary_dilation(b_can, iterations=1)
    cov = np.maximum(b_raw * 0.6, heavy * 1.0)
    return PAPER * (1 - cov[..., None]) + INK * cov[..., None]


def riso_xor(d):
    fam = {23: (0,), 36: (2,)}     # canonical (1,7) -> fluo pink ; (2,4) -> medium blue
    lab = d['canon']
    # raw sign/order variants add the yellow drum when the raw code is not the family's first variant
    raw = d['raw']
    extra = np.zeros_like(raw, bool)
    for code in np.unique(raw[raw >= 0]):
        c1, c2 = divmod(int(code), 16)
        extra[raw == code] = (c1 != min(c1, 15 - c1)) ^ (c2 != min(c2, 15 - c2))   # odd number of sign flips
    lab2 = np.where(extra & (lab >= 0), lab + 1000, lab)
    cmap = {23: (0,), 36: (2,), 1023: (0, 1), 1036: (2, 1)}
    return style_riso(lab2, d['status'], d['tconv'], cmap, ['#ff48b0', '#ffe800', '#3255a4'])


def hero(tag='xor_s4_e1.2'):
    d = load(f'cache/maps/hero_{tag}.npz')
    raw_rgb = newton_xor(d, 'raw'); can_rgb = newton_xor(d, 'canon')
    save(raw_rgb, f'{G}/{tag}_raw_newton.png'); save(can_rgb, f'{G}/{tag}_canon_newton.png')
    save(ink_xor(d), f'{G}/{tag}_ink.png')
    save(riso_xor(d), f'{G}/{tag}_riso.png')
    rgb, _ = style_dark_time(d['status'], d['tconv']); save(rgb, f'{G}/{tag}_darktime.png')
    save(style_spectral(d['status'], d['tconv']), f'{G}/{tag}_spectral.png')
    # diptych with legend
    fig = plt.figure(figsize=(26, 14.2), facecolor=BG)
    for i, (rgb, t) in enumerate([(raw_rgb, 'raw identity: which boolean function each hidden unit learned, in order, with sign'),
                                  (can_rgb, 'the same runs modulo hidden-unit permutation and tanh sign flip')]):
        ax = fig.add_axes([0.01 + i * 0.5, 0.165, 0.48, 0.75]); ax.imshow(downs(rgb[::-1], 2), interpolation='lanczos'); ax.axis('off')
        ax.set_title(t, color='#e8e2d6', fontsize=17, family='serif', pad=10)
    # legend: raw classes grouped by family
    raw = d['raw']; conv = d['status'] == 0
    codes, cnt = np.unique(raw[conv], return_counts=True)
    cols = xor_raw_colors(raw)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.14]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    order = np.argsort(-cnt)
    for j, k in enumerate(order[:16]):
        c1, c2 = divmod(int(codes[k]), 16)
        x = 0.01 + (j % 8) * 0.124; y = 0.62 - (j // 8) * 0.45
        ax.add_patch(plt.Rectangle((x, y), 0.018, 0.3, color=cols[codes[k]]))
        ax.text(x + 0.022, y + 0.15, f'h₁={NAMES.get(c1, c1)}, h₂={NAMES.get(c2, c2)}  {cnt[k] / raw.size:.1%}',
                color='#cfc8bc', fontsize=10.5, va='center', family='serif')
    frac = [float((d['status'] == s).mean()) for s in range(3)]
    fig.text(0.5, 0.972, f'XOR, 2-2-1 tanh net, full-batch GD η = 1.2, 2048² initialisations on a random 2-plane (seed 4). '
             f'Warm family = solution type {{AND, OR}}-like units, cool = {{x₁∧¬x₂, ¬x₁∧x₂}}. Grey = stuck on a plateau '
             f'({frac[1]:.0%}), near-black = diverged ({frac[2]:.0%}). Brightness = log convergence time.',
             color='#a9a39a', fontsize=12.5, ha='center', family='serif')
    fig.savefig(f'{G}/diptych_{tag}.png', dpi=100, facecolor=BG); plt.close(fig)


def eta_sheet():
    fs = sorted(glob.glob('cache/maps/eta_xor_eta*.npz'), key=lambda s: float(s.split('eta')[-1][:-4]))
    fs += ['cache/maps/hero_xor_s4_e1.2.npz']
    fs = sorted(fs, key=lambda f: load(f)['meta']['eta'])
    for key in ('raw', 'canon'):
        n = len(fs); cols = 4; rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(6 * cols, 6.5 * rows), facecolor=BG)
        for i, f in enumerate(fs):
            d = load(f); eta = d['meta']['eta']
            rgb = newton_xor(d, key)
            k = max(rgb.shape[0] // 768, 1)
            ax = fig.add_axes([i % cols / cols + 0.004, 1 - (i // cols + 1) / rows + 0.004, 1 / cols - 0.008, 1 / rows - 0.035])
            ax.imshow(downs(rgb[::-1], k), interpolation='lanczos'); ax.axis('off')
            s = d['sharp'][d['status'] == 0]
            ax.set_title(f'η = {eta:g}   max λ at solution {s.max():.3f}  (2/η = {2 / eta:.3f})', color='#e8e2d6', fontsize=13, family='serif')
        if n % cols:
            ax = fig.add_axes([n % cols / cols + 0.02, 1 - rows / rows + 0.03, 1 / cols - 0.04, 1 / rows - 0.08]); ax.axis('off')
            txt = ('XOR 2-2-1 tanh net, full-batch GD.\nOne random 2-plane (seed 4) through\ninitialisation space, ±3, 1024² runs each.\n\n'
                   + ('Hue = raw solution identity\n(ordered pair of hidden-unit functions).' if key == 'raw' else
                      'Hue = canonical solution\n(modulo unit permutation and sign flip).')
                   + '\nGrey = plateau at T = 20 000, near-black = diverged.\n\nThe fan on the left appears near\nη = 1.0 and thickens with η.')
            ax.text(0, 1, txt, color='#cfc8bb', fontsize=14, family='serif', va='top', linespacing=1.5)
        fig.savefig(f'{G}/eta_series_xor_{key}.png', dpi=100, facecolor=BG); plt.close(fig)


def zoom_sheet(tag='zX'):
    fs = sorted(glob.glob(f'cache/zoom/{tag}_L*.npz'), key=lambda s: int(s.split('_L')[-1][:-4]))
    for style in ('raw', 'canon', 'spectral'):
        n = len(fs); cols = 3; rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(7 * cols, 7.5 * rows), facecolor=BG)
        for i, f in enumerate(fs):
            d = load(f); m = d['meta']
            if 'loss' not in d:
                d['loss'] = np.zeros(d['status'].shape)
            rgb = style_spectral(d['status'], d['tconv']) if style == 'spectral' else newton_xor(d, style)
            ax = fig.add_axes([i % cols / cols + 0.004, 1 - (i // cols + 1) / rows + 0.004, 1 / cols - 0.008, 1 / rows - 0.03])
            ax.imshow(rgb[::-1], interpolation='lanczos'); ax.axis('off')
            ax.set_title(f'L{m["level"]}  width {2 * m["half"]:.2e}', color='#e8e2d6', fontsize=14, family='serif')
        fig.savefig(f'{G}/zoom_{tag}_{style}.png', dpi=100, facecolor=BG); plt.close(fig)


if __name__ == '__main__':
    what = sys.argv[1:] or ['hero', 'eta', 'zoom']
    if 'hero' in what: hero()
    if 'eta' in what: eta_sheet()
    if 'zoom' in what: zoom_sheet()
