"""Render the congestion-game Lyapunov planes (reads cache/cong_*.npz).
python render_lyap.py [stills|sequence|video|all]"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from render_lib import GAL, PAPER, INK, spectral, dark_diverging, save_png, flip, u8, to_mp4_gif, font

NAMES = ['full', 'z1_crossing', 'z2_hooks', 'z3_shrimp', 'z4_shrimp', 'z5_shrimp', 'z6_chain']


def load(name):
    d = np.load(f'cache/cong_{name}.npz')
    return d['L'].astype(np.float64), d['s'], d['y']


def stills():
    for name in NAMES:
        if not os.path.exists(f'cache/cong_{name}.npz'):
            continue
        L, s, y = load(name)
        save_png(flip(spectral(L)), f'{GAL}/lyap_{name}_spectral.png')
        if name in ('full', 'z1_crossing', 'z4_shrimp', 'z3_shrimp'):
            save_png(flip(spectral(L, pairing='verdigris_copper')), f'{GAL}/lyap_{name}_verdigris.png')
            save_png(flip(spectral(L, pairing='aurora_ember')), f'{GAL}/lyap_{name}_aurora.png')
            save_png(flip(dark_diverging(L)), f'{GAL}/lyap_{name}_dark.png')
        print('stills', name, flush=True)
    for n in ('full', 'z4_shrimp', 'z3_shrimp'):
        if os.path.exists(f'cache/cong_{n}.npz'):
            plate(n)


def plate(name):
    """Scientific-plate idiom: paper ground, ruled frame, coordinates, caption."""
    L, s, y = load(name)
    img = flip(spectral(L))
    fig = plt.figure(figsize=(12, 14.2), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.1, 0.2, 0.84, 0.72])
    ax.imshow(img, extent=[s[0], s[-1], y[0], y[-1]], aspect='auto', interpolation='nearest')
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=10, direction='out', length=4, width=0.6)
    ax.set_xlabel(r'effective step size  $s=\eta\,(a+b)$', color=INK, fontsize=13)
    ax.set_ylabel(r'equilibrium load of link 1  $y^*=(2b-a)/(a+b)$', color=INK, fontsize=13)
    frac = np.mean(L > 0)
    fig.text(0.1, 0.955, 'PLATE  —  Two agents, two links, multiplicative weights', fontsize=17, color=INK,
             family='DejaVu Serif')
    fig.text(0.1, 0.932, f'Largest Lyapunov exponent of $u_{{t+1}}=u_t-s\\,(\\sigma(u_t)-y^*)$ '
             f'(both agents start from the same mixed strategy)', fontsize=11.5, color=INK)
    cap = (f'Window s ∈ [{s[0]:.4g}, {s[-1]:.4g}],  y* ∈ [{y[0]:.5g}, {y[-1]:.5g}];  {L.shape[1]}×{L.shape[0]} px, float64, '
           f'3000 transient + 5000 averaged iterations, u₀ = 0.1.\n'
           f'Measured: λ per pixel (nats/iteration); λ>0 on {100*frac:.1f}% of the window, max λ = {L.max():.3f}.\n'
           'Declared colour: Sohl-Dickstein Spectral split — λ<0 (periodic learning) purple→blue→green→pale, '
           'λ>0 (chaotic learning)\ndeep red→orange→pale, each side rank-normalised separately; the dark seam is λ = 0. '
           'Straight-edged "tears" are genuine\ncoexisting attractors (the fixed start u₀ lands in a different basin), not artefacts.')
    fig.text(0.1, 0.045, cap, fontsize=10, color=INK, linespacing=1.5, va='bottom')
    fig.savefig(f'{GAL}/plate_{name}.png', facecolor=PAPER)
    plt.close(fig)


SEQ = ['full', 'z1_crossing', 'z4_shrimp', 'z5_shrimp']


def sequence():
    """Nested zoom plate: each panel marks the next window."""
    have = [n for n in SEQ if os.path.exists(f'cache/cong_{n}.npz')]
    data = {n: load(n) for n in have}
    fig, axs = plt.subplots(2, 2, figsize=(16, 17.2), dpi=160, facecolor=PAPER)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08, wspace=0.14, hspace=0.16)
    for i, n in enumerate(have):
        ax = axs.flat[i]; L, s, y = data[n]
        ax.imshow(flip(spectral(L)), extent=[s[0], s[-1], y[0], y[-1]], aspect='auto', interpolation='nearest')
        ax.tick_params(labelsize=9, colors=INK)
        mag = (100 - 1) / (s[-1] - s[0])
        ax.set_title(f'{"ABCD"[i]}.  s-magnification ×{mag:,.0f}   (y*-magnification ×{1/(y[-1]-y[0]):,.0f})',
                     fontsize=12, color=INK, loc='left')
        if i + 1 < len(have):
            _, s2, y2 = data[have[i + 1]]
            ax.add_patch(Rectangle((s2[0], y2[0]), s2[-1] - s2[0], y2[-1] - y2[0], fill=False, ec='black', lw=3.2))
            ax.add_patch(Rectangle((s2[0], y2[0]), s2[-1] - s2[0], y2[-1] - y2[0], fill=False, ec='white', lw=1.2))
    fig.text(0.06, 0.955, 'Shrimps inside the chaos of a learning rule: nested zoom into the (s, y*) Lyapunov plane',
             fontsize=17, color=INK, family='DejaVu Serif')
    fig.text(0.06, 0.02, 'MWU in a two-agent, two-link congestion game (symmetric start). Colour: Spectral split of the largest '
             'Lyapunov exponent at λ = 0, each panel rank-normalised on its own pixels (declared).\nBoxes mark the next panel. '
             'Periodic windows (purple/green) recur at every scale with the same shrimp morphology (Gallas 1993).',
             fontsize=10.5, color=INK, linespacing=1.5)
    fig.savefig(f'{GAL}/lyap_zoom_sequence.png', facecolor=PAPER)
    plt.close(fig)


def video():
    fr = np.load('cache/cong_zoom_frames.npy', mmap_mode='r')
    path = np.load('cache/cong_zoom_path.npy')
    tmp = 'cache/frames_zoom'; os.makedirs(tmp, exist_ok=True)
    nf, R, _ = fr.shape
    f_small = font(22, mono=True)
    from PIL import ImageDraw
    k = 0
    order = list(range(nf)) + [nf - 1] * 48  # hold the last frame 2 s
    for i in order:
        L = np.asarray(fr[i], np.float64)
        img = Image.fromarray(u8(flip(spectral(L)))).resize((1080, 1080), Image.NEAREST)
        a, b, c, d = path[i]
        dr = ImageDraw.Draw(img)
        txt = f's {0.5*(a+b):.5f}  y* {0.5*(c+d):.7f}   zoom ×{99/(b-a):,.0f} / ×{1/(d-c):,.0f}'
        dr.rectangle([0, 1080 - 40, 1080, 1080], fill=(20, 20, 22))
        dr.text((16, 1080 - 34), txt, fill=(235, 230, 215), font=f_small)
        img.save(f'{tmp}/{k:05d}.png'); k += 1
    to_mp4_gif(tmp, 'lyap_zoom', fps=24, gif_width=540, gif_fps=12)


if __name__ == '__main__':
    what = sys.argv[1:] or ['all']
    if 'stills' in what or 'all' in what:
        stills()
    if 'sequence' in what or 'all' in what:
        sequence()
    if 'video' in what or 'all' in what:
        video()
