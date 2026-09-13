"""Render piece A ("Four Roots"): depth-3 scalar factorisation basin maps, all styles, from cache."""
import os, sys, glob, json
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_lib import *

G = 'gallery'
os.makedirs(G, exist_ok=True)
RISO_INKS = ['#ff48b0', '#ffe800', '#3255a4']          # fluo pink, yellow, medium blue (stencil.wiki)
RISO_MAP = {0: (1,), 1: (0,), 2: (2,), 3: (0, 2)}      # +++ yellow; +-- pink; -+- blue; --+ pink+blue overprint
BG = '#141217'


def downs(rgb, k):
    """area-average downsample by integer k (anti-aliases boundaries honestly)."""
    H, W, _ = rgb.shape
    return rgb[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k, 3).mean((1, 3))


def heroes(which=None):
    for f in sorted(glob.glob('cache/maps/hero_f3_*.npz')):
        tag = os.path.basename(f)[5:-4]
        if which and tag not in which:
            continue
        d = load(f)
        lab, st, tc = d['raw'], d['status'], d['tconv']
        rgb, _ = style_newton(lab, st, tc, F3_COLORS, vmin=0.3)
        save(rgb, f'{G}/{tag}_newton.png')
        save(style_ink(lab, st, weight_between=3, weight_edge=2), f'{G}/{tag}_ink.png')
        save(style_riso(lab[::2, ::2], st[::2, ::2], tc[::2, ::2], RISO_MAP, RISO_INKS), f'{G}/{tag}_riso.png')
        rgb, _ = style_dark_time(st, tc)
        save(rgb, f'{G}/{tag}_darktime.png')
        save(style_spectral(st, tc), f'{G}/{tag}_spectral.png')
        print('hero', tag, flush=True)


def triptych(tag='f3_s0.5_e1.1'):
    d = load(f'cache/maps/hero_{tag}.npz')
    st, tc = d['status'], d['tconv']
    a, _ = style_newton(d['raw'], st, tc, F3_COLORS, vmin=0.3)
    b, _ = style_newton(d['canon'], st, tc, F3_CANON, vmin=0.3)
    c = style_spectral(st, tc)
    panels = [downs(x[::-1], 2) for x in (a, b, c)]
    titles = ['which solution: 4 sign classes of xyz = 1',
              'modulo permuting (x, y, z): 2 classes',
              'modulo all symmetries: 1 class, only whether']
    fig = plt.figure(figsize=(30, 11.2), facecolor=BG)
    for i, (p, t) in enumerate(zip(panels, titles)):
        ax = fig.add_axes([0.01 + i * 0.33, 0.06, 0.32, 0.86]); ax.imshow(p, interpolation='lanczos'); ax.axis('off')
        ax.set_title(t, color='#e8e2d6', fontsize=22, family='serif', pad=14)
    fig.text(0.5, 0.02, 'f(x,y,z) = ¼(xyz − 1)², full-batch GD, η = 1.1, slice x+y+z = √3·0.5, window ±3.5, 4096² runs. '
             'Left/middle: hue = solution, brightness = log₁₀ convergence time. Right: Sohl-Dickstein Spectral split '
             '(converged purple→yellow by speed, diverged red→yellow).', color='#a9a39a', fontsize=13, ha='center', family='serif')
    fig.savefig(f'{G}/triptych_{tag}.png', dpi=100, facecolor=BG)
    plt.close(fig)


def eta_sheet():
    fs = sorted(glob.glob('cache/maps/eta_f3_eta*.npz'), key=lambda s: float(s.split('eta')[-1][:-4]))
    n = len(fs); cols = 3 if n <= 9 else 4; rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(8 * cols, 8.6 * rows), facecolor=BG)
    for i, f in enumerate(fs):
        d = load(f); eta = d['meta']['eta']
        rgb, _ = style_newton(d['raw'], d['status'], d['tconv'], F3_COLORS, vmin=0.3)
        ax = fig.add_axes([i % cols / cols + 0.005, 1 - (i // cols + 1) / rows + 0.005, 1 / cols - 0.01, 1 / rows - 0.03])
        ax.imshow(downs(rgb[::-1], 2), interpolation='lanczos'); ax.axis('off')
        ax.set_title(f'η = {eta:g}   ηλ*_min = {eta * 1.5:.2f}', color='#e8e2d6', fontsize=20, family='serif')
    fig.savefig(f'{G}/eta_series_f3.png', dpi=100, facecolor=BG); plt.close(fig)


def zoom_sheet(tag='zA', style='newton'):
    fs = sorted(glob.glob(f'cache/zoom/{tag}_L*.npz'), key=lambda s: int(s.split('_L')[-1][:-4]))
    n = len(fs); cols = 5; rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6 * cols, 6.5 * rows), facecolor=BG if style != 'ink' else '#f3eee2')
    for i, f in enumerate(fs):
        d = load(f); m = d['meta']
        if style == 'newton':
            rgb, _ = style_newton(d['raw'], d['status'], d['tconv'], F3_COLORS, vmin=0.3)
        elif style == 'ink':
            rgb = style_ink(d['raw'], d['status'], weight_between=2, weight_edge=1)
        else:
            rgb = style_spectral(d['status'], d['tconv'])
        ax = fig.add_axes([i % cols / cols + 0.004, 1 - (i // cols + 1) / rows + 0.004, 1 / cols - 0.008, 1 / rows - 0.03])
        ax.imshow(downs(rgb[::-1], 2), interpolation='lanczos'); ax.axis('off')
        col = '#e8e2d6' if style != 'ink' else '#1c1b1a'
        ax.set_title(f'L{m["level"]}  width {2 * m["half"]:.2e}  (×{3.5 / m["half"]:.1e})', color=col, fontsize=15, family='serif')
    fig.savefig(f'{G}/zoom_{tag}_{style}.png', dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)


def splits(tags=('f3_s0.5_e1.1_tassel', 'f3_s0.5_e1.1_tasselzoom', 'f3_s0.5_e1.1'), pairings=('hubble_sho', 'aurora_ember')):
    """extra boundary-split palettes from art/color-research (declared variants of the Spectral split)."""
    for tag in tags:
        d = load(f'cache/maps/hero_{tag}.npz')
        for pr in pairings:
            save(style_spectral(d['status'], d['tconv'], pr, nan_color='#101010'), f'{G}/{tag}_split_{pr}.png')
        print('splits', tag, flush=True)


if __name__ == '__main__':
    what = sys.argv[1:] or ['heroes', 'triptych', 'eta', 'zoom']
    if 'heroes' in what: heroes()
    if 'heroes2' in what: heroes(['f3_s0_e0.3_wide', 'f3_s0.5_e1.1_tassel', 'f3_s0.5_e1.1_tasselzoom', 'f3_s1.5_e1.2'])
    if 'splits' in what: splits()
    if 'triptych' in what: triptych()
    if 'eta' in what: eta_sheet()
    if 'zoom' in what:
        for s in ['newton', 'ink', 'spectral']:
            zoom_sheet('zA', s)
