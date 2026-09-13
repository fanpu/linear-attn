"""Hero images in the primary Spectral style (Sohl-Dickstein's colouring), in two forms:
  gallery/hero_<out>_spectral_print.png     axis-free, integer nearest-neighbour upscale
  gallery/hero_<out>_spectral_labelled.png  his figure idiom: axes, end-point tick labels
Also any other style via --style.

  python render_hero.py zoomA:0 overview
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import styles as S
from common_render import cdf_img
from render_windows import load, label_of

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('out')
p.add_argument('--style', default='spectral')
p.add_argument('--px', type=int, default=2048)
args = p.parse_args()

w = load(args.name)
M = w['M']
R = M.shape[0]
if args.style in ('spectral', 'magma'):
    fn = {'spectral': S.spectral, 'magma': S.dark_magma}[args.style]
else:
    # split pairings from the shared colour-research module (declared aesthetic variants);
    # near_boundary='large' = his measure, where large |M| (slow) sits at the edge
    import sys
    sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
    import palettes as P
    fn = lambda m: (np.clip(P.render_split(m, args.style, near_boundary='large')[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)
img = fn(M)
f = max(1, args.px // R)
Image.fromarray(S.upscale(img, f)).save(f'gallery/hero_{args.out}_{args.style}_print.png')


def truncate_sci_notation(numbers):
    """his helper: keep digits until the two end labels differ in four digits"""
    n1, n2 = '{:.15e}'.format(numbers[0]), '{:.15e}'.format(numbers[1])
    s1, e1 = n1.split('e'); s2, e2 = n2.split('e')
    idx = min(len(s1), len(s2))
    for i in range(idx):
        if s1[i] != s2[i] or e1 != e2:
            idx = i + 4 + (1 if i == 0 else 0)
            break
    e1 = e1[0] + e1[2:].lstrip('0').rjust(2, '0') if False else e1
    return [f'{s1[:idx]}e{int(e1)}', f'{s2[:idx]}e{int(e2)}']


x0, x1 = w['c0'] - w['hw'], w['c0'] + w['hw']
y0, y1 = w['c1'] - w['hwy'], w['c1'] + w['hwy']
y = cdf_img(M)
cmap = 'Spectral' if args.style == 'spectral' else 'magma'
fig, ax = plt.subplots(figsize=(8, 8), dpi=args.px / 8)
if args.style == 'spectral':
    ax.imshow(y, extent=[x0, x1, y0, y1], origin='lower', vmin=-1, vmax=1, cmap='Spectral',
              aspect='auto', interpolation='nearest')
else:
    ax.imshow(img, extent=[x0, x1, y0, y1], aspect='auto', interpolation='nearest')
axes = w.get('axes', 'lr_lr')
xl = {'lr_lr': 'Input layer learning rate', 'sigma_lr': 'Weight init scale sigma',
      'wd_lr': 'Weight decay lambda'}[axes]
yl = {'lr_lr': 'Output layer learning rate'}.get(axes, 'Learning rate')
if w['nonlin'] == 'liu_cos':
    xl, yl = 'Learning rate for parameter a', 'Learning rate for parameter b'
ax.set_xlabel(xl); ax.set_ylabel(yl)
ax.set_xticks([x0, x1], truncate_sci_notation(10.0 ** np.array([x0, x1])))
ax.set_yticks([y0, y1], truncate_sci_notation(10.0 ** np.array([y0, y1])), rotation=90)
lab = ax.get_xticklabels(); lab[0].set_horizontalalignment('left'); lab[1].set_horizontalalignment('right')
lab = ax.get_yticklabels(); lab[0].set_verticalalignment('bottom'); lab[1].set_verticalalignment('top')
title = (f'Trainability dependence on per-layer learning rates\n1 hidden layer, {label_of(w)}'
         if axes == 'lr_lr' else f'Trainability, {xl.lower()} vs learning rate\n1 hidden layer, {label_of(w)}')
if w['nonlin'] == 'liu_cos':
    title = (f'Trainability of a two-parameter rippled quadratic (no network)\n'
             f'L = a^2 + 0.6ab + b^2 + {float(w.get("eps", 0.05))}(1 + cos(2pi(a-b)/{float(w.get("lam", 0.2))}))')
ax.set_title(title)
fig.tight_layout()
fig.savefig(f'gallery/hero_{args.out}_{args.style}_labelled.png')
print('hero', args.out)
