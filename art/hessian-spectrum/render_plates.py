"""Class-count series as observatory spectral plates (exact Hessian eigenvalues, small MLP).

python render_plates.py [--styles emission,absorption,silver,negative] [--src exact|lanczos]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_common import (SERIES, slog, islog, load_exact, load_lanczos, exposure_profile, log_density,
                           wavelength_rgb, hexrgb, GAL, n_structural)

ap = argparse.ArgumentParser()
ap.add_argument('--styles', default='emission,absorption,silver,negative')
ap.add_argument('--src', default='exact')
ap.add_argument('--W', type=int, default=2400)
ap.add_argument('--tau', type=float, default=2e-3)
args = ap.parse_args()
import render_common; render_common.set_tau(args.tau)

STY = {
    'emission': dict(ground='#050505', ink='#d9d4c7', faint='#6b665c', hue=True, absorb=False),
    'absorption': dict(ground='#0b0b0d', ink='#d9d4c7', faint='#6b665c', hue=True, absorb=True),
    'silver': dict(ground='#0c0b0a', ink='#cfc6b4', faint='#5d574d', hue=False, absorb=False, tint='#f1e9d8'),
    'negative': dict(ground='#ece6d8', ink='#2a2520', faint='#9a9080', hue=False, absorb=False, tint='#1e1a16',
                     neg=True),
}

specs = []
for C in SERIES:
  try:
    if args.src == 'exact':
        d = load_exact(C)
        lam = d['H_eig']
        desc = f'MNIST digits 0–{C - 1}  ·  MLP 100-32-32-{C}  ·  P = {int(d["P"]):,}  ·  exact spectrum'
    else:
        d = load_lanczos(C)
        lam = d['H_ritz']
        desc = f'MNIST digits 0–{C - 1}  ·  MLP 784-128-128-{C}  ·  P = {int(d["P"]):,}  ·  Lanczos Ritz values'
    k = n_structural(d)
    specs.append(dict(C=C, lam=lam, k=k, kgap=int(d['count_H'][0]), desc=desc, loss=float(d['loss']),
                      acc=float(d['acc'])))
  except (IndexError, FileNotFoundError):
    print('missing', C)

allS = np.concatenate([slog(s['lam']) for s in specs])
x0, x1 = np.floor(allS.min() * 4) / 4 - 0.25, np.ceil(allS.max() * 4) / 4 + 0.25
W = args.W
xs = np.linspace(x0, x1, W)
wl = np.interp(xs, [x0, x1], [395, 690])
hue = wavelength_rgb(wl)


def strip_rgb(expo, st, h):
    """(h, W, 3) strip; slit profile softens top and bottom edges."""
    yy = np.linspace(-1, 1, h)
    slit = np.clip(1.25 - np.abs(yy) ** 6 * 1.25, 0, 1)[:, None]
    ground = hexrgb(st['ground'])
    if st.get('absorb'):
        cont = 0.12 + 0.88 * hue                           # continuum, then lines absorb it
        img = cont[None] * (1 - 0.94 * expo[None, :, None])
        img = img * slit[..., None] + ground * (1 - slit[..., None])
    elif st['hue']:
        col = hue * 0.85 + 0.15
        a = expo[None, :, None] * slit[..., None]
        img = ground * (1 - a) + col[None] * a
    else:
        tint = hexrgb(st['tint'])
        a = expo[None, :, None] * slit[..., None]
        img = ground * (1 - a) + tint * a
    return np.clip(img, 0, 1)


for style in args.styles.split(','):
    st = STY[style]
    plt.rcParams.update({'font.family': 'P052', 'text.color': st['ink']})
    nrow = len(specs)
    H_in = 3.1 * nrow + 2.9
    fig = plt.figure(figsize=(W / 100, H_in), dpi=100, facecolor=st['ground'])
    L, R = 0.155, 0.93
    fig.text(0.5, 1 - 0.55 / H_in, 'BULK AND OUTLIERS', ha='center', va='top', fontsize=44, color=st['ink'])
    fig.text(0.5, 1 - 1.35 / H_in, 'Hessian eigenvalue spectra of six networks trained to classify C digits  ·  '
             'count the isolated lines', ha='center', va='top', fontsize=19, color=st['faint'], style='italic')
    for i, s in enumerate(specs):
        top = 1 - (2.0 + 3.1 * i) / H_in
        # microdensitometer tracing (log count per column)
        axT = fig.add_axes([L, top - 0.9 / H_in, R - L, 0.8 / H_in], facecolor='none')
        dens = log_density(s['lam'], x0, x1, W // 2, sigma_px=2.5)
        axT.fill_between(np.linspace(x0, x1, W // 2), 0, np.log10(1 + dens), color=st['faint'], alpha=0.35, lw=0)
        axT.plot(np.linspace(x0, x1, W // 2), np.log10(1 + dens), color=st['ink'], lw=0.9)
        axT.set_xlim(x0, x1); axT.set_ylim(0, 3.9); axT.axis('off')
        # the plate
        axS = fig.add_axes([L, top - 2.15 / H_in, R - L, 1.2 / H_in])
        expo, _ = exposure_profile(s['lam'], x0, x1, W, sigma_px=1.3)
        img = strip_rgb(expo, st, 120)
        axS.imshow(img, extent=(x0, x1, 0, 1), aspect='auto', interpolation='lanczos')
        axS.set_xlim(x0, x1); axS.axis('off')
        # line identifications for the k lines above the widest gap
        top_l = np.sort(s['lam'])[::-1][:s['k']]
        for j, l in enumerate(top_l):
            xx = slog(l)
            axT.plot([xx, xx], [0.05, 0.55], color=st['ink'], lw=0.8)
        # labels
        fig.text(0.02, top - 1.35 / H_in, f'C = {s["C"]}', fontsize=34, color=st['ink'], va='center')
        fig.text(0.02, top - 1.95 / H_in, f'train loss {s["loss"]:.3f}', fontsize=13, color=st['faint'],
                 va='center', family='Nimbus Mono PS')
        fig.text(R + 0.008, top - 1.55 / H_in, f'{s["k"]}', fontsize=34, color=st['ink'], va='center')
        fig.text(R + 0.008, top - 1.95 / H_in, 'lines' if s['k'] != 1 else 'line', fontsize=13,
                 color=st['faint'], va='center', family='Nimbus Mono PS')
        fig.text(L, top - 2.42 / H_in, s['desc'], fontsize=12, color=st['faint'], va='center',
                 family='Nimbus Mono PS')
    # ruler: eigenvalue scale
    axR = fig.add_axes([L, 0.6 / H_in, R - L, 0.25 / H_in], facecolor='none')
    ticks = [-1e-1, -1e-2, 0, 1e-2, 1e-1, 1, 10] if args.tau >= 1e-3 else [-1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1, 10]
    tk = [t for t in ticks if x0 <= slog(t) <= x1]
    axR.set_xlim(x0, x1); axR.set_ylim(0, 1)
    for sp in ['top', 'left', 'right']:
        axR.spines[sp].set_visible(False)
    axR.spines['bottom'].set_color(st['faint'])
    axR.set_yticks([])
    axR.set_xticks(slog(np.array(tk)))
    axR.set_xticklabels([('0' if t == 0 else f'{t:g}') for t in tk], color=st['ink'], fontsize=14)
    minor = np.concatenate([m * 10.0 ** e for e in range(-5, 2) for m in [np.arange(2, 10)]])
    minor = np.concatenate([minor, -minor])
    axR.set_xticks(slog(minor[(slog(minor) > x0) & (slog(minor) < x1)]), minor=True)
    axR.tick_params(colors=st['faint'], which='both', direction='in')
    fig.text(0.5, 0.12 / H_in, f'Hessian eigenvalue λ  (symmetric-log axis, linear within ±{args.tau:g})', fontsize=12, color=st['faint'],
             family='Nimbus Mono PS', va='center', ha='center')
    out = f'{GAL}/plates_{args.src}_{style}.png'
    fig.savefig(out, dpi=100, facecolor=st['ground'])
    plt.close(fig)
    print('wrote', out)
