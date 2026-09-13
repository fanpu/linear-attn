"""Barcodes and gel lanes of the exact Hessian spectra (class-count series).

Barcode: each eigenvalue is a hairline bar at its symlog position (black on paper); bars merge into the bulk
block, outliers stand alone. Digits under each code = C and the measured outlier count.
Gel: the same spectra as vertical lanes, bands = eigenvalues with exposure saturation (declared idiom).

python render_barcode.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_common import SERIES, slog, load_exact, exposure_profile, hexrgb, GAL, n_structural

specs = [load_exact(C) for C in SERIES]
# barcodes show only the positive side above lambda >= 1e-4 (bulk block starts there); declared crop
x0, x1 = slog(3e-4), max(slog(s['H_eig'].max()) for s in specs) + 0.15


def barcode(style):
    W, rowh = 2400, 300
    paper = {'paper': '#f6f3ec', 'night': '#0a0a0a'}[style]
    ink = {'paper': '#111111', 'night': '#f2efe6'}[style]
    fig = plt.figure(figsize=(W / 100, (len(specs) * rowh + 260) / 100), dpi=100, facecolor=paper)
    Ht = len(specs) * rowh + 260
    fig.text(0.5, 1 - 60 / Ht, 'Hessian spectra, barcoded', ha='center', va='top', fontsize=34, color=ink,
             family='P052')
    for i, s in enumerate(specs):
        C, k = int(s['C']), n_structural(s)
        y0 = 1 - (160 + (i + 1) * rowh) / Ht
        ax = fig.add_axes([0.08, y0 + 70 / Ht, 0.84, (rowh - 110) / Ht], facecolor=paper)
        lam = s['H_eig']; lam = lam[slog(lam) > x0]
        expo, _ = exposure_profile(lam, x0, x1, W * 2, sigma_px=0.9, d0=0.25)
        bars = (expo > 0.5).astype(float)                         # hard threshold -> crisp bars
        rgb = hexrgb(paper) * (1 - bars[None, :, None]) + hexrgb(ink) * bars[None, :, None]
        ax.imshow(np.repeat(rgb, 2, 0), extent=(x0, x1, 0, 1), aspect='auto', interpolation='nearest')
        ax.set_xlim(x0, x1); ax.axis('off')
        # outliers (eigenvectors in span of class-mean gradients) drop below the code like EAN guard bars
        for l in np.sort(s['H_eig'])[::-1][:k]:
            ax.plot([slog(l)] * 2, [-0.12, 0.02], color=ink, lw=1.6, clip_on=False, solid_capstyle='butt')
        # guard bars at both ends, EAN-style
        for gx in (x0 + 0.01, x0 + 0.03, x1 - 0.03, x1 - 0.01):
            ax.plot([gx, gx], [-0.12, 1], color=ink, lw=2.2, clip_on=False)
        digits = f'{C:02d}  {k:02d}  {int(s["P"]):05d}'
        fig.text(0.5, y0 + 40 / Ht, digits, ha='center', va='center', fontsize=26, color=ink,
                 family='Nimbus Mono PS')
        fig.text(0.93, y0 + 40 / Ht, f'C={C} · {k} outlier{"s" if k != 1 else ""}', ha='right', va='center',
                 fontsize=12, color=ink, family='Nimbus Mono PS', alpha=0.7)
    fig.text(0.08, 30 / Ht, f'bars = exact eigenvalues λ ≥ 3e-4 (long bars: outliers), symlog position (τ=1e-4); digits: C, outlier count (eigvecs in span of class-mean gradients), P',
             fontsize=12, color=ink, family='Nimbus Mono PS', alpha=0.7)
    out = f'{GAL}/barcode_{style}.png'
    fig.savefig(out, dpi=100, facecolor=paper); plt.close(fig); print('wrote', out)


def gel():
    Hpx, lanew, gap = 2400, 150, 70
    n = len(specs)
    Wpx = n * lanew + (n + 1) * gap + 300
    agar = hexrgb('#0d1633')
    fig = plt.figure(figsize=(Wpx / 100, (Hpx + 400) / 100), dpi=100, facecolor='#05070f')
    Ht = Hpx + 400
    ax = fig.add_axes([0, 250 / Ht, 1, Hpx / Ht])
    img = np.ones((Hpx, Wpx, 3)) * agar
    yy = np.linspace(-1, 1, lanew)
    prof = np.clip(1.3 - np.abs(yy) ** 4 * 1.3, 0, 1)
    lo, hi = slog(-2e-2), max(slog(s['H_eig'].max()) for s in specs) + 0.2
    band = hexrgb('#ffb347'); core = hexrgb('#fff6e0')
    for i, s in enumerate(specs):
        _, dn = exposure_profile(s['H_eig'], lo, hi, Hpx, sigma_px=1.6)
        expo = np.clip(0.55 * (1 - np.exp(-dn / 0.7)) + 0.45 * np.log1p(dn) / np.log1p(40.0), 0, 1)
        expo = expo[::-1]                     # large eigenvalues at the top of the gel
        xL = 300 + gap + i * (lanew + gap)
        a = expo[:, None] * prof[None, :]
        col = band * a[..., None] + (core - band) * (a[..., None] ** 3)
        img[:, xL:xL + lanew] = np.clip(img[:, xL:xL + lanew] * (1 - a[..., None]) + col, 0, 1)
        fig.text((xL + lanew / 2) / Wpx, (250 + Hpx + 40) / Ht, f'C={int(s["C"])}', ha='center', fontsize=22,
                 color='#d7deef', family='Nimbus Sans')
        fig.text((xL + lanew / 2) / Wpx, 200 / Ht, f'{n_structural(s)}', ha='center', fontsize=26,
                 color='#ffb347', family='Nimbus Sans')
    ax.imshow(img, aspect='auto', extent=(0, Wpx, lo, hi), interpolation='bilinear')
    ax.set_xlim(0, Wpx); ax.set_ylim(lo, hi)
    for t in [-1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1, 10]:
        if lo < slog(t) < hi:
            ax.plot([230, 280], [slog(t)] * 2, color='#8a96b8', lw=1.5)
            ax.text(215, slog(t), '0' if t == 0 else f'{t:g}', ha='right', va='center', color='#8a96b8',
                    fontsize=16, family='Nimbus Sans')
    ax.axis('off')
    fig.text(0.5, 110 / Ht, 'lanes: exact Hessian spectra, band = eigenvalue (symlog); number = eigenvectors in span of class-mean gradients',
             ha='center', fontsize=13, color='#8a96b8', family='Nimbus Sans')
    fig.savefig(f'{GAL}/gel.png', dpi=100, facecolor='#05070f'); plt.close(fig); print('wrote gel')


barcode('paper'); barcode('night'); gel()
