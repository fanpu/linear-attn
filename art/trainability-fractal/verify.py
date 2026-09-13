"""Verification (fractals doc s11): box counting per zoom level, local slopes versus
absolute box size across all zoom levels, 1-ulp precision floor, resolution check and
the quadratic null model. CPU only; reads cache, writes cache/verify_*.json and
gallery/verify_*.png.

  python verify.py zoomA null_quadratic
"""
import glob
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from boxcount import box_counts, fit_dimension
from common_render import edges

INK = '#1c1a1e'; ACC = '#c8324a'; ACC2 = '#1f6f9f'; GREY = '#8a847c'


def load_zoom(tag):
    fs = sorted(glob.glob(f'cache/zoom_{tag}/kf_*.npz'))
    out = []
    for f in fs:
        d = np.load(f)
        out.append(dict(k=int(f[-7:-4]), M=d['measure'], c0=float(d['c0']), c1=float(d['c1']),
                        hw=float(d['hw']), res=int(d['res'])))
    meta = json.load(open(f'cache/zoom_{tag}/path.json'))
    for o in out:
        if o['k'] < len(meta['keyframes']):
            o.update({kk: v for kk, v in meta['keyframes'][o['k']].items() if kk.startswith('flip') or kk == 'edge_px_frac'})
    return out


def analyse(tag, bmin=2, bmax_frac=8):
    kfs = load_zoom(tag)
    rows = []
    for o in kfs:
        E = edges(o['M'])
        R = o['M'].shape[0]
        s, c = box_counts(E)
        fit = fit_dimension(s, c, bmin, R // bmax_frac) if (c > 0).sum() > 2 and c[s == bmin].sum() > 0 else None
        eps = s * (2 * o['hw'] / R)                     # absolute box size in decades of lr
        rows.append(dict(k=o['k'], hw=o['hw'], res=R, zoom_decades=math.log10(kfs[0]['hw'] / o['hw']),
                         sizes=s.tolist(), counts=c.tolist(), eps=eps.tolist(), fit=fit,
                         conv=float((o['M'] < 0).mean()), edge_frac=float(E.mean()),
                         flip_frac_edge=o.get('flip_frac_edge'), flip_frac=o.get('flip_frac')))
    return rows


def local_slopes(sizes, counts):
    s = np.array(sizes, float); c = np.array(counts, float)
    ok = c > 0
    s, c = s[ok], c[ok]
    ls = -np.diff(np.log10(c)) / np.diff(np.log10(s))
    mid = np.sqrt(s[1:] * s[:-1])
    return mid, ls


def plot(tag_rows, fname, title):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), dpi=130)
    fig.patch.set_facecolor('#f4f0e6')
    for ax in axs:
        ax.set_facecolor('#f4f0e6')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    colors = [ACC, ACC2, GREY, '#2e8b57']
    for (tag, rows), col in zip(tag_rows, colors):
        # (a) box counts, normalised per level, vs box size in px
        ax = axs[0]
        for r in rows:
            s = np.array(r['sizes']); c = np.array(r['counts'])
            if c.sum() == 0:
                continue
            ax.plot(s, c, '-', color=col, alpha=0.35, lw=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('box side b (pixels)'); ax.set_ylabel('occupied boxes N(b)')
        ax.set_title('(a) box counts, one line per zoom level', loc='left')
        # (b) local slope vs absolute box size (decades of learning rate)
        ax = axs[1]
        for r in rows:
            mid, ls = local_slopes(r['sizes'], r['counts'])
            eps = mid * (2 * r['hw'] / r['res'])
            sel = (mid >= 2) & (mid <= r['res'] / 16)
            ax.plot(eps[sel], ls[sel], 'o', ms=2.5, color=col, alpha=0.6)
        ax.set_xscale('log')
        ax.set_ylim(0.5, 2.1)
        ax.axhline(1, color=GREY, lw=0.8, ls=':')
        ax.set_xlabel('absolute box side  (decades of learning rate)')
        ax.set_ylabel('local slope  -dlogN/dlog b')
        ax.set_title('(b) local box-counting slope across zoom levels', loc='left')
        # (c) fitted D and ulp-flip fraction vs zoom depth
        ax = axs[2]
        z = [r['zoom_decades'] for r in rows if r['fit']]
        D = [r['fit']['D'] for r in rows if r['fit']]
        ax.plot(z, D, '-o', color=col, ms=4, label=f'{tag}: D (fit b=2..R/8)')
        fl = [(r['zoom_decades'], r['flip_frac_edge']) for r in rows if r['flip_frac_edge'] is not None]
        if fl:
            ax2 = ax.twinx() if not hasattr(plot, '_ax2') else plot._ax2
            plot._ax2 = ax2
            ax2.plot([a for a, b in fl], [b for a, b in fl], '--s', color=col, ms=3, alpha=0.6,
                     label=f'{tag}: 1-ulp flip fraction of boundary px')
            ax2.set_ylim(0, 1); ax2.set_ylabel('fraction of boundary pixels flipped by 1 ulp')
    axs[2].set_xlabel('zoom depth (decades)'); axs[2].set_ylabel('box-counting dimension D')
    axs[2].set_ylim(0.8, 2.05)
    axs[2].set_title('(c) dimension and precision floor vs depth', loc='left')
    h1, l1 = axs[2].get_legend_handles_labels()
    if hasattr(plot, '_ax2'):
        h2, l2 = plot._ax2.get_legend_handles_labels(); h1 += h2; l1 += l2
        del plot._ax2
    axs[2].legend(h1, l1, fontsize=8, frameon=False, loc='lower left')
    fig.suptitle(title, x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    tags = sys.argv[1:]
    allrows = []
    for tag in tags:
        rows = analyse(tag)
        json.dump(rows, open(f'cache/verify_{tag}.json', 'w'), indent=1)
        allrows.append((tag, rows))
        for r in rows:
            f = r['fit']
            fs = f"D={f['D']:.3f}+-{f['se']:.3f} b={f['bmin']}..{f['bmax']} ({f['decades']:.2f} dec, r2={f['r2']:.4f})" if f else 'no boundary'
            print(f"{tag} k={r['k']:2d} zoom={r['zoom_decades']:5.1f} conv={r['conv']:.2f} edge={r['edge_frac']:.4f} {fs} flip_edge={r['flip_frac_edge']}")
    plot(allrows, f'gallery/verify_boxcount_{"_".join(tags)}.png',
         'Box counting of the trainability boundary: ' + ' vs '.join(tags))
