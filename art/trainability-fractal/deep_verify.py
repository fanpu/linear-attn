"""Box counting on a native 1024^2 (or larger) window over >2 decades of box size,
plus a consistency check against the 256^2 keyframe of the same window.

  python deep_verify.py deep_zoomA4_1024_f64 zoomA:4
Writes cache/verify_deep_<name>.json and gallery/verify_deep_<name>.png.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from boxcount import box_counts, fit_dimension
from common_render import edges
from render_windows import load

name = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else None
w = load(name)
M = w['M']
R = M.shape[0]
E = edges(M)
s, c = box_counts(E)
out = dict(name=name, res=R, conv_frac=float((M < 0).mean()), edge_px=int(E.sum()),
           seconds=float(w.get('seconds', np.nan)), dtype=str(w.get('dtype', '')))
fits = {}
for lo, hi in [(2, R // 4), (2, 32), (8, 128), (32, R // 4), (4, R // 8)]:
    f = fit_dimension(s, c, lo, hi)
    fits[f'{lo}-{hi}'] = f
out['fits'] = fits
ls = np.diff(np.log10(c)) / np.diff(np.log10(1.0 / s))
out['local_slopes'] = {f'{int(s[i])}-{int(s[i+1])}': float(ls[i]) for i in range(len(ls))}
out['sizes'] = s.tolist(); out['counts'] = c.tolist()

if ref is not None:
    r = load(ref)
    Mr = r['M']
    k = R // Mr.shape[0]
    # nearest sample of the native grid at the coarse cell centres (cell k/2 offset)
    Ms = M[k // 2::k, k // 2::k]
    Er = edges(Mr); Es = edges(Ms)
    sr, cr = box_counts(Er); ss, cs = box_counts(Es)
    out['ref'] = dict(name=ref, res=int(Mr.shape[0]), conv_frac=float((Mr < 0).mean()),
                      conv_frac_subsampled=float((Ms < 0).mean()),
                      label_agreement=float(((Mr < 0) == (Ms < 0)).mean()),
                      D_ref=fit_dimension(sr, cr, 2, Mr.shape[0] // 8),
                      D_subsampled=fit_dimension(ss, cs, 2, Mr.shape[0] // 8),
                      edge_px_ref=int(Er.sum()), edge_px_subsampled=int(Es.sum()))

json.dump(out, open(f'cache/verify_deep_{name}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('sizes', 'counts')}, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
f = fits[f'2-{R // 4}']
x = np.log10(1.0 / s)
ax[0].plot(x, np.log10(c), 'o', color='#5e4fa2', label=f'native {R}²')
xx = np.linspace(x.min(), x.max(), 10)
sel = (s >= 2) & (s <= R // 4)
b0 = np.mean(np.log10(c[sel]) - f['D'] * x[sel])
ax[0].plot(xx, f['D'] * xx + b0, '-', color='#9e0142', lw=1,
           label=f"fit b=2–{R//4} px ({f['decades']:.1f} dec): D={f['D']:.2f}±{f['se']:.2f}")
ax[0].plot(xx, 1 * xx + np.log10(c[sel]).mean() - x[sel].mean(), ':', color='grey', lw=1, label='slope 1 (smooth curve)')
ax[0].plot(xx, 2 * xx + np.log10(c[sel]).mean() - 2 * x[sel].mean(), '--', color='grey', lw=1, label='slope 2 (area)')
ax[0].set_xlabel('log10 (1 / box size in px)'); ax[0].set_ylabel('log10 N(b)  occupied boxes')
ax[0].legend(fontsize=8, frameon=False)
mid = np.sqrt(s[:-1] * s[1:])
ax[1].semilogx(mid, ls, 'o-', color='#5e4fa2')
ax[1].axhline(1, color='grey', ls=':'); ax[1].axhline(2, color='grey', ls='--')
ax[1].set_xlabel('box size b (px, native grid)'); ax[1].set_ylabel('local slope')
ax[1].set_ylim(0.8, 2.1)
ax[1].set_title('local slope between successive box sizes', fontsize=9)
fig.suptitle(f'{name}: box counting on the converge/diverge edge set', fontsize=10)
fig.tight_layout()
fig.savefig(f'gallery/verify_deep_{name}.png', dpi=150)
