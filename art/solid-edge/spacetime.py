"""Space-time volume from cache (no compute): label volume of
art/trainability-fractal/cache/windows/steps_zoomA2_384.npz measure_T (100, 384, 384),
T = 10..1000, over the 10^1 zoom window. Checks against the source README and writes
cache/spacetime_labels.npz, cache/spacetime_summary.json, cache/preview/spacetime_*.png.

  OMP_NUM_THREADS=4 ../.venv/bin/python spacetime.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from se_analysis import boundary3d, dimension_of_measure, edges

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/home/fzeng/ml/research/art/trainability-fractal/cache/windows/steps_zoomA2_384.npz'
d = np.load(SRC)
MT = d['measure_T']                    # float32 (100, 384, 384); rows = eta1 (row 0 = bottom), cols = eta0
cps = d['checkpoints']
R = int(d['res'])
c0, c1, hw = float(d['c0']), float(d['c1']), float(d['hw'])

# ---- sign convention check (source: converged <-> negative measure) ----
checks = {}
checks['final_equals_T1000'] = bool(np.array_equal(np.sign(MT[-1]), np.sign(d['measure'].astype(np.float32))))
checks['measure_nonzero'] = bool((MT != 0).all())
# small-lr corner (bottom rows = smallest eta1) must be trainable under "negative = converged"
checks['low_lr_rows_conv_frac_T1000'] = float((MT[-1, :16] < 0).mean())
checks['high_lr_rows_conv_frac_T1000'] = float((MT[-1, -16:] < 0).mean())
L = (MT < 0).astype(np.uint8)          # 1 = converged (trainable)

rows = []
for i, T in enumerate(cps):
    f, s, c = dimension_of_measure(MT[i].astype(np.float64), 2, 48)
    rows.append(dict(T=int(T), conv=float(L[i].mean()), edge_px=int(edges(MT[i]).sum()), D=f['D'], se=f['se'], r2=f['r2']))
readme_conv = {10: 0.648, 100: 0.493, 1000: 0.489}
readme_D = {10: 1.05, 30: 1.27, 100: 1.37, 250: 1.42, 500: 1.41, 1000: 1.41}
cmp = []
for T in sorted(set(readme_conv) | set(readme_D)):
    r = rows[list(cps).index(T)]
    cmp.append(dict(T=T, conv=round(r['conv'], 4), conv_readme=readme_conv.get(T), D=round(r['D'], 3), D_readme=readme_D.get(T)))
B = boundary3d(L)
summary = dict(source=SRC, window=dict(c0=c0, c1=c1, hw=hw, res=R), checkpoints=cps.tolist(), checks=checks,
               compare_readme=cmp, per_T=rows, boundary_voxels_6nbr=int(B.sum()),
               boundary_voxel_frac=float(B.mean()), note='no 3D box dimension: T is not a length')
os.makedirs(os.path.join(HERE, 'cache', 'preview'), exist_ok=True)
np.savez_compressed(os.path.join(HERE, 'cache', 'spacetime_labels.npz'), labels=L, checkpoints=cps,
                    log10_eta0=c0 + ((np.arange(R) + 0.5) / R * 2 - 1) * hw,
                    log10_eta1=c1 + ((np.arange(R) + 0.5) / R * 2 - 1) * hw)
json.dump(summary, open(os.path.join(HERE, 'cache', 'spacetime_summary.json'), 'w'), indent=1)
for r in cmp:
    print(r)
print('checks', checks, 'boundary voxels', B.sum())

# ---- previews (matplotlib; labels drawn nearest-neighbour, two flat colours) ----
cm = matplotlib.colors.ListedColormap(['#c8324a', '#2b4f8c'])   # diverged, converged (declared)
ext = [c0 - hw, c0 + hw, c1 - hw, c1 + hw]
sel = [0, 2, 9, 24, 49, 99]
fig, axs = plt.subplots(2, 3, figsize=(12, 8.4), dpi=110)
for ax, i in zip(axs.flat, sel):
    ax.imshow(L[i], origin='lower', cmap=cm, vmin=0, vmax=1, interpolation='nearest', extent=ext)
    ax.set_title(f'T = {cps[i]}: trainable {100*rows[i]["conv"]:.1f}%, D2 = {rows[i]["D"]:.2f}', fontsize=9)
    ax.set_xlabel('log10 eta0'); ax.set_ylabel('log10 eta1')
fig.suptitle('Space-time volume, horizontal slices (blue = converged, red = diverged)')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'cache', 'preview', 'spacetime_T_slices.png')); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
yc = R // 2
axs[0].imshow(L[:, yc, :], origin='lower', cmap=cm, vmin=0, vmax=1, interpolation='nearest', aspect='auto',
              extent=[c0 - hw, c0 + hw, cps[0] - 5, cps[-1] + 5])
axs[0].set_title(f'vertical slice at log10 eta1 = {c1 + ((yc + .5) / R * 2 - 1) * hw:.3f}', fontsize=9)
axs[0].set_xlabel('log10 eta0'); axs[0].set_ylabel('training steps T (declared time axis)')
axs[1].imshow(L[:, :, R // 2], origin='lower', cmap=cm, vmin=0, vmax=1, interpolation='nearest', aspect='auto',
              extent=[c1 - hw, c1 + hw, cps[0] - 5, cps[-1] + 5])
axs[1].set_title(f'vertical slice at log10 eta0 = {c0 + ((R // 2 + .5) / R * 2 - 1) * hw:.3f}', fontsize=9)
axs[1].set_xlabel('log10 eta1'); axs[1].set_ylabel('T')
ax = axs[2]
ax.plot(cps, [r['conv'] for r in rows], color='#2b4f8c', label='trainable fraction')
ax.set_xscale('log'); ax.set_xlabel('T'); ax.set_ylabel('trainable fraction')
ax2 = ax.twinx(); ax2.plot(cps, [r['D'] for r in rows], color='#c8324a', label='2D box-count D (b=2-32)')
ax2.set_ylabel('D (per-T slice)')
ax.legend(loc='upper center', fontsize=8); ax2.legend(loc='center right', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'cache', 'preview', 'spacetime_vertical_and_curves.png')); plt.close(fig)
print('previews written')
