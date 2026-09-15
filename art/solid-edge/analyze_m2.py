"""M2 verification (spec §11) and previews, CPU only; reads cache/vol/*.

  OMP_NUM_THREADS=4 ../.venv/bin/python analyze_m2.py

-> cache/m2_summary.json, cache/preview/m2_*.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

from se_analysis import box_counts, box_counts3d, boundary3d, edges, edges3d, fit_dimension, local_slopes

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, 'cache', 'vol')
PV = os.path.join(HERE, 'cache', 'preview')
SRCW = '/home/fzeng/ml/research/art/trainability-fractal/cache/windows'
CM = matplotlib.colors.ListedColormap(['#c8324a', '#2b4f8c'])   # diverged, converged (declared)
os.makedirs(PV, exist_ok=True)
S = {}


def load(name):
    fn = os.path.join(V, f'{name}_final.npz')
    if not os.path.exists(fn):
        return None
    d = np.load(fn)
    return dict(M=d['measure'], L=(d['measure'] < 0), mask=d['f64_mask'], meta=json.loads(str(d['meta'])),
                grid=json.loads(str(d['grid'])), le0=np.log10(d['eta0']), le1=np.log10(d['eta1']), ls=np.log10(d['sigma']))


def d3_table(L, ranges):
    E = edges3d(L)
    s, c = box_counts3d(E)
    fits = {f'{a}-{b}': fit_dimension(s, c, a, b) for a, b in ranges}
    return dict(sizes=s.tolist(), counts=c.tolist(), local_slopes=local_slopes(s, c).tolist(), fits=fits,
                edge_cells=int(E.sum()), boundary_voxels_6nbr=int(boundary3d(L).sum()), conv=float(L.mean()))


def oblique(L, n_slices=12, seed=0, fit=None):
    """12 random oblique planes, each through a uniformly random 2x2x2 edge cell (so every slice meets
    the boundary) with a random orientation, redrawn until the square lies inside the volume;
    nearest-voxel sampling at voxel spacing.
    Slice side R/2+1 -> edge image R/2 (tiles exactly for b <= R/8)."""
    R = L.shape[0]
    side = R // 2 + 1
    fit = fit or ((2, 16) if R >= 256 else (1, 16))
    cand = np.argwhere(edges3d(L))
    rng = np.random.default_rng(seed)
    out, imgs = [], []
    if len(cand) == 0:
        return dict(side=side, fit=list(fit), slices=[], D_mean=None, D_sd=None, one_plus_D_mean=None, n_valid=0), []
    h = side / 2 - 0.5
    for q in range(n_slices):
        for tries in range(100000):      # rejection: random edge cell + random orientation until the square fits
            n = rng.normal(size=3); n /= np.linalg.norm(n)
            a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, a); u /= np.linalg.norm(u); v = np.cross(n, u)
            c = cand[rng.integers(len(cand))] + 0.5
            corners = np.array([c + su * h * u + sv * h * v for su in (-1, 1) for sv in (-1, 1)])
            if corners.min() >= 0 and corners.max() <= R - 1:
                break
        else:
            break
        x = np.arange(side) - side / 2 + 0.5
        X, Y = np.meshgrid(x, x, indexing='ij')
        P = c[None, None, :] + X[..., None] * u + Y[..., None] * v
        idx = np.rint(P).astype(int)
        img = L[idx[..., 0], idx[..., 1], idx[..., 2]]
        E = edges(np.where(img, -1.0, 1.0))
        s, cnt = box_counts(E)
        f = fit_dimension(s, cnt, *fit) if (cnt[s == fit[0]].sum() > 0 and (cnt > 0).sum() >= 4) else None
        f2 = fit_dimension(s, cnt, 1, R // 8) if f else None
        out.append(dict(normal=n.round(4).tolist(), center=c.tolist(), conv=float(img.mean()), edge_px=int(E.sum()),
                        D=(f['D'] if f else None), se=(f['se'] if f else None), D_b1_R8=(f2['D'] if f2 else None), tries=tries + 1))
        imgs.append(img)
    Ds = np.array([o['D'] for o in out if o['D'] is not None])
    D2 = np.array([o['D_b1_R8'] for o in out if o['D_b1_R8'] is not None])
    return dict(side=side, fit=list(fit), slices=out, D_mean=float(Ds.mean()), D_sd=float(Ds.std(ddof=1)),
                one_plus_D_mean=float(1 + Ds.mean()), n_valid=int(len(Ds)),
                one_plus_D_b1_R8_mean=float(1 + D2.mean())), imgs


def six_slices(v, name, title):
    L = v['L']; R = L.shape[0]
    E = edges3d(L)
    coords = [v['ls'], v['le1'], v['le0']]
    names = ['log10 sigma', 'log10 eta1', 'log10 eta0']
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.8), dpi=110)
    planes = []
    for ax in range(3):
        per = E.sum(axis=tuple(a for a in range(3) if a != ax))
        kmax = int(np.argmax(per))
        k2 = R // 2 if abs(R // 2 - kmax) > R // 16 else R // 4
        planes += [(ax, k2), (ax, kmax)]
    for n_, (ax, k) in enumerate(planes):
        a = axs[n_ % 2, ax]
        other = [q for q in range(3) if q != ax]
        yv, xv = coords[other[0]], coords[other[1]]
        img = np.take(L, k, axis=ax)
        a.imshow(img, origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', aspect='auto',
                 extent=[xv[0], xv[-1], yv[0], yv[-1]])
        a.set_xlabel(names[other[1]]); a.set_ylabel(names[other[0]])
        a.set_title(f'{names[ax]} = {coords[ax][k]:.3f} (plane {k}), conv {100*img.mean():.0f}%', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(PV, f'm2_{name}_slices.png')); plt.close(fig)


vols = {n: load(n) for n in ['A128', 'nullA128', 'B256', 'nullB256', 'B256_sub64', 'nullB256_sub64']}
pt = os.path.join(V, 'probe_table.json')
if os.path.exists(pt):
    S['probes'] = json.load(open(pt))
    for f in ['B_window.json', 'null_window.json']:
        if os.path.exists(os.path.join(V, f)):
            S[f[:-5]] = json.load(open(os.path.join(V, f)))

for name, v in vols.items():
    if v is None:
        continue
    R = v['L'].shape[0]
    rng = [(1, 16), (2, 16)] if R == 64 else ([(1, 32), (2, 32), (1, 16)] if R == 128 else [(1, 64), (2, 16), (1, 32), (2, 32), (2, 64), (4, 64)])
    S[name] = dict(res=R, grid={k: v['grid'][k] for k in v['grid'] if k not in ('overview_pix',)},
                   f32=v['meta'].get('f32'), shell=v['meta'].get('shell'), audit=v['meta'].get('audit'),
                   plane=v['meta'].get('plane'), final_conv=float(v['L'].mean()), f64_frac=float(v['mask'].mean()),
                   boxcount=d3_table(v['L'], rng))
    if R >= 128:
        ob, imgs = oblique(v['L'])
        S[name]['oblique'] = ob
    if R >= 128 and imgs:
        S[name]['oblique'] = ob
        fig, axs = plt.subplots(3, 4, figsize=(12, 9.4), dpi=100)
        for a, img, o in zip(axs.flat, imgs, ob['slices']):
            a.imshow(img, origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest')
            a.set_title(f'n={np.round(o["normal"], 2).tolist()} D={o["D"]:.2f}' if o['D'] else 'no boundary', fontsize=7)
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'{name}: 12 random oblique slices ({ob["side"]}^2 voxel-spaced, nearest voxel); '
                     f'1 + mean D = {ob["one_plus_D_mean"]:.2f}', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(PV, f'm2_{name}_oblique.png')); plt.close(fig)
    if R >= 128:
        six_slices(v, name, f'{name} ({R}^3, float32 + float64 shell): axis slices, blue = converged, red = diverged')

# --- A: sigma = 1 plane vs float64 overview ---
A = vols['A128']
if A is not None:
    g = json.load(open(os.path.join(V, 'A128', 'grid.json')))
    k1, pix = g['sigma_plane'], np.array(g['overview_pix'])
    ref = np.load(f'{SRCW}/hero_overview_tanh_1024_f64.npz')['measure']
    Epad = np.zeros_like(ref, bool); Epad[:-1, :-1] = edges(ref)
    near = maximum_filter(Epad.astype(np.uint8), size=17)[np.ix_(pix, pix)] > 0     # within 8 px = one A voxel
    sub = ref[np.ix_(pix, pix)] < 0
    fin = A['L'][k1]
    f32 = np.load(os.path.join(V, 'A128', 'f32.npy'))[k1] < 0
    res = dict(sigma_plane=k1, n=int(sub.size), n_near=int(near.sum()))
    for tag, P in [('final', fin), ('f32', f32)]:
        dis = P != sub
        res[tag] = dict(agreement=float(1 - dis.mean()), disagreements=int(dis.sum()), disagree_near=int((dis & near).sum()),
                        disagree_rate_near=float((dis & near).sum() / max(1, near.sum())), disagree_far=int((dis & ~near).sum()))
    S['A128']['sigma1_vs_overview'] = res
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), dpi=100)
    ext = [A['le0'][0], A['le0'][-1], A['le1'][0], A['le1'][-1]]
    for a, img, t in [(axs[0], sub, 'float64 1024^2 overview, every 8th pixel'), (axs[1], fin, 'A128 sigma = 1 plane (f32 + f64 shell)')]:
        a.imshow(img, origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', extent=ext); a.set_title(t, fontsize=9)
    rgb = np.full(sub.shape + (3,), 0.93); rgb[near] = 0.75; rgb[fin != sub] = 0.1
    axs[2].imshow(rgb, origin='lower', interpolation='nearest', extent=ext)
    axs[2].set_title(f'disagreements (black) {res["final"]["disagreements"]}/{res["n"]}; grey = near boundary', fontsize=9)
    for a in axs:
        a.set_xlabel('log10 eta0'); a.set_ylabel('log10 eta1')
    fig.tight_layout(); fig.savefig(os.path.join(PV, 'm2_A128_sigma1_vs_overview.png')); plt.close(fig)

# --- resolution doubling ---
for par, sub in [('B256', 'B256_sub64'), ('nullB256', 'nullB256_sub64')]:
    P, Q = vols[par], vols[sub]
    if P is None or Q is None:
        continue
    g = json.load(open(os.path.join(V, sub, 'grid.json')))
    kb, ib, jb = g['block']
    blk = P['L'][kb * 32:kb * 32 + 32, ib * 32:ib * 32 + 32, jb * 32:jb * 32 + 32]
    fine = Q['L']
    maj = fine.reshape(32, 2, 32, 2, 32, 2).mean(axis=(1, 3, 5))
    decided = maj != 0.5
    e32, e64 = int(edges3d(blk).sum()), int(edges3d(fine).sum())
    s32, c32 = box_counts3d(edges3d(blk), [1, 2, 4, 8])
    s64, c64 = box_counts3d(edges3d(fine), [1, 2, 4, 8, 16])
    res = dict(block=[kb, ib, jb], conv_1x=float(blk.mean()), conv_2x=float(fine.mean()), edge_cells_1x=e32, edge_cells_2x=e64,
               edge_scaling_exponent=float(np.log2(e64 / max(1, e32))),
               D_1x_b1_8=fit_dimension(s32, c32, 1, 8), D_2x_b1_16=fit_dimension(s64, c64, 1, 16),
               D_2x_b2_16=fit_dimension(s64, c64, 2, 16),
               majority_agreement=float(((maj > 0.5) == blk)[decided].mean()), ties=float((~decided).mean()))
    S[par]['resolution_doubling'] = res
    mid = 16
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), dpi=110)
    axs[0].imshow(blk[mid], origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest')
    axs[0].set_title(f'{par} block {g["block"]}, 32^3, sigma plane {mid}', fontsize=9)
    axs[1].imshow(fine[2 * mid], origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest')
    axs[1].set_title(f'same extent recomputed at 64^3, plane {2*mid} (edge cells x{e64/max(1,e32):.2f})', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(PV, f'm2_{par}_resdoubling.png')); plt.close(fig)

# --- B plane check preview ---
for name in ['B256', 'nullB256']:
    v = vols[name]
    if v is None or not v['meta'].get('plane'):
        continue
    pl = v['meta']['plane']; k = pl['sigma_plane_index']
    P64 = np.load(os.path.join(V, name, 'plane_f64.npy')) < 0
    f32 = np.load(os.path.join(V, name, 'f32.npy'))[k] < 0
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), dpi=110)
    ext = [v['le0'][0], v['le0'][-1], v['le1'][0], v['le1'][-1]]
    axs[0].imshow(P64, origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', extent=ext)
    axs[0].set_title(f'{name}: float64 recompute of plane k={k} (log10 sigma {pl["log10_sigma"]:.3f})', fontsize=9)
    axs[1].imshow(v['L'][k], origin='lower', cmap=CM, vmin=0, vmax=1, interpolation='nearest', extent=ext)
    axs[1].set_title(f'final volume (f32 + f64 shell): agree {100*pl["agree_final"]:.2f}%', fontsize=9)
    rgb = np.full(P64.shape + (3,), 0.93); rgb[f32 != P64] = (0.8, 0.5, 0.1); rgb[v['L'][k] != P64] = 0.1
    axs[2].imshow(rgb, origin='lower', interpolation='nearest', extent=ext)
    axs[2].set_title(f'orange: f32 flips ({100*(1-pl["agree_f32"]):.2f}%); black: final misses', fontsize=9)
    for a in axs:
        a.set_xlabel('log10 eta0'); a.set_ylabel('log10 eta1')
    fig.tight_layout(); fig.savefig(os.path.join(PV, f'm2_{name}_plane_check.png')); plt.close(fig)

# --- box-count figure ---
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110)
cols = {'A128': '#2b4f8c', 'nullA128': '#9fb3d1', 'B256': '#c8324a', 'nullB256': '#8a847c', 'B256_sub64': '#e08a3c', 'nullB256_sub64': '#555555'}
for name, col in cols.items():
    if name not in S:
        continue
    b = S[name]['boxcount']; s = np.array(b['sizes']); c = np.array(b['counts'])
    axs[0].loglog(s, c, 'o-', color=col, label=name)
    axs[1].semilogx(np.sqrt(s[1:] * s[:-1]), b['local_slopes'], 'o-', color=col, label=name)
axs[0].set_xlabel('box side b (voxels)'); axs[0].set_ylabel('occupied boxes'); axs[0].legend(fontsize=8)
axs[1].axhline(2, color='k', lw=0.5); axs[1].set_xlabel('b'); axs[1].set_ylabel('local slope'); axs[1].legend(fontsize=8)
fig.suptitle('3D box counting on 2x2x2 edge cells (final labels)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(PV, 'm2_boxcount.png')); plt.close(fig)

json.dump(S, open(os.path.join(HERE, 'cache', 'm2_summary.json'), 'w'), indent=1)
for name in cols:
    if name in S:
        b = S[name]['boxcount']
        print(name, f'conv={b["conv"]:.4f} edge={b["edge_cells"]} B6={b["boundary_voxels_6nbr"]}',
              {k: f'{f["D"]:.3f}±{f["se"]:.3f}' for k, f in b['fits'].items()}, 'slopes', np.round(b['local_slopes'], 2).tolist())
        if 'oblique' in S[name]:
            o = S[name]['oblique']; print('   oblique', {k: o[k] for k in o if k != 'slices'})
        if 'resolution_doubling' in S[name]:
            print('   resdoubling', {k: v for k, v in S[name]['resolution_doubling'].items() if not isinstance(v, dict)})
        print('   shell', S[name]['shell'] and {k: v for k, v in S[name]['shell'].items() if k != 'rounds'},
              'audit', S[name]['audit'] and {k: S[name]['audit'][k] for k in ('voxels', 'flips', 'flip_rate')})
if 'A128' in S and 'sigma1_vs_overview' in S['A128']:
    print('A sigma1', S['A128']['sigma1_vs_overview'])
