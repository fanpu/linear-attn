"""Deep-zoom keyframe sequence with automatic boundary-rich target selection.

Keyframe k has half-width hw_k = hw_0 * 10^(-k*dec) decades (log10 learning-rate units),
centred at (c0_k, c1_k). The next centre is chosen greedily inside keyframe k: among
candidate sub-windows of the next size that lie inside the current window, pick the one
with the most boundary boxes at a 4-pixel box size (sign-change edges, his definition),
requiring both phases to occupy >=10% of the sub-window. With --path, centres are read
from an earlier run instead (used to recompute a path at higher resolution).

Per keyframe we also measure a precision floor: a 64x64 block around the next centre is
recomputed with eta0 and eta1 multiplied by (1 + 2^-52) (one ulp) and we record the
fraction of pixels whose converge/diverge label flips (overall, and among edge pixels).

  gpu_run.sh python zoom_compute.py --tag explore --res 256 --depth 14
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import tfractal as tf
from common_render import edges

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--nonlin', default='tanh')
p.add_argument('--res', type=int, default=256)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--dec', type=float, default=0.25, help='decades of zoom per keyframe')
p.add_argument('--depth', type=float, default=14.0)
p.add_argument('--c0', type=float, default=1.5)
p.add_argument('--c1', type=float, default=1.5)
p.add_argument('--hw', type=float, default=4.5)
p.add_argument('--path', default=None, help='json with centres to follow')
p.add_argument('--only', default=None, help='comma list of keyframe indices (with --path)')
p.add_argument('--minibatch', type=int, default=None)
p.add_argument('--chunk', type=int, default=32768)
p.add_argument('--floor', type=int, default=1, help='measure ulp flip floor')
p.add_argument('--device', default='cuda')
args = p.parse_args()

if args.device == 'cuda':
    tf.setup_gpu(0.10)
else:
    torch.set_num_threads(4)
out_dir = f'cache/zoom_{args.tag}'
os.makedirs(out_dir, exist_ok=True)
prob = tf.make_problem(0, nonlin=args.nonlin, device=args.device)
K = int(round(args.depth / args.dec)) + 1
ratio = 10 ** (-args.dec)
R = args.res

path = None
if args.path:
    path = json.load(open(args.path))['keyframes']
    K = len(path)
only = set(int(x) for x in args.only.split(',')) if args.only else None

meta_path = f'{out_dir}/path.json'
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(args=vars(args), keyframes=[])
c0, c1, hw = args.c0, args.c1, args.hw


def coherent_labels(M, min_size=6):
    """Phase labels with dust removed (connected components smaller than min_size px of
    either phase flipped). Used ONLY for choosing zoom targets, never for rendering."""
    from scipy import ndimage
    L = M < 0
    for ph in (True, False):
        mask = L if ph else ~L
        lab, nlab = ndimage.label(mask)
        if nlab == 0:
            continue
        sizes = np.bincount(lab.ravel())
        small = (sizes < min_size); small[0] = False
        L = L ^ small[lab]
    return L


def choose_next(M, hw, c0, c1):
    """Greedy target: the next-size sub-window with the most occupied 4-px boxes of
    *coherent* boundary (dust removed), both phases >= 15% of the sub-window, with a mild
    preference for staying central. (v1 counted raw edges and walked into a region of
    isolated diverged 'dust' pixels at eta0 ~ 1e5.5; kept as a documented negative.)"""
    R = M.shape[0]                            # keyframes may differ in resolution
    L = coherent_labels(M)
    E = edges(np.where(L, -1.0, 1.0))
    s = int(round(R * ratio))
    ncand = 13
    best = None
    for iy in np.linspace(0, R - s, ncand).astype(int):
        for ix in np.linspace(0, R - s, ncand).astype(int):
            e = E[iy:iy + s - 1, ix:ix + s - 1]
            fc = L[iy:iy + s, ix:ix + s].mean()
            if min(fc, 1 - fc) < 0.15:
                continue
            b = 4
            hh = (e.shape[0] // b) * b
            occ = e[:hh, :hh].reshape(hh // b, b, hh // b, b).any(axis=(1, 3)).sum()
            dist = math.hypot(ix + s / 2 - R / 2, iy + s / 2 - R / 2) / R
            score = occ * (1.0 - 0.3 * dist)
            if best is None or score > best[0]:
                best = (score, ix, iy)
    if best is None:
        return c0, c1, None
    _, ix, iy = best
    px = 2 * hw / R
    nc0 = c0 - hw + (ix + s / 2) * px
    nc1 = c1 - hw + (iy + s / 2) * px
    return nc0, nc1, float(best[0])


t_all = time.time()
for k in range(K):
    if path:
        c0, c1, hw = path[k]['c0'], path[k]['c1'], path[k]['hw']
        if only is not None and k not in only:
            continue
    fn = f'{out_dir}/kf_{k:03d}.npz'
    if os.path.exists(fn):
        d = np.load(fn)
        M = d['measure']
        print(f'[{k}] cached', flush=True)
    else:
        t0 = time.time()
        e0, e1 = tf.log_grid(c0, c1, hw, R, dev=args.device)
        M = tf.run_grid(prob, e0, e1, steps=args.steps, chunk=args.chunk,
                        minibatch=args.minibatch, verbose=False).reshape(R, R)
        dt = time.time() - t0
        # relative lr spacing between neighbouring pixels
        rel_sp = 2 * hw / R * math.log(10)
        np.savez(fn, measure=M, c0=c0, c1=c1, hw=hw, res=R, steps=args.steps,
                 nonlin=args.nonlin, seconds=dt, rel_spacing=rel_sp)
        fe = edges(M).mean()
        print(f'[{k}] c=({c0:.15f},{c1:.15f}) hw={hw:.3e} conv={np.mean(M<0):.3f} '
              f'edge_frac={fe:.4f} rel_sp={rel_sp:.2e} {dt:.0f}s ({R*R/dt:.0f} px/s)', flush=True)
    entry = dict(k=k, c0=c0, c1=c1, hw=hw, res=int(M.shape[0]))
    nc0 = nc1 = None
    if not path:
        nc0, nc1, score = choose_next(M, hw, c0, c1)
        entry['score'] = score
    # precision floor: 1-ulp perturbation on a 64x64 block of this keyframe's grid
    if args.floor and 'flip_frac' not in (meta['keyframes'][k] if k < len(meta['keyframes']) else {}):
        Rk = M.shape[0]
        B = min(64, Rk)
        e0, e1 = tf.log_grid(c0, c1, hw, Rk, dev=args.device)
        e0 = e0.view(Rk, Rk); e1 = e1.view(Rk, Rk)
        if not path and nc0 is not None:
            # block centred on the chosen next centre (where the boundary is)
            px = 2 * hw / Rk
            cx = int(round((nc0 - (c0 - hw)) / px)); cy = int(round((nc1 - (c1 - hw)) / px))
        else:
            cx = cy = Rk // 2
        x0 = min(max(cx - B // 2, 0), Rk - B); y0 = min(max(cy - B // 2, 0), Rk - B)
        sl = (slice(y0, y0 + B), slice(x0, x0 + B))
        ulp = 1.0 + 2.0 ** -52
        Mp = tf.run_grid(prob, (e0[sl] * ulp).reshape(-1), (e1[sl] * ulp).reshape(-1),
                         steps=args.steps, minibatch=args.minibatch, verbose=False).reshape(B, B)
        Mb = M[sl]
        flips = (Mp < 0) != (Mb < 0)
        Eb = np.zeros((B, B), bool)
        Eb[:-1, :-1] |= edges(Mb); Eb[1:, 1:] |= edges(Mb); Eb[1:, :-1] |= edges(Mb); Eb[:-1, 1:] |= edges(Mb)
        entry['flip_frac'] = float(flips.mean())
        entry['flip_frac_edge'] = float(flips[Eb].mean()) if Eb.any() else 0.0
        entry['edge_px_frac'] = float(Eb.mean())
        print(f'    ulp floor: flips {flips.mean():.4f} of block, {entry["flip_frac_edge"]:.4f} of edge px '
              f'(edge px {Eb.mean():.3f})', flush=True)
    if k < len(meta['keyframes']):
        old = meta['keyframes'][k]
        old.update(entry)
    else:
        meta['keyframes'].append(entry)
    json.dump(meta, open(meta_path, 'w'), indent=1)
    if not path:
        c0, c1, hw = nc0, nc1, hw * ratio
print(f'done in {time.time()-t_all:.0f}s', flush=True)
