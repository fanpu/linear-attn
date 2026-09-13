"""Nested zoom sequence with centres chosen on basin boundaries.

Level 0 is the given window.  At each level the next centre is the pixel, inside the central
half of the current image, whose (R/factor)-sized neighbourhood contains the most distinct outcome
labels (ties -> most boundary pixels).  So every centre sits on a multi-basin boundary by
construction, and the deepest centre is inside every window (usable as a zoom-film target).

  python compute_zoom.py fact3 --s 0.5 --eta 1.1 --win -3.5 3.5 -3.5 3.5 --levels 9 --factor 10 --res 1024 --tag zA
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import gpu_setup, slice_grid
import cuda_gd, problems as PR
from fractal import boundary

ap = argparse.ArgumentParser()
ap.add_argument('problem')
ap.add_argument('--eta', type=float, required=True)
ap.add_argument('--win', type=float, nargs=4, required=True)
ap.add_argument('--levels', type=int, default=8)
ap.add_argument('--factor', type=float, default=10)
ap.add_argument('--res', type=int, default=1024)
ap.add_argument('--T', type=int, default=20000)
ap.add_argument('--s', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--H', type=int, default=2)
ap.add_argument('--key', default='raw')
ap.add_argument('--centers', type=float, nargs='*', default=None, help='fixed centres a1 b1 a2 b2 ... (skip auto choice)')
ap.add_argument('--tag', required=True)
args = ap.parse_args()
gpu_setup()
os.makedirs('cache/zoom', exist_ok=True)
if args.problem == 'fact3':
    prob, c0, u, v = PR.fact3(args.s)
else:
    prob, c0, u, v = PR.xor(args.seed, args.H)

a0, a1, b0, b1 = args.win
ca, cb, half = (a0 + a1) / 2, (b0 + b1) / 2, (a1 - a0) / 2
R = args.res
levels = []
for L in range(args.levels):
    t0 = time.time()
    th0 = slice_grid(c0, u, v, ca - half, ca + half, cb - half, cb + half, R)
    if args.problem == 'fact3':
        r = cuda_gd.run('fact', th0, args.eta, args.T)
        cl = PR.fact3_classify(r['theta'], r['status'])
    else:
        r = cuda_gd.run('mlp', th0, args.eta, args.T, X=PR.XOR_X, Y=PR.XOR_T, H=args.H)
        cl = PR.xor_classify(prob, r['theta'], r['status'])
    lab = cl[args.key].reshape(R, R)
    out = f'cache/zoom/{args.tag}_L{L}.npz'
    meta = dict(vars(args), level=L, center=[ca, cb], half=half, wall=time.time() - t0)
    np.savez_compressed(out, status=r['status'].reshape(R, R), tconv=r['tconv'].astype(np.float32).reshape(R, R),
                        **{k: x.reshape(R, R) for k, x in cl.items()}, center=c0, u=u, v=v,
                        win=np.array([ca - half, ca + half, cb - half, cb + half]), meta=json.dumps(meta))
    nb = int(boundary(lab).sum())
    print(f'L{L} centre=({ca:.15g},{cb:.15g}) half={half:.3e} boundary_px={nb} labels={len(np.unique(lab))} '
          f'frac={[round(float((r["status"] == k).mean()), 3) for k in range(3)]} {time.time() - t0:.1f}s', flush=True)
    if L == args.levels - 1:
        break
    if args.centers:
        ca, cb = args.centers[2 * L], args.centers[2 * L + 1]
    else:
        w = max(int(R / args.factor), 3)
        labs = np.unique(lab)
        cnt = np.zeros((R, R))
        for k in labs:
            cnt += ndimage.maximum_filter((lab == k).astype(np.uint8), size=w)
        bnd = ndimage.uniform_filter(boundary(lab).astype(float), size=w)
        score = bnd / max(bnd.max(), 1e-9) + 0.25 * cnt / len(labs)
        q = R // 4
        sub = score[q:R - q, q:R - q]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        i += q; j += q
        # snap to the nearest boundary pixel
        by, bx = np.nonzero(boundary(lab))
        k = np.argmin((by - i) ** 2 + (bx - j) ** 2)
        i, j = by[k], bx[k]
        ca = ca - half + 2 * half * j / (R - 1)
        cb = cb - half + 2 * half * i / (R - 1)
    half /= args.factor
