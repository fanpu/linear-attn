"""2-D loss planes through three weight vectors (Garipov-style affine planes).

theta(x, y) = P0 + x u + y v, with u = (P1-P0)/|P1-P0| and v the Gram-Schmidt unit vector toward P2.
Coordinates are in raw (unnormalised) weight-space distance; all three nets have the same architecture
so no filter normalisation is needed for planes through trained points.

Planes:
  hero:  perm   = {A, B, pi(B)}          (train subset + full test)
         bezier = {A, B, C}  (+ projection of the quadratic Bezier, which lies exactly in this plane)
         bezm   = {A, pi(B), C'}
  width: perm plane for every width in cache/width_<ds>_w*_weights.pt (train subset only)

usage: python compute_planes.py hero mnist --res 181 --ntrain 10000
       python compute_planes.py width mnist --res 97 --ntrain 5000
"""
import sys, argparse, glob
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('kind', choices=['hero', 'width'])
ap.add_argument('ds')
ap.add_argument('--res', type=int, default=181)
ap.add_argument('--ntrain', type=int, default=10000)
ap.add_argument('--ntest', type=int, default=10000)
ap.add_argument('--test_planes', default='perm')
ap.add_argument('--pad', type=float, default=0.35)
ap.add_argument('--tag', default='')
ap.add_argument('--planes', default='perm,bezier,bezm')
ap.add_argument('--G', type=int, default=32)
args = ap.parse_args()
gpu_setup()
Xtr, ytr, Xte, yte = load_data(args.ds)
g = torch.Generator().manual_seed(0)
sub = torch.randperm(len(Xtr), generator=g)[:args.ntrain].to(DEV)
Xs, ys_ = Xtr[sub], ytr[sub]
Xte, yte = Xte[:args.ntest], yte[:args.ntest]


def dot(p, q):
    return sum((p[k].double() * q[k].double()).sum().item() for k in p)


@torch.no_grad()
def grid_eval(P0, u, v, xs, ys, X, y):
    res_l = np.zeros((len(ys), len(xs)))
    res_a = np.zeros((len(ys), len(xs)))
    coords = [(j, i, xx, yy) for j, yy in enumerate(ys) for i, xx in enumerate(xs)]
    for s in range(0, len(coords), args.G):
        cc = coords[s:s + args.G]
        c = torch.tensor([[q[2], q[3]] for q in cc], device=DEV, dtype=torch.float32)
        Ws = {k: P0[k][None] + c[:, 0].view(-1, *[1] * P0[k].dim()) * u[k][None]
                 + c[:, 1].view(-1, *[1] * P0[k].dim()) * v[k][None] for k in P0}
        l, a = eval_stack(Ws, X, y)
        for q, li, ai in zip(cc, l, a):
            res_l[q[0], q[1]] = li
            res_a[q[0], q[1]] = ai
    return res_l, res_a


def plane(P0, P1, P2, name, out, log, test=True, extra_pts=None):
    u = {k: (P1[k] - P0[k]).double() for k in P0}
    nu = math.sqrt(dot(u, u))
    u = {k: x / nu for k, x in u.items()}
    w = {k: (P2[k] - P0[k]).double() for k in P0}
    c = dot(w, u)
    v = {k: w[k] - c * u[k] for k in P0}
    nv = math.sqrt(dot(v, v))
    v = {k: x / nv for k, x in v.items()}
    u = {k: x.float() for k, x in u.items()}
    v = {k: x.float() for k, x in v.items()}
    pts = np.array([[0, 0], [nu, 0], [c, nv]])
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).max()
    ctr = (lo + hi) / 2
    half = span * (0.5 + args.pad)
    xs = np.linspace(ctr[0] - half, ctr[0] + half, args.res)
    ys = np.linspace(ctr[1] - half, ctr[1] + half, args.res)
    t = time.time()
    out[f'{name}_xs'], out[f'{name}_ys'], out[f'{name}_pts'] = xs, ys, pts
    out[f'{name}_train'], out[f'{name}_train_acc'] = grid_eval(P0, u, v, xs, ys, Xs, ys_)
    if test:
        out[f'{name}_test'], out[f'{name}_test_acc'] = grid_eval(P0, u, v, xs, ys, Xte, yte)
    log(f'  plane {name}: {args.res}^2 in {time.time() - t:.0f}s; train loss min {out[f"{name}_train"].min():.4f} '
        f'max {out[f"{name}_train"].max():.2f}; |P1-P0|={nu:.2f}')
    return u, v


if args.kind == 'hero':
    tag = f'planes_hero_{args.ds}{args.tag}'
    log = Logger(os.path.join(CACHE, tag + '.log'))
    Wt = torch.load(os.path.join(CACHE, f'hero_{args.ds}{args.tag}_weights.pt'), weights_only=False)
    A, B, Bp, C, Cp = [to_dev(Wt[k]) for k in ['A', 'B', 'Bp', 'C', 'Cp']]
    out = dict(res=args.res, ntrain=args.ntrain, ntest=args.ntest)
    todo = args.planes.split(',')
    for name, (P0, P1, P2, ctrl) in {'perm': (A, B, Bp, None), 'bezier': (A, B, C, C), 'bezm': (A, Bp, Cp, Cp)}.items():
        if name not in todo:
            continue
        u, v = plane(P0, P1, P2, name, out, log, test=name in args.test_planes.split(','))
        if ctrl is not None:
            cur = []
            for t in np.linspace(0, 1, 201):
                q = bezier(P0, ctrl, P1, t)
                d = {k: (q[k] - P0[k]).double() for k in P0}
                cur.append([dot(d, {k: x.double() for k, x in u.items()}), dot(d, {k: x.double() for k, x in v.items()})])
            out[f'{name}_curve'] = np.array(cur)
        np.savez_compressed(os.path.join(CACHE, tag + '.npz'), **out)
else:
    tag = f'planes_width_{args.ds}'
    log = Logger(os.path.join(CACHE, tag + '.log'))
    path = os.path.join(CACHE, tag + '.npz')
    out = dict(np.load(path)) if os.path.exists(path) else dict(res=args.res, ntrain=args.ntrain)
    files = sorted(glob.glob(os.path.join(CACHE, f'width_{args.ds}_w*_weights.pt')),
                   key=lambda f: int(f.split('_w')[-1].split('_')[0]))
    for f in files:
        w = int(f.split('_w')[-1].split('_')[0])
        if f'w{w}_train' in out:
            continue
        Wt = torch.load(f, weights_only=False)
        A, B, Bp = [to_dev(Wt[k]) for k in ['A', 'B', 'Bp']]
        plane(A, B, Bp, f'w{w}', out, log, test=False)
        np.savez_compressed(path, **out)
log('done')
