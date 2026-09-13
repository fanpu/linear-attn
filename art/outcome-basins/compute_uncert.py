"""Uncertainty exponent / riddled-basin test (Grebogi, McDonald, Ott & Yorke 1983).

For M random base points x in a window, and perturbation sizes eps (fraction of window width),
a point is eps-uncertain if the outcome at x - eps d, x, x + eps d are not all equal
(d a random unit direction in the slice).  f(eps) ~ eps^alpha, boundary dimension D = 2 - alpha.
Riddled / intermingled basins give alpha ~ 0 (f does not shrink as eps -> 0).

  python compute_uncert.py fact3 --s 1 --eta 1.1 --win -3 3 -3 3 --tag f3_e1.1
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import gpu_setup
import cuda_gd, problems as PR

ap = argparse.ArgumentParser()
ap.add_argument('problem')
ap.add_argument('--eta', type=float, required=True)
ap.add_argument('--win', type=float, nargs=4, required=True)
ap.add_argument('--M', type=int, default=100000)
ap.add_argument('--eps', type=float, nargs=3, default=[-1, -12, 23], help='log10 lo, log10 hi, count')
ap.add_argument('--T', type=int, default=20000)
ap.add_argument('--s', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--H', type=int, default=2)
ap.add_argument('--key', default='raw')
ap.add_argument('--tag', required=True)
args = ap.parse_args()
gpu_setup()
os.makedirs('cache/uncert', exist_ok=True)

if args.problem == 'fact3':
    prob, c0, u, v = PR.fact3(args.s)
    run = lambda th: cuda_gd.run('fact', th, args.eta, args.T)
    cls = lambda th, r: PR.fact3_classify(r['theta'], r['status'])[args.key]
else:
    prob, c0, u, v = PR.xor(args.seed, args.H)
    run = lambda th: cuda_gd.run('mlp', th, args.eta, args.T, X=PR.XOR_X, Y=PR.XOR_T, H=args.H)
    cls = lambda th, r: PR.xor_classify(prob, r['theta'], r['status'])[args.key]

rng = np.random.default_rng(12345)
a0, a1, b0, b1 = args.win
W = a1 - a0
A = rng.uniform(a0, a1, args.M); B = rng.uniform(b0, b1, args.M)
ang = rng.uniform(0, 2 * np.pi, args.M)
def theta(a, b):
    return c0[None, :] + a[:, None] * u[None, :] + b[:, None] * v[None, :]
t0 = time.time()
r = run(theta(A, B)); base = cls(None, r)
eps = 10 ** np.linspace(*args.eps[:2], int(args.eps[2]))
frac, frac_se = [], []
for e in eps:
    da, db = e * W * np.cos(ang), e * W * np.sin(ang)
    rp = run(theta(A + da, B + db)); cp = cls(None, rp)
    rm = run(theta(A - da, B - db)); cm = cls(None, rm)
    unc = (cp != base) | (cm != base)
    frac.append(unc.mean()); frac_se.append(np.sqrt(unc.mean() * (1 - unc.mean()) / args.M))
    print(f'eps={e:.2e} f={unc.mean():.5f}', flush=True)
frac = np.array(frac)
sel = frac > 20 / args.M
x, y = np.log10(eps[sel]), np.log10(frac[sel])
alpha = np.polyfit(x, y, 1)[0] if sel.sum() >= 3 else np.nan
print(f'alpha={alpha:.3f}  D=2-alpha={2 - alpha:.3f}  wall={time.time() - t0:.0f}s')
np.savez(f'cache/uncert/{args.tag}.npz', eps=eps, frac=frac, frac_se=np.array(frac_se), base=base,
         A=A, B=B, alpha=alpha, meta=json.dumps(vars(args)))
