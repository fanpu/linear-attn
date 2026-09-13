"""Frame sequences for films: an eta sweep (fixed window) or an exponential zoom (fixed eta).

  python compute_frames.py fact3 --s 0.5 eta --eta0 0.05 --eta1 1.32 --n 240 --win -3.5 3.5 -3.5 3.5 --res 1080 --tag f3_etafilm
  python compute_frames.py fact3 --s 0.5 --eta 1.1 zoom --center A B --half0 3.5 --half1 3e-8 --n 360 --res 1080 --tag f3_zoomfilm
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import gpu_setup, slice_grid
import cuda_gd, problems as PR

ap = argparse.ArgumentParser()
ap.add_argument('problem')
ap.add_argument('mode', choices=['eta', 'zoom'])
ap.add_argument('--eta', type=float, default=None)
ap.add_argument('--eta0', type=float); ap.add_argument('--eta1', type=float)
ap.add_argument('--win', type=float, nargs=4)
ap.add_argument('--center', type=float, nargs=2); ap.add_argument('--half0', type=float); ap.add_argument('--half1', type=float)
ap.add_argument('--start_center', type=float, nargs=2, default=None, help='frame-0 centre; drifts to --center as (half/half0)^2')
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--res', type=int, default=1080)
ap.add_argument('--T', type=int, default=20000)
ap.add_argument('--s', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--H', type=int, default=2)
ap.add_argument('--tag', required=True)
args = ap.parse_args()
gpu_setup()
d = f'cache/frames/{args.tag}'
os.makedirs(d, exist_ok=True)
json.dump(vars(args), open(f'{d}/args.json', 'w'))
if args.problem == 'fact3':
    prob, c0, u, v = PR.fact3(args.s)
else:
    prob, c0, u, v = PR.xor(args.seed, args.H)
R = args.res
T0 = time.time()
for i in range(args.n):
    f = f'{d}/{i:04d}.npz'
    if os.path.exists(f):
        continue
    x = i / max(args.n - 1, 1)
    if args.mode == 'eta':
        eta = args.eta0 + (args.eta1 - args.eta0) * x
        win = args.win
    else:
        eta = args.eta
        half = args.half0 * (args.half1 / args.half0) ** x
        ca, cb = args.center
        if args.start_center:
            w = (half / args.half0) ** 2
            ca, cb = ca + (args.start_center[0] - ca) * w, cb + (args.start_center[1] - cb) * w
        win = [ca - half, ca + half, cb - half, cb + half]
    th0 = slice_grid(c0, u, v, *win, R)
    if args.problem == 'fact3':
        r = cuda_gd.run('fact', th0, eta, args.T); cl = PR.fact3_classify(r['theta'], r['status'])
    else:
        r = cuda_gd.run('mlp', th0, eta, args.T, X=PR.XOR_X, Y=PR.XOR_T, H=args.H); cl = PR.xor_classify(prob, r['theta'], r['status'])
    extra = {}
    if args.problem == 'xor':
        extra['loss'] = r['loss'].astype(np.float32).reshape(R, R)
    np.savez_compressed(f, status=r['status'].reshape(R, R), tconv=r['tconv'].astype(np.float32).reshape(R, R),
                        raw=cl['raw'].astype(np.int16).reshape(R, R), canon=cl['canon'].astype(np.int16).reshape(R, R),
                        eta=eta, win=np.array(win), **extra)
    if i % 10 == 0:
        print(f'frame {i}/{args.n} eta={eta:.4f} win={win[1]-win[0]:.3e} elapsed={time.time()-T0:.0f}s', flush=True)
print('done', time.time() - T0)
