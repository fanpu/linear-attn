"""Compute one basin map (a 2D slice of initialisation space) and cache raw measurements.

  python compute_map.py fact3 --s 1 --eta 1.0 --win -3 3 -3 3 --res 1024 --tag f3_wide
  python compute_map.py xor --seed 0 --eta 3 --win -3 3 -3 3 --res 1024 --tag xor_e3
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import gpu_setup, slice_grid, DEV, DT
import cuda_gd
import problems as PR

ap = argparse.ArgumentParser()
ap.add_argument('problem')
ap.add_argument('--eta', type=float, nargs='+', required=True)
ap.add_argument('--win', type=float, nargs=4, required=True, metavar=('A0', 'A1', 'B0', 'B1'))
ap.add_argument('--res', type=int, default=512)
ap.add_argument('--T', type=int, default=20000)
ap.add_argument('--tol', type=float, default=1e-12)
ap.add_argument('--s', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--H', type=int, default=2)
ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--tag', required=True)
ap.add_argument('--save_theta', action='store_true')
args = ap.parse_args()

gpu_setup()
os.makedirs('cache/maps', exist_ok=True)
if args.problem == 'fact3':
    prob, c0, u, v = PR.fact3(args.s)
else:
    prob, c0, u, v = PR.xor(args.seed, args.H, args.scale)
a0, a1, b0, b1 = args.win
R = args.res
th0 = slice_grid(c0, u, v, a0, a1, b0, b1, R)
for eta in args.eta:
    name = f'cache/maps/{args.tag}' + (f'_eta{eta:g}' if len(args.eta) > 1 else '') + '.npz'
    if os.path.exists(name):
        print('exists', name); continue
    t0 = time.time()
    if args.problem == 'fact3':
        r = cuda_gd.run('fact', th0, eta, args.T, tol=args.tol)
    else:
        r = cuda_gd.run('mlp', th0, eta, args.T, tol=args.tol, X=PR.XOR_X, Y=PR.XOR_T, H=args.H)
    if args.problem == 'fact3':
        cl = PR.fact3_classify(r['theta'], r['status'])
        th = torch.as_tensor(r['theta'], dtype=DT, device=DEV)
        sharp = prob.sharpness(th).cpu().numpy(); del th
    else:
        cl = PR.xor_classify(prob, r['theta'], r['status'])
        sharp = np.concatenate([prob.gn_sharpness(torch.as_tensor(r["theta"][i:i + (1 << 18)], dtype=DT, device=DEV)).cpu().numpy()
                                for i in range(0, R * R, 1 << 18)])
    wall = time.time() - t0
    meta = dict(vars(args), eta=eta, wall=wall, frac=[float((r['status'] == k).mean()) for k in range(3)])
    extra = dict(theta=r['theta'].astype(np.float32)) if args.save_theta else {}
    np.savez_compressed(name, status=r['status'].reshape(R, R), tconv=r['tconv'].astype(np.float32).reshape(R, R),
                        loss=r['loss'].astype(np.float32).reshape(R, R), sharp=sharp.astype(np.float32).reshape(R, R),
                        **{k: v_.reshape(R, R) for k, v_ in cl.items()},
                        center=c0, u=u, v=v, win=np.array(args.win), meta=json.dumps(meta), **extra)
    print(f'{name}: eta={eta} frac conv/nc/div={meta["frac"]} wall={wall:.0f}s', flush=True)
