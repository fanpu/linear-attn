"""Extra idea 1: the finite-time Lyapunov exponent over the (sigma_w, sigma_b) plane (and a zoom),
the continuous signed quantity underneath the binary frontier. Same CRN draws as the frontier maps.

  python compute_lyapunov.py --N 100 --res 1024 --window 0 4 0 4 --tag full
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import lyapunov_grid, grid_axes, LayerCache

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--D", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--window", type=float, nargs=4, default=[0, 4, 0, 4])
ap.add_argument("--dtype", default="f32")
ap.add_argument("--tag", default="full")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
dt = torch.float64 if args.dtype == "f64" else torch.float32
xs, ys = grid_axes(*args.window, args.res)
SW, SB = np.meshgrid(xs, ys)
t = time.time()
rec_layers = (250, 300, 400, 600, 800)
lam, rec = lyapunov_grid(SW.ravel(), SB.ravel(), args.N, args.D, seed=args.seed, dtype=dt, chunk=32768, record_layers=rec_layers)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                   f"lyap_{args.tag}_N{args.N}_s{args.seed}_D{args.D}_{args.dtype}_r{args.res}.npz")
np.savez_compressed(out, lam=lam.reshape(SW.shape), rec=rec.reshape(len(rec_layers), *SW.shape), rec_layers=rec_layers,
                    window=args.window, N=args.N, D=args.D, wall=time.time() - t)
print("done", out, time.time() - t, "frac lam>0", (lam > 0).mean())
