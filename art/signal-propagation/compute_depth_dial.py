"""Extra idea 2: depth as the dial. One fixed random erf network (N=100, CRN), L^l recorded at
many depths l in a single pass, so every frame is the same network read out at a different depth.

  python compute_depth_dial.py --N 100 --res 512
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import frontier_grid, grid_axes, meanfield_erf_L

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--D", type=int, default=1000)
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
layers_rec = sorted(set(np.unique(np.round(np.geomspace(1, args.D, 90)).astype(int)).tolist()))
xs, ys = grid_axes(0, 4, 0, 4, args.res)
SW, SB = np.meshgrid(xs, ys)
t = time.time()
r = frontier_grid(SW.ravel(), SB.ravel(), args.N, args.D, seed=0, dtype=torch.float32, chunk=32768, record_layers=layers_rec)
# mean-field at the same depths
mf = []
for l in layers_rec:
    mf.append(meanfield_erf_L(SW, SB, l, n_avg=1)[0])
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f"depthdial_N{args.N}_r{args.res}.npz"),
                    L=r["L_rec"].reshape(len(layers_rec), *SW.shape).astype(np.float32), layers=layers_rec,
                    L_mf=np.stack(mf).astype(np.float32), N=args.N, wall=time.time() - t)
print("done", time.time() - t)
