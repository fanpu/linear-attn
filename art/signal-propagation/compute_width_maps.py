"""Full-plane finite-width frontier maps (paper setup: erf, D=1000, (sigma_w, sigma_b) in [0,4]^2)
for a list of widths and seeds, with width-nested CRN draws (sp_core.layer_draw).

  python compute_width_maps.py --res 512 --Ns 8 11 16 ... --seeds 0 --dtype f32 --tag widths
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import frontier_grid, grid_axes, LayerCache, meanfield_erf_L

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--Ns", type=int, nargs="+", default=[20, 100, 1000])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--D", type=int, default=1000)
ap.add_argument("--dtype", default="f32")
ap.add_argument("--window", type=float, nargs=4, default=[0, 4, 0, 4])
ap.add_argument("--chunk", type=int, default=32768)
ap.add_argument("--tag", default="widths")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
dtype = torch.float64 if args.dtype == "f64" else torch.float32
here = os.path.dirname(os.path.abspath(__file__))
xs, ys = grid_axes(*args.window, args.res)
SW, SB = np.meshgrid(xs, ys)
t0 = time.time()
Lmf, Lmf_avg = meanfield_erf_L(SW, SB, args.D)
np.savez_compressed(os.path.join(here, "cache", f"widthmap_mf_D{args.D}_r{args.res}.npz"), L_D=Lmf, L_avg=Lmf_avg,
                    window=args.window)
for seed in args.seeds:
    for N in args.Ns:
        out = os.path.join(here, "cache", f"widthmap_{args.tag}_N{N}_s{seed}_D{args.D}_{args.dtype}_r{args.res}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        chunk = args.chunk if N <= 300 else max(2048, args.chunk // 8)
        r = frontier_grid(SW.ravel(), SB.ravel(), N, args.D, seed=seed, dtype=dtype, chunk=chunk, tau_hit=1e-10)
        np.savez_compressed(out, L_D=r["L_D"].reshape(SW.shape), L_avg=r["L_avg"].reshape(SW.shape),
                            t_hit=r["t_hit"].reshape(SW.shape).astype(np.int16), N=N, seed=seed, D=args.D,
                            window=args.window, dtype=args.dtype, wall=time.time() - t)
        print(time.strftime("%H:%M:%S"), f"N={N} seed={seed} chaotic={np.mean(r['L_avg'] > 1e-5):.3f} {time.time() - t:.0f}s", flush=True)
print("done", time.time() - t0)
