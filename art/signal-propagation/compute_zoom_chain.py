"""Nested zoom chain on the finite-width frontier, native resolution at every level.

At each level the full window is computed at --res x --res (no upsampling), binarised at
tau, and the next window (side / --step) is centred on the sub-window with the strongest
ordered/chaotic mixing (boundary density x 4p(1-p)). Every level uses the same CRN draws.

  python compute_zoom_chain.py --N 100 --D 1000 --levels 8 --step 4 --res 256 --dtype f32 --tag probe
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import frontier_grid, grid_axes, pick_zoom_center, LayerCache, meanfield_erf_L

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--D", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--levels", type=int, default=8)
ap.add_argument("--step", type=float, default=4.0)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--dtype", default="f64")
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--window", type=float, nargs=4, default=[0, 4, 0, 4])
ap.add_argument("--centers", default=None, help="json file with fixed windows to recompute (resolution check)")
ap.add_argument("--chunk", type=int, default=65536)
ap.add_argument("--tag", default="chain")
ap.add_argument("--perturb", type=float, default=0.0)
args = ap.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
dtype = torch.float64 if args.dtype == "f64" else torch.float32
tau_hit = 1e-10  # same for f32/f64 (f32 floor ~1e-13)
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "cache", f"zoom_{args.tag}_N{args.N}_D{args.D}_s{args.seed}_{args.dtype}_r{args.res}.npz")
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

layers = LayerCache(args.seed, args.N, args.D, "cuda", dtype)
if args.centers:
    windows = [tuple(w) for w in json.load(open(args.centers))]
else:
    windows = [tuple(args.window)]
res = {"L_D": [], "L_avg": [], "t_hit": [], "L_mf": []}
t0 = time.time()
lev = 0
while lev < (len(windows) if args.centers else args.levels):
    x0, x1, y0, y1 = windows[lev]
    xs, ys = grid_axes(x0, x1, y0, y1, args.res)
    SW, SB = np.meshgrid(xs, ys)
    t = time.time()
    r = frontier_grid(SW.ravel(), SB.ravel(), args.N, args.D, seed=args.seed, dtype=dtype,
                      chunk=args.chunk, layers=layers, tau_hit=tau_hit, input_perturb=args.perturb)
    L = r["L_D"].reshape(args.res, args.res)
    La = r["L_avg"].reshape(args.res, args.res)
    B = La > args.tau
    Lmf, _ = meanfield_erf_L(SW, SB, args.D)
    for k, v in (("L_D", L), ("L_avg", La), ("t_hit", r["t_hit"].reshape(args.res, args.res)), ("L_mf", Lmf)):
        res[k].append(v.astype(np.float64 if k != "t_hit" else np.int32))
    log(f"level {lev} window {windows[lev]} chaotic frac {B.mean():.3f} ({time.time()-t:.0f}s)")
    if not args.centers and lev + 1 < args.levels:
        frac = 1.0 / args.step
        cx, cy = pick_zoom_center(B, frac, margin=frac / 2 + 0.02)
        wx, wy = (x1 - x0) * frac, (y1 - y0) * frac
        mx, my = x0 + cx * (x1 - x0), y0 + cy * (y1 - y0)
        windows.append((mx - wx / 2, mx + wx / 2, my - wy / 2, my + wy / 2))
    np.savez_compressed(out, windows=np.array(windows[: lev + 1]), **{k: np.stack(v) for k, v in res.items()},
                        N=args.N, D=args.D, seed=args.seed, dtype=args.dtype, tau=args.tau, res=args.res,
                        step=args.step, wall=time.time() - t0)
    lev += 1
json.dump([list(w) for w in windows], open(out.replace(".npz", "_windows.json"), "w"))
log(f"done {out} wall {time.time()-t0:.0f}s")
