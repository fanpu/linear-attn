"""Mean-field (infinite-width) analogue of t_hit on the width-map grid: first depth (checked every 5 layers,
as in frontier_grid) at which the closed-form L^l drops below 1e-10. CPU, closed form. The mean-field L
below ~1e-15 is cancellation noise (2 (E phi^2 - E phi phi')), so t_hit is the clean ordered-side shade.

  python compute_mf_thit.py --res 512 --D 1000
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sp_core import grid_axes, erf_cov

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--D", type=int, default=1000)
ap.add_argument("--window", type=float, nargs=4, default=[0, 4, 0, 4])
args = ap.parse_args()
xs, ys = grid_axes(*args.window, args.res)
SW, SB = np.meshgrid(xs, ys)
sw2, sb2 = SW ** 2, SB ** 2
q, q12 = sw2 + sb2, sb2.copy()
t_hit = np.full(SW.shape, args.D + 1, dtype=np.int32)
for l in range(1, args.D + 1):
    E2, E12 = erf_cov(q, q, q), erf_cov(q, q, q12)
    if l % 5 == 0:
        L = 2 * (E2 - E12)
        t_hit = np.where((t_hit > args.D) & (L < 1e-10), l, t_hit)
    q, q12 = sw2 * E2 + sb2, sw2 * E12 + sb2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f"widthmap_mf_thit_D{args.D}_r{args.res}.npz")
np.savez_compressed(out, t_hit=t_hit)
print(out, (t_hit <= args.D).mean())
