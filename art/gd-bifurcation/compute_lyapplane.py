"""Two-step-size Lyapunov plane ("Markus-Lyapunov" construction) for GD with a cyclic learning-rate schedule.

GD on 1/2(x1x2x3x4 - 1)^2 whose step size cycles through a pattern of two values, e.g. "AB" means
eta_t = A, B, A, B, ...  For each pixel (A, B) we compute the Lyapunov exponent of the oscillating mode,
using the exact balanced-line sub-dynamics u <- u - eta_t (u^4 - 1) u^3 (float64).  The diagonal A = B is
exactly the Lyapunov strip of the constant-step-size cascade.  A random subsample of pixels is re-run
with the full 4-coordinate GD (max Lyapunov exponent by tangent propagation) as a check.

Usage: python compute_lyapplane.py --pattern AB --res 4096 --lo 0.35 --hi 1.05
Output: cache/lyapplane_<pattern>_<res>.npz
"""
import argparse
import time
from multiprocessing import Pool
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
K = 4


def rows_job(args):
    a_vals, b, pattern, burn, rec = args
    A = a_vals[None, :]
    B = b[:, None]
    seq = [A if ch == "A" else B for ch in pattern]
    L = len(seq)
    u = np.full((len(b), len(a_vals)), 1.0001)
    lam = np.zeros_like(u)
    with np.errstate(all="ignore"):
        for t in range(burn + rec):
            e = seq[t % L]
            if t >= burn:
                lam += np.log(np.abs(nm.bal_deriv(u, e, K)) + 1e-300)
            u = nm.bal_step(u, e, K)
            u = np.where(np.abs(u) > 1e6, np.nan, u)
    lam /= rec
    return lam.astype(np.float32)


def _gpu_step(u, lam, e, acc: bool):
    import torch
    if acc:
        lam = lam + torch.log(torch.abs(1.0 - e * (7 * u ** 6 - 3 * u ** 2)) + 1e-300)
    u = u - e * (u ** 4 - 1.0) * u ** 3
    return u, lam


_compiled = None


def gpu_plane(vals, valsb, pattern, burn, rec, rows_per_chunk=1024):
    """Same computation as rows_job on the GPU; the step is fused with torch.compile."""
    import torch
    global _compiled
    torch.cuda.set_per_process_memory_fraction(0.10)
    if _compiled is None:
        _compiled = torch.compile(_gpu_step)
    dev = "cuda"
    out = []
    for i in range(0, len(valsb), rows_per_chunk):
        B = torch.tensor(valsb[i:i + rows_per_chunk], dtype=torch.float64, device=dev)[:, None].expand(-1, len(vals)).contiguous()
        A = torch.tensor(vals, dtype=torch.float64, device=dev)[None, :].expand(B.shape[0], -1).contiguous()
        u = torch.full_like(A, 1.0001)
        lam = torch.zeros_like(u)
        for t in range(burn + rec):
            e = A if pattern[t % len(pattern)] == "A" else B
            u, lam = _compiled(u, lam, e, t >= burn)
            if t % 256 == 0:
                u = torch.where(torch.abs(u) > 1e6, torch.full_like(u, float("nan")), u)
        out.append((lam / rec).float().cpu().numpy())
    return np.vstack(out)


def full_check(A, B, pattern, burn, rec, seed=0):
    N = len(A)
    x = np.broadcast_to(np.array([1.1, 0.9, 1.05, 0.95]), (N, 4)).copy()
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lam = np.zeros(N)
    L = len(pattern)
    with np.errstate(all="ignore"):
        for t in range(burn + rec):
            e = A if pattern[t % L] == "A" else B
            if t >= burn - 200:
                v = nm.prod_tangent(x, v, e)
                nv = np.linalg.norm(v, axis=1)
                if t >= burn:
                    lam += np.log(nv)
                v /= nv[:, None]
            x = nm.prod_step(x, e)
    return lam / rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="AB")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=1.05)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--rec", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--blo", type=float, default=None)
    ap.add_argument("--bhi", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    vals = a.lo + (a.hi - a.lo) * (np.arange(a.res) + 0.5) / a.res
    blo = a.lo if a.blo is None else a.blo
    bhi = a.hi if a.bhi is None else a.bhi
    valsb = blo + (bhi - blo) * (np.arange(a.res) + 0.5) / a.res
    if a.gpu:
        lam = gpu_plane(vals, valsb, a.pattern, a.burn, a.rec)
    else:
        blocks = np.array_split(valsb, max(1, a.res // 64))
        with Pool(a.workers) as pool:
            parts = pool.map(rows_job, [(vals, b, a.pattern, a.burn, a.rec) for b in blocks])
        lam = np.vstack(parts)  # rows = B, cols = A
    rng = np.random.default_rng(0)
    idx = rng.integers(0, a.res, size=(4000, 2))
    Af, Bf = vals[idx[:, 1]], valsb[idx[:, 0]]
    lf = full_check(Af, Bf, a.pattern, a.burn, a.rec)
    lb = lam[idx[:, 0], idx[:, 1]]
    ok = np.isfinite(lf) & np.isfinite(lb)
    chaos = ok & (lb > 0.02)
    agree = float(np.mean(np.abs(lf[chaos] - lb[chaos]) < 0.05)) if chaos.any() else float("nan")
    print(f"full-GD check: chaotic pixels {chaos.sum()}, |lambda_full - lambda_bal| < 0.05 for {agree:.3f}; "
          f"sign agreement (lambda>0.02) {np.mean((lf[ok] > 0.02) == (lb[ok] > 0.02)):.3f}")
    np.savez_compressed(f"{CACHE}/lyapplane_{a.pattern}{a.tag}_{a.res}.npz", lam=lam, vals=vals, vals_a=vals, vals_b=valsb, pattern=a.pattern,
                        check_idx=idx, check_full=lf, check_bal=lb, burn=a.burn, rec=a.rec)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
