"""Period-doubling points of the deep linear network by simulation bisection (method B).

Same predicate as compute_feigenbaum.py method B: after T steps from the declared init, is the
observable c1 = u*_1^T W_3W_2W_1 v*_1 NOT periodic with period p (tolerance tol)?  64 etas per round,
3 rounds per bifurcation.  The prediction from the exact scalar reduction (aligned + balanced
singular values, per-mode objective 1/2(rho^3 - sigma)^2) is eta_n(DLN) = eta_n(bal3) / sigma_1^(4/3).

Output: cache/feigenbaum_dln.json
"""
import json
import sys
import time
import numpy as np
import torch
from common import dln_target, dln_forward, dln_grad, DT
from compute_dln import make_init
from compute_feigenbaum import deltas

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
torch.set_num_threads(4)


def predicate(M, u1, v1, init, seed, etas, p, T, W, tol):
    N = len(etas)
    Wt = make_init(init, N, seed=seed)
    e = torch.tensor(etas, dtype=DT)[:, None, None, None]
    for _ in range(T):
        Wt = Wt - e * dln_grad(Wt, M)[0]
    H = []
    for _ in range(W + p):
        H.append((u1 @ dln_forward(Wt) @ v1).numpy())
        Wt = Wt - e * dln_grad(Wt, M)[0]
    H = np.array(H)
    dev = np.abs(H[p:] - H[:-p]).max(0)
    alive = np.isfinite(H).all(0)
    return (dev > tol) & alive


def main():
    init = sys.argv[1] if len(sys.argv) > 1 else "random"
    t0 = time.time()
    M = dln_target()
    U, S, Vh = torch.linalg.svd(M)
    u1, v1 = U[:, 0], Vh[0]
    s1 = float(S[0])
    A = json.load(open(f"{CACHE}/feigenbaum_A.json"))["A_bal3"]["eta"]
    pred = [a / s1 ** (4 / 3) for a in A]
    out = {}
    for T, tol in [(20000, 1e-5), (200000, 1e-8)]:
        found, hw = [], []
        for n in range(1, 8):
            p = 2 ** (n - 1)
            lo = pred[n - 1] - 0.3 * (pred[n - 1] - (pred[n - 2] if n > 1 else 0.8 * pred[0]))
            hi = pred[n - 1] + 0.3 * (pred[n] - pred[n - 1])
            ok = True
            for r in range(3):
                grid = np.linspace(lo, hi, 64)
                pr = predicate(M, u1, v1, init, 0, grid, p, T, 4 * p + 64, tol)
                i = int(np.argmax(pr))
                if not pr.any() or i == 0:
                    ok = False
                    break
                lo, hi = grid[i - 1], grid[i]
            found.append(0.5 * (lo + hi) if ok else float("nan"))
            hw.append(0.5 * (hi - lo) if ok else float("nan"))
            print(f"T={T} n={n}: eta={found[-1]:.10g} pred={pred[n - 1]:.10g} rel.diff={(found[-1] - pred[n - 1]) / pred[n - 1]:.2e}  {time.time() - t0:.0f}s", flush=True)
        d, ds = deltas(found, hw)
        out[f"T{T}"] = dict(eta=found, halfwidth=hw, pred=pred[:7], delta=d, delta_sigma=ds, tol=tol)
        print("delta", np.round(d, 4), flush=True)
    json.dump(out, open(f"{CACHE}/feigenbaum_dln_{init}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
