"""Deep linear network (matrix factorisation) bifurcation sweep, torch float64.

Setting (Ghosh et al. 2025, App. B.1, with the 1/2 in the loss):
  f(W) = 1/2 || W_3 W_2 W_1 - M ||_F^2,  W_l in R^{5x5},  M rank 3 with singular values 10, 6, 3.
Recorded per iterate: c_i = u*_i^T (W_3 W_2 W_1) v*_i for i=1,2 (the end-to-end matrix projected on
the target's singular pairs; equals the oscillating singular value once aligned) and log10 loss.
Max Lyapunov exponent via forward-mode JVP tangent propagation.

Usage: python compute_dln.py --init aligned|random [--n 16000 --burn 20000 --rec 2048]
"""
import argparse
import time
import numpy as np
import torch
from common import dln_target, dln_forward, dln_grad, DT

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


def make_init(kind, N, d=5, L=3, seed=0, dev="cpu"):
    if kind == "aligned":  # Ghosh et al. eq. (4): W_L = 0, W_l = alpha I
        W = torch.zeros(L, d, d, dtype=DT)
        for l in range(L - 1):
            W[l] = 0.1 * torch.eye(d, dtype=DT)
    else:
        g = torch.Generator().manual_seed(seed)
        W = 0.5 * torch.randn(L, d, d, generator=g, dtype=DT) / d ** 0.5
    return W.to(dev).expand(N, L, d, d).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="aligned")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--burn", type=int, default=20000)
    ap.add_argument("--rec", type=int, default=2048)
    ap.add_argument("--lo", type=float, default=0.028)
    ap.add_argument("--hi", type=float, default=0.0675)
    ap.add_argument("--lyap_tail", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    t0 = time.time()
    M = dln_target().to(dev)
    U, S, Vh = torch.linalg.svd(M)
    u1, v1, u2, v2 = U[:, 0], Vh[0], U[:, 1], Vh[1]
    etas = torch.linspace(a.lo, a.hi, a.n, dtype=DT, device=dev)
    W = make_init(a.init, a.n, seed=a.seed, dev=dev)
    V = torch.randn(W.shape, dtype=DT, device=dev, generator=torch.Generator(dev).manual_seed(1))
    V /= V.flatten(1).norm(dim=1)[:, None, None, None]
    e4 = etas[:, None, None, None]
    C1 = torch.empty(a.n, a.rec, dtype=torch.float32)
    C2 = torch.empty(a.n, a.rec, dtype=torch.float32)
    LL = torch.empty(a.n, a.rec, dtype=torch.float32)
    lyap = torch.zeros(a.n, dtype=DT, device=dev)
    alive = torch.ones(a.n, dtype=torch.bool, device=dev)
    gfun = lambda Wx: dln_grad(Wx, M)[0]
    for t in range(a.burn + a.rec):
        rec = t >= a.burn
        if t >= a.burn - a.lyap_tail:
            G, jv = torch.func.jvp(gfun, (W,), (V,))
            V = V - e4 * jv
            nv = V.flatten(1).norm(dim=1)
            if rec:
                lyap += torch.log(nv)
            V = V / nv[:, None, None, None]
        else:
            G = gfun(W)
        W = W - e4 * G
        bad = ~torch.isfinite(W).flatten(1).all(1) | (W.flatten(1).abs().amax(1) > 1e4)
        alive &= ~bad
        W[bad] = 0.1
        if rec:
            E = dln_forward(W)
            i = t - a.burn
            C1[:, i] = (u1 @ E @ v1).float().cpu()
            C2[:, i] = (u2 @ E @ v2).float().cpu()
            LL[:, i] = torch.log10(0.5 * ((E - M) ** 2).flatten(1).sum(1) + 1e-300).float().cpu()
        if t % 5000 == 0:
            print(f"t={t} alive={alive.float().mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
    am = alive.cpu().numpy()
    C1, C2, LL = C1.numpy(), C2.numpy(), LL.numpy()
    C1[~am] = np.nan
    C2[~am] = np.nan
    LL[~am] = np.nan
    # balance diagnostics at the end: spread of per-layer top singular values
    sv = torch.linalg.svdvals(W).cpu().numpy()  # (N,L,d)
    np.savez(f"{CACHE}/bif_dln_{a.init}{a.tag}.npz", etas=etas.cpu().numpy(), C1=C1, C2=C2, logloss=LL,
             lyap=(lyap / a.rec).cpu().numpy(), alive=am, layer_sv=sv, svals=S.cpu().numpy(),
             burn=a.burn, rec=a.rec, init=a.init, seed=a.seed)
    print(f"done in {time.time() - t0:.0f}s alive {am.mean():.3f}")


if __name__ == "__main__":
    main()
