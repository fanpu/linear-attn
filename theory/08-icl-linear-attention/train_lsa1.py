"""Reproduce Zhang, Frei & Bartlett (2024): one-layer LSA trained from their balanced init converges to W*.

Trains the FULL (d+1)x(d+1) matrices W_PV and W_KQ (every entry free) by large-batch SGD on fresh prompts
with Adam + cosine decay, float64, d=20, N=40.  (Plain SGD at learning rates that finish in reasonable time blows up:
heavy-tailed minibatch gradients kick the off-manifold blocks, whose curvature ~ E|x|^4 is large.  The exact
population gradient flow lives in lsa_gradflow.py.)
Saves snapshots for the hero animation and the final weights.

  .venv/bin/python 08-icl-linear-attention/train_lsa1.py --cov ar1 --steps 6000
"""
import argparse, math, time, pathlib
import numpy as np
import torch
from icl_core import *

p = argparse.ArgumentParser()
p.add_argument("--cov", default="ar1")          # iso | ar1 | randexp (ZFB Sec 4.3: diag Exp(1) per prompt)
p.add_argument("--d", type=int, default=20)
p.add_argument("--N", type=int, default=40)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch", type=int, default=16384)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--opt", default="adam")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--nsnap", type=int, default=240)
a = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.08)
dev = "cuda"
D = torch.float64
g = torch.Generator(device=dev).manual_seed(a.seed)
torch.manual_seed(a.seed)
d, N = a.d, a.N

Lam = make_cov("iso" if a.cov == "randexp" else a.cov, d)
Gam = zfb_gamma(Lam, N)
if a.cov == "randexp":
    # ZFB Thm 4.5 limit: W*_KQ block = E[Gam_tau Lam_tau^2]^{-1} E[Lam_tau^2] (up to balancing), diag Exp(1)
    lam_s = torch.distributions.Exponential(torch.ones(4_000_000, d, dtype=D)).sample()
    Gt = (N + 1) / N * lam_s + lam_s.sum(-1, keepdim=True) / N
    blk = torch.diag((lam_s**2).mean(0) / (Gt * lam_s**2).mean(0))
    c = torch.linalg.norm(blk) ** 0.5
    Wpv_star = torch.zeros(d + 1, d + 1, dtype=D); Wpv_star[d, d] = c
    Wkq_star = torch.zeros(d + 1, d + 1, dtype=D); Wkq_star[:d, :d] = blk / c
else:
    Wpv_star, Wkq_star = zfb_global_min(Lam, N)

# ZFB Assumption 3.3 init: W_PV = sigma e_{d+1} e_{d+1}^T, W_KQ = sigma [[Theta Theta^T, 0],[0, 0]], ||Theta Theta^T||_F = 1
Theta = torch.randn(d, d, dtype=D, generator=torch.Generator().manual_seed(a.seed))
TT = Theta @ Theta.T
TT = TT / torch.linalg.norm(TT)
sig = 0.5 * math.sqrt(2 / (torch.linalg.matrix_norm(Gam, 2).item() * math.sqrt(d)))
Wpv = torch.zeros(d + 1, d + 1, dtype=D); Wpv[d, d] = sig
Wkq = torch.zeros(d + 1, d + 1, dtype=D); Wkq[:d, :d] = sig * TT
Wpv, Wkq = Wpv.to(dev).requires_grad_(), Wkq.to(dev).requires_grad_()
Lam_d = Lam.to(dev)
L_chol = torch.linalg.cholesky(Lam_d)


def batch(B, M):
    Z = torch.randn(B, M + 1, d, device=dev, dtype=D, generator=g)
    if a.cov == "randexp":
        lam = -torch.log(torch.rand(B, 1, d, device=dev, dtype=D, generator=g))
        Z = Z * lam.sqrt()
    else:
        Z = Z @ L_chol.T
    w = torch.randn(B, d, device=dev, dtype=D, generator=g)
    X, xq = Z[:, :M], Z[:, M]
    y = torch.einsum("bmd,bd->bm", X, w)
    return X, y, xq, torch.einsum("bd,bd->b", xq, w)


ev = [batch(a.batch, N) for _ in range(4)]


def eval_loss():
    with torch.no_grad():
        return float(np.mean([0.5 * ((lsa_predict(embed(X, y, xq), Wpv, Wkq) - yq) ** 2).mean().item() for X, y, xq, yq in ev]))


# population risk (x2 loss) at W*: closed form for fixed covariance
risk_star = 0.5 * risk_precond_gd1(torch.linalg.inv(Gam), Lam, N).item() if a.cov != "randexp" else float("nan")

snap_steps = sorted(set([0] + list(np.unique(np.round(np.geomspace(1, a.steps, a.nsnap)).astype(int)))))
snaps = dict(step=[], Wpv=[], Wkq=[], loss=[])
opt = torch.optim.Adam([Wpv, Wkq], lr=a.lr) if a.opt == "adam" else torch.optim.SGD([Wpv, Wkq], lr=a.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / a.steps)))
t0 = time.time()
for step in range(a.steps + 1):
    if step in snap_steps:
        snaps["step"].append(step); snaps["loss"].append(eval_loss())
        snaps["Wpv"].append(Wpv.detach().cpu().numpy().copy()); snaps["Wkq"].append(Wkq.detach().cpu().numpy().copy())
    if step % 500 == 0:
        Beff = effective_preconditioner(Wpv.detach().cpu(), Wkq.detach().cpu(), d)
        err = (torch.linalg.norm(Beff - Wpv_star[d, d] * Wkq_star[:d, :d]) / torch.linalg.norm(Wpv_star[d, d] * Wkq_star[:d, :d])).item()
        print(f"step {step:6d}  loss {snaps['loss'][-1] if snaps['step'][-1]==step else float('nan'):.5f}  "
              f"L(W*) {risk_star:.5f}  relerr(B_eff) {err:.4f}  {time.time()-t0:.0f}s", flush=True)
    if step == a.steps:
        break
    X, y, xq, yq = batch(a.batch, N)
    loss = 0.5 * ((lsa_predict(embed(X, y, xq), Wpv, Wkq) - yq) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

out = pathlib.Path(__file__).parent / "cache" / f"lsa1_{a.cov}_d{d}_N{N}_s{a.seed}.npz"
np.savez(out, steps=np.array(snaps["step"]), Wpv=np.array(snaps["Wpv"]), Wkq=np.array(snaps["Wkq"]),
         loss=np.array(snaps["loss"]), Wpv_star=Wpv_star.numpy(), Wkq_star=Wkq_star.numpy(), Lam=Lam.numpy(),
         Gam=Gam.numpy(), risk_star=risk_star, sigma0=sig, lr=a.lr, batch=a.batch)
print("saved", out)
