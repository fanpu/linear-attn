"""Exact gradient flow on the POPULATION loss of a one-layer LSA (ZFB 2024, Sec. 3.3), no sampling.

From ZFB's init (Assumption 3.3) only w22^PV =: u and W11^KQ =: V move (the other blocks have zero gradient),
and the prediction is  y_hat = x_q^T A (1/N) sum_i y_i x_i  with  A = u V^T.  The population loss is then the
closed form 1/2 * risk_precond_gd1(A, Lam, N) (tested against Monte Carlo and ZFB Thm 4.2 in test_core.py),
so we can integrate  d(u,V)/dt = -grad L  with an adaptive ODE solver in float64.

  .venv/bin/python 08-icl-linear-attention/lsa_gradflow.py --cov ar1
"""
import argparse, math, pathlib
import numpy as np
import torch
from scipy.integrate import solve_ivp
from icl_core import make_cov, zfb_gamma, zfb_global_min, risk_precond_gd1

p = argparse.ArgumentParser()
p.add_argument("--cov", default="ar1")
p.add_argument("--d", type=int, default=20)
p.add_argument("--N", type=int, default=40)
p.add_argument("--T", type=float, default=3000.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--nsnap", type=int, default=400)
a = p.parse_args()
torch.set_default_dtype(torch.float64)
d, N = a.d, a.N
if a.cov == "randexp":
    # ZFB Sec. 4.3: every prompt has its own diagonal covariance with i.i.d. Exponential(1) entries.
    # Population loss = average of the fixed-covariance closed form over a large fixed sample of covariances.
    lam_s = torch.distributions.Exponential(torch.ones(d)).sample((200_000,))
    Lam = torch.eye(d)
    Gam = zfb_gamma(2 * Lam, N)  # only used for the init-scale bound
    Gt = (N + 1) / N * lam_s + lam_s.sum(-1, keepdim=True) / N
    blk = torch.diag((lam_s**2).mean(0) / (Gt * lam_s**2).mean(0))          # E[Gam Lam^2]^-1 E[Lam^2]  (ZFB 4.12)
    cst = torch.linalg.norm(blk) ** 0.5
    Wpv_star = torch.zeros(d + 1, d + 1); Wpv_star[d, d] = cst
    Wkq_star = torch.zeros(d + 1, d + 1); Wkq_star[:d, :d] = blk / cst
else:
    Lam = make_cov(a.cov, d)
    Gam = zfb_gamma(Lam, N)
    Wpv_star, Wkq_star = zfb_global_min(Lam, N)


def risk_diag_batch(A, lam, M):
    """risk_precond_gd1 for diagonal covariances lam [K,d], averaged over K (w ~ N(0,I), x_q ~ N(0,Lam))."""
    A2 = A**2                                                     # A2[k,i] = A_ki^2
    Bii = lam @ A2                                                # [K,d]  diag of B = A^T Lam A
    trL = lam.sum(-1)
    r = trL - 2 * (lam**2 * torch.diagonal(A)[None]).sum(-1) + (1 + 1 / M) * (lam**2 * Bii).sum(-1) + (lam * Bii).sum(-1) * trL / M
    return r.mean()

Theta = torch.randn(d, d, generator=torch.Generator().manual_seed(a.seed))
TT = Theta @ Theta.T
TT = TT / torch.linalg.norm(TT)
sig = 0.5 * math.sqrt(2 / (torch.linalg.matrix_norm(Gam, 2).item() * math.sqrt(d)))  # sigma^2 ||Gam|| sqrt(d) < 2


def loss(theta):
    u, V = theta[0], theta[1:].view(d, d)
    if a.cov == "randexp":
        return 0.5 * risk_diag_batch(u * V.T, lam_s, N)
    return 0.5 * risk_precond_gd1(u * V.T, Lam, N)


def rhs(t, th):
    th = torch.tensor(th, requires_grad=True)
    (g,) = torch.autograd.grad(loss(th), th)
    return (-g).numpy()


th0 = torch.cat([torch.tensor([sig]), (sig * TT).flatten()]).numpy()
ts = np.concatenate([[0], np.geomspace(1e-2, a.T, a.nsnap - 1)])
sol = solve_ivp(rhs, (0, a.T), th0, t_eval=ts, method="LSODA", rtol=1e-9, atol=1e-12)
U = sol.y[0]
V = sol.y[1:].T.reshape(-1, d, d)
L = np.array([loss(torch.tensor(sol.y[:, i])).item() for i in range(len(ts))])
Lstar = loss(torch.cat([Wpv_star[d, d:d+1], Wkq_star[:d, :d].flatten()])).item()
Wkq = np.zeros((len(ts), d + 1, d + 1)); Wkq[:, :d, :d] = V
Wpv = np.zeros((len(ts), d + 1, d + 1)); Wpv[:, d, d] = U
err_kq = np.linalg.norm(Wkq - Wkq_star.numpy(), axis=(1, 2)) / np.linalg.norm(Wkq_star.numpy())
print(f"final loss {L[-1]:.8f}  closed-form L* {Lstar:.8f}  |W_KQ - W*|/|W*| = {err_kq[-1]:.2e}  "
      f"w22 = {U[-1]:.6f} vs {Wpv_star[d, d].item():.6f}")
out = pathlib.Path(__file__).parent / "cache" / f"gradflow_{a.cov}_d{d}_N{N}_s{a.seed}.npz"
np.savez(out, t=ts, Wpv=Wpv, Wkq=Wkq, loss=L, loss_star=Lstar, Wpv_star=Wpv_star.numpy(), Wkq_star=Wkq_star.numpy(),
         Lam=Lam.numpy(), Gam=Gam.numpy(), err_kq=err_kq, sigma0=sig)
print("saved", out)
