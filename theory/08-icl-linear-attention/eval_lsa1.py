"""Evaluate the trained one-layer LSA weights (train_lsa1.py) and the gradient-flow limits against closed forms.
CPU, float64.  Writes cache/lsa1_eval.json used by render_repro.py.

  (1) in-context risk vs test context length M     -- trained weights (Monte Carlo) vs risk_precond_gd1(B_eff, Lam, M)
  (2) covariate scaling x -> c x at test time       -- same, with Lam_test = c^2 Lam (ZFB Sec. 4.2: y_hat -> c^2 x^T w)
  (3) random-covariance training (ZFB Sec. 4.3)     -- slope of y_hat against x_q^T w on fresh random-covariance prompts
"""
import json, pathlib
import numpy as np
import torch
from icl_core import *

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)
HERE = pathlib.Path(__file__).parent
g = torch.Generator().manual_seed(7)
d, N = 20, 40
Ms = [5, 10, 20, 30, 40, 60, 80, 120, 160, 240, 320]
Cs = [0.5, 0.63, 0.79, 1.0, 1.26, 1.59, 2.0]
out = {"Ms": Ms, "Cs": Cs}


def mc_risk(Wpv, Wkq, Lam, M, n=2**18, bs=2**13):
    Lc = torch.linalg.cholesky(Lam)
    acc = 0.0
    for _ in range(n // bs):
        Z = torch.randn(bs, M + 1, d, generator=g) @ Lc.T
        w = torch.randn(bs, d, generator=g)
        X, xq = Z[:, :M], Z[:, M]
        y = torch.einsum("bmd,bd->bm", X, w)
        acc += ((lsa_predict(embed(X, y, xq), Wpv, Wkq) - (xq * w).sum(-1)) ** 2).sum().item()
    return acc / (n // bs * bs)


for cov in ["iso", "ar1"]:
    z = np.load(HERE / "cache" / f"lsa1_{cov}_d20_N40_s0.npz")
    Lam = torch.tensor(z["Lam"])
    Wpv, Wkq = torch.tensor(z["Wpv"][-1]), torch.tensor(z["Wkq"][-1])
    Beff = effective_preconditioner(Wpv, Wkq, d)
    Gi = torch.linalg.inv(zfb_gamma(Lam, N))
    ent = {"relerr_B": (torch.linalg.norm(Beff - Gi) / torch.linalg.norm(Gi)).item(),
           "offblock_norm": (torch.linalg.norm(Wkq[d]) + torch.linalg.norm(Wpv[d, :d]) + torch.linalg.norm(Wkq[:, d])).item(),
           "steps": int(z["steps"][-1])}
    # scalar-step GD-1 baseline with the best step for this covariance (closed form, optimized numerically)
    etas = torch.linspace(0.01, 1.5, 3000)
    rs = torch.stack([risk_precond_gd1(e * torch.eye(d), Lam, N) for e in etas])
    eta_best = etas[rs.argmin()].item()
    ent["eta_gd1"] = eta_best
    ent["risk_M"] = {"trained_mc": [], "theory_star": [], "theory_trained": [], "gd1_scalar": [], "ridge_mc": []}
    for M in Ms:
        ent["risk_M"]["trained_mc"].append(mc_risk(Wpv, Wkq, Lam, M))
        ent["risk_M"]["theory_star"].append(risk_precond_gd1(Gi, Lam, M).item())
        ent["risk_M"]["theory_trained"].append(risk_precond_gd1(Beff.T, Lam, M).item())
        ent["risk_M"]["gd1_scalar"].append(risk_precond_gd1(eta_best * torch.eye(d), Lam, M).item())
    ent["risk_scale"] = {"trained_mc": [], "theory_star": [], "gd1_scalar": []}
    for c in Cs:
        Lc2 = c**2 * Lam
        ent["risk_scale"]["trained_mc"].append(mc_risk(Wpv, Wkq, Lc2, N) / c**2)
        ent["risk_scale"]["theory_star"].append(risk_precond_gd1(Gi, Lc2, N).item() / c**2)
        ent["risk_scale"]["gd1_scalar"].append(risk_precond_gd1(eta_best * torch.eye(d), Lc2, N).item() / c**2)
    out[cov] = ent
    print(cov, {k: v for k, v in ent.items() if not isinstance(v, dict)}, flush=True)
    print("  risk M=40: trained MC", ent["risk_M"]["trained_mc"][Ms.index(40)], "theory", ent["risk_M"]["theory_star"][Ms.index(40)])

# ---- random covariances (ZFB 4.3): weights from exact gradient flow, which equal Thm 4.5's W* to 1e-11
z = np.load(HERE / "cache" / "gradflow_randexp_d20_N40_s0.npz")
Wpv, Wkq = torch.tensor(z["Wpv"][-1]), torch.tensor(z["Wkq"][-1])
rc = {"M": [], "slope": [], "risk": [], "scatter": None}
for M in [10, 40, 160, 640, 2560]:
    bs, num, den, rr = max(128, 4096 * 40 // M), 0.0, 0.0, 0.0
    ys, yh = [], []
    reps = max(8, 32768 // bs)
    for _ in range(reps):
        lam = -torch.log(torch.rand(bs, 1, d, generator=g))
        Z = torch.randn(bs, M + 1, d, generator=g) * lam.sqrt()
        w = torch.randn(bs, d, generator=g)
        X, xq = Z[:, :M], Z[:, M]
        y = torch.einsum("bmd,bd->bm", X, w)
        p = lsa_predict(embed(X, y, xq), Wpv, Wkq)
        t = (xq * w).sum(-1)
        num += (p * t).sum().item(); den += (t * t).sum().item(); rr += ((p - t) ** 2).sum().item()
        ys.append(t[:100]); yh.append(p[:100])
    rc["M"].append(M); rc["slope"].append(num / den); rc["risk"].append(rr / (reps * bs))
    if M == 2560:
        rc["scatter"] = [torch.cat(ys).tolist(), torch.cat(yh).tolist()]
    print("randexp M", M, "slope", num / den)
# ZFB's N, M -> infinity prediction: E[Lam^2] E[Lam^3]^-1 E[Lam] = 2/6 = 1/3 ; finite-N version via Gamma_tau
lam_s = torch.distributions.Exponential(torch.ones(d)).sample((400_000,))
Gt = (N + 1) / N * lam_s + lam_s.sum(-1, keepdim=True) / N
blk = (lam_s**2).mean(0) / (Gt * lam_s**2).mean(0)
bm = float(blk.mean())
# M -> inf: y_hat = sum_i b_i lam_i x_i w_i with b = E[Gam Lam^2]^-1 E[Lam^2] (-> E[Lam^2]/E[Lam^3] = 1/3 as N -> inf)
rc["b_finiteN"] = bm
rc["zfb_gain"] = bm                                   # ZFB's E_Lam_new[b Lam_new] = b (E lam = 1): 1/3 in the N -> inf limit
rc["slope_theory"] = 2 * bm                           # least-squares slope of y_hat on x_q^T w: b E[lam^2]/E[lam]
rc["plateau_theory"] = d * (6 * bm**2 - 4 * bm + 1)   # E (y_hat - x_q^T w)^2 as M -> inf (x^3 moments of Exp(1): 1, 2, 6)
rc["null_risk"] = d
out["randexp"] = rc
json.dump(out, open(HERE / "cache" / "lsa1_eval.json", "w"))
print("saved", rc["b_finiteN"], rc["slope_theory"], rc["plateau_theory"])
