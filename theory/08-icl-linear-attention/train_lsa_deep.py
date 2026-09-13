"""Multi-layer linear self-attention vs multi-step (preconditioned) gradient descent.

Architecture = Ahn, Cheng, Daneshmand & Sra (2023), eq. (4):
    Z_{l+1} = Z_l + (1/n) P_l Z_l M Z_l^T Q_l Z_l,     prediction = -[Z_L]_{d+1, n+1}
with M masking the query column out of keys/values.  Three parameterizations trained on fresh prompts:
    dense : P_l, Q_l free (d+1)x(d+1)                                   -- "the transformer"
    pgd   : P_l = [[0,0],[0,1]], Q_l = -[[A_l,0],[0,0]]  (Ahn eq. 8)     -- L steps of GD preconditioned by A_l
    gd    : as pgd with A_l = eta_l I                                    -- L steps of plain GD, tuned step sizes
Reported risk is the EXCESS risk E(y_hat - <w, x_q>)^2 on 2^18 fresh prompts (float64).

  .venv/bin/python 08-icl-linear-attention/train_lsa_deep.py --covs iso --kinds gd,dense --Ls 1,2,3,4 --out iso
"""
import argparse, json, math, pathlib, time, itertools
import numpy as np
import torch
from icl_core import make_cov, ridge_w, gd_w

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cpu")
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--kinds", default="gd,pgd,dense")
p.add_argument("--Ls", default="1,2,3,4")
p.add_argument("--covs", default="iso,ar1")
p.add_argument("--ns", default="20")
p.add_argument("--d", type=int, default=10)
p.add_argument("--sigmas", default="0.0")
p.add_argument("--out", default="main")
a = p.parse_args()
dev, DT = a.device, torch.float64
TDT = torch.float32           # training precision (evaluation is float64)
torch.set_num_threads(4)
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed)


def prompts(B, n, d, Lc, sigma, dt=DT):
    Lc = Lc.to(dt)
    Z = torch.randn(B, n + 1, d, device=dev, dtype=dt, generator=gen) @ Lc.T
    w = torch.randn(B, d, device=dev, dtype=dt, generator=gen)
    X, xq = Z[:, :n], Z[:, n]
    y = torch.einsum("bnd,bd->bn", X, w) + sigma * torch.randn(B, n, device=dev, dtype=dt, generator=gen)
    return X, y, xq, (xq * w).sum(-1)


def lsa_forward(params, kind, X, y, xq):
    B, n, d = X.shape
    DT_ = X.dtype
    params = {k: [t.to(DT_) for t in v] for k, v in params.items()}
    if kind in ("gd", "pgd"):
        # Ahn et al. Lemma 1: the sparse parameterization (8) is exactly L steps of GD preconditioned by A_l.
        # Computed directly in weight space (d x d ops), which is much cheaper than the token form.
        I = torch.eye(d, device=X.device, dtype=DT_)
        lrs = [A if kind == "pgd" else A * I for A in params["Q"]]
        return (gd_w(X, y, lrs) * xq).sum(-1)
    Z = torch.cat([torch.cat([X, y[..., None]], -1), torch.cat([xq, xq.new_zeros(B, 1)], -1)[:, None]], 1).transpose(1, 2)
    for l in range(len(params["Q"])):
        P, Q = params["P"][l], params["Q"][l]
        Zc = Z[..., :n]
        Z = Z + P @ (Zc @ Zc.transpose(1, 2)) @ Q @ Z / n
    return -Z[:, d, n]


def train(kind, L, d, n, sigma, cov, steps, B):
    Lam = make_cov(cov, d).to(dev)
    Lc = torch.linalg.cholesky(Lam)
    if kind == "dense":
        params = {"P": [(0.01 * torch.randn(d + 1, d + 1, device=dev, dtype=TDT)).requires_grad_() for _ in range(L)],
                  "Q": [(0.01 * torch.randn(d + 1, d + 1, device=dev, dtype=TDT)).requires_grad_() for _ in range(L)]}
    elif kind == "pgd":
        params = {"Q": [(0.3 * torch.eye(d, device=dev, dtype=TDT) + 0.01 * torch.randn(d, d, device=dev, dtype=TDT)).requires_grad_() for _ in range(L)]}
    else:
        params = {"Q": [torch.tensor(0.3, device=dev, dtype=TDT).requires_grad_() for _ in range(L)]}
    allp = [t for v in params.values() for t in v]
    opt = torch.optim.Adam(allp, lr=a.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.1)
    curve = []
    for s in range(steps):
        X, y, xq, yq = prompts(B, n, d, Lc, sigma, TDT)
        loss = ((lsa_forward(params, kind, X, y, xq) - yq) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(allp, 1.0)
        opt.step(); sch.step()
        if s % 50 == 0:
            curve.append(loss.item())
    with torch.no_grad():
        r = []
        for _ in range(4):
            X, y, xq, yq = prompts(16384, n, d, Lc, sigma)
            r.append(((lsa_forward(params, kind, X, y, xq) - yq) ** 2).mean().item())
    return float(np.mean(r)), curve, {k: [t.detach().cpu().numpy().tolist() for t in v] for k, v in params.items()}


@torch.no_grad()
def ridge_risk(d, n, sigma, cov):
    Lam = make_cov(cov, d).to(dev); Lc = torch.linalg.cholesky(Lam)
    r = []
    for _ in range(4):
        X, y, xq, yq = prompts(16384, n, d, Lc, sigma)
        lam = max(sigma**2, 1e-10)                      # Bayes-optimal ridge for w ~ N(0, I)
        r.append(((ridge_w(X, y, lam) * xq).sum(-1) - yq).pow(2).mean().item())
    return float(np.mean(r))


configs = [(k, L, a.d, n, s, cov) for cov in a.covs.split(",") for s in map(float, a.sigmas.split(",")) for n in map(int, a.ns.split(","))
           for L in map(int, a.Ls.split(",")) for k in a.kinds.split(",")]
out = pathlib.Path(__file__).parent / "cache" / f"lsa_deep_{a.out}.json"
res = json.load(open(out)) if out.exists() else {"runs": [], "ridge": {}}
done = {tuple(r["cfg"]) for r in res["runs"]}
for cfg in configs:
    key = f"{cfg[2]}_{cfg[3]}_{cfg[4]}_{cfg[5]}"
    if key not in res["ridge"]:
        res["ridge"][key] = ridge_risk(*cfg[2:])
    if tuple(cfg) in done:
        continue
    t0 = time.time()
    risk, curve, params = train(*cfg, steps=a.steps, B=a.batch)
    res["runs"].append({"cfg": list(cfg), "risk": risk, "curve": curve, "params": params if cfg[0] != "dense" or cfg[1] <= 4 else None})
    print(f"{cfg}  risk {risk:.5f}  ridge {res['ridge'][key]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out, "w"))
print("saved", out)
