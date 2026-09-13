"""Build-on: does Saxe's 'plateau ~ 1/strength' law survive in a softmax attention head?

Task ("copy-and-transform"): a sequence of T tokens e_t = [x_t ; onehot(t)], x_t ~ N(0, I_d).
Target read out at the last position: y = M x_1, where M = sum_k s_k u_k v_k^T (rank r).
Model: ONE softmax attention head, no MLP/LayerNorm/residual:
    a_t = softmax_t( (W_Q e_T) . (W_K e_t) / sqrt(d_k) ),   y_hat = W_O W_V sum_t a_t e_t
Every weight starts at N(0, sigma^2); trained with full-batch gradient descent (small lr, time t = lr x steps) on a fixed
dataset of n_data sequences whose token contents are exactly whitened (empirical E[x x^T] = I jointly over all positions),
so every second moment the linear parts see equals its population value.
Many models (different targets / seeds / controls) are trained in parallel as a leading batch axis.

Controls:  attn="learn" (normal), "uniform" (W_Q=W_K=0 frozen -> a_t = 1/T), "oracle" (a_1 = 1 fixed -> pure
2-layer linear net on x_1, where Saxe's closed form must hold).

  .venv/bin/python 03-saxe-dynamics/compute_attention.py <experiment> [--device cuda]
experiments: showcase | sweep_rank1 | sweep_mode3
"""
import argparse
import math
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"


def random_orthogonal(n, gen):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def train(svals, attn_modes, seeds, *, d=8, T=8, dk=16, dh=16, sigma=0.01, lr=0.02, t_end=200.0, n_data=2048,
          n_rec=400, device="cpu", dtype=torch.float32, target_seed=0):
    """svals: list of per-model singular-value lists (padded to d); attn_modes: per-model str; seeds: per-model int."""
    M = len(svals)
    D = d + T
    gen = torch.Generator().manual_seed(target_seed)
    U = random_orthogonal(d, gen); V = random_orthogonal(d, gen)
    S = torch.zeros(M, d, dtype=torch.float64)
    for m, sv in enumerate(svals):
        S[m, : len(sv)] = torch.tensor(sv, dtype=torch.float64)
    Mt = torch.einsum("ik,mk,jk->mij", U, S, V).to(device, dtype)  # (M, d, d)
    params = {}
    for name, shape in dict(WQ=(dk, D), WK=(dk, D), WV=(dh, D), WO=(d, dh)).items():
        w = torch.stack([torch.randn(*shape, generator=torch.Generator().manual_seed(1000 * seeds[m] + {"WQ": 1, "WK": 2, "WV": 3, "WO": 4}[name]),
                                     dtype=torch.float64) * sigma for m in range(M)])
        params[name] = w.to(device, dtype).requires_grad_(True)
    learn_qk = torch.tensor([a == "learn" for a in attn_modes], device=device)
    uniform = torch.tensor([a == "uniform" for a in attn_modes], device=device)
    oracle = torch.tensor([a == "oracle" for a in attn_modes], device=device)
    with torch.no_grad():
        for name in ("WQ", "WK"):
            params[name][uniform | oracle] = 0.0
    WOV0 = (params["WO"] @ params["WV"]).detach().clone()
    WQK0 = (params["WQ"].transpose(1, 2) @ params["WK"]).detach().clone()
    pos = torch.eye(T, device=device, dtype=dtype)
    n_steps = int(round(t_end / lr))
    rec_every = max(1, n_steps // n_rec)
    Ud, Vd = U.to(device, dtype), V.to(device, dtype)
    recs = {k: [] for k in ("t", "loss", "a1", "modes", "sv_dov", "sv_ov", "sv_dqk", "qk_pos")}
    g = torch.Generator().manual_seed(12345)
    X = torch.randn(n_data, T * d, generator=g, dtype=torch.float64)
    X = X - X.mean(0)
    C = X.T @ X / n_data
    evals, evecs = torch.linalg.eigh(C)
    X = X @ evecs @ torch.diag(evals ** -0.5) @ evecs.T  # exact empirical whitening
    x = X.reshape(n_data, T, d).to(device, dtype)
    E = torch.cat([x, pos.expand(n_data, T, T)], dim=-1)  # (B, T, D)
    y = torch.einsum("mij,bj->mbi", Mt, x[:, 0])
    t0 = time.time()
    for step in range(n_steps + 1):
        q = torch.einsum("mkd,bd->mbk", params["WQ"], E[:, -1])
        k = torch.einsum("mkd,btd->mbtk", params["WK"], E)
        logits = torch.einsum("mbk,mbtk->mbt", q, k) / math.sqrt(dk)
        a = torch.softmax(logits, dim=-1)
        a = torch.where(uniform[:, None, None], torch.full_like(a, 1.0 / T), a)
        onehot1 = torch.zeros_like(a); onehot1[..., 0] = 1.0
        a = torch.where(oracle[:, None, None], onehot1, a)
        z = torch.einsum("mbt,btd->mbd", a, E)
        yhat = torch.einsum("moh,mhd,mbd->mbo", params["WO"], params["WV"], z)
        loss_m = 0.5 * ((y - yhat) ** 2).sum(-1).mean(-1)  # (M,)
        if step % rec_every == 0:
            with torch.no_grad():
                WOV = params["WO"] @ params["WV"]
                WQK = params["WQ"].transpose(1, 2) @ params["WK"]
                recs["t"].append(step * lr)
                recs["loss"].append(loss_m.double().cpu())
                recs["a1"].append(a[..., 0].mean(-1).double().cpu())
                recs["modes"].append(torch.einsum("ik,mij,jk->mk", Ud, WOV[:, :, :d], Vd).double().cpu())
                recs["sv_dov"].append(torch.linalg.svdvals((WOV - WOV0).double()).cpu())
                recs["sv_ov"].append(torch.linalg.svdvals(WOV[:, :, :d].double()).cpu())
                recs["sv_dqk"].append(torch.linalg.svdvals((WQK - WQK0).double()).cpu()[:, :4])
                recs["qk_pos"].append((WQK[:, d + T - 1, d] - WQK[:, d + T - 1, d + 1:].mean(-1)).double().cpu())
            if step % (rec_every * 50) == 0:
                print(f"  step {step}/{n_steps}  t={step * lr:.1f}  loss {loss_m.mean().item():.4f}  "
                      f"a1 {a[..., 0].mean().item():.3f}  {time.time() - t0:.0f}s", flush=True)
        if step == n_steps:
            break
        grads = torch.autograd.grad(loss_m.sum(), list(params.values()))
        with torch.no_grad():
            for (name, p), g in zip(params.items(), grads):
                if name in ("WQ", "WK"):
                    g = g * learn_qk[:, None, None]
                p -= lr * g
    out = {k: torch.stack(v).numpy() if k != "t" else np.array(v) for k, v in recs.items()}
    out.update(S=S.numpy(), attn=np.array(attn_modes), seeds=np.array(seeds), sigma=sigma, lr=lr, T=T, d=d,
               n_data=n_data, U=U.numpy(), V=V.numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--t_end", type=float, default=None)
    a = ap.parse_args()
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.08)
    CACHE.mkdir(exist_ok=True)
    if a.experiment == "showcase":
        sv = [3.0, 1.5, 0.75]
        modes = ["learn", "learn", "learn", "uniform", "oracle"]
        seeds = [0, 1, 2, 0, 0]
        out = train([sv] * len(modes), modes, seeds, t_end=a.t_end or 150.0, lr=0.05, device=a.device)
    elif a.experiment == "sweep_rank1":
        strengths = np.round(np.logspace(np.log10(0.5), np.log10(4.0), 8), 3)
        svals, modes, seeds = [], [], []
        for s in strengths:
            for mode in ("learn", "uniform", "oracle"):
                for sd in range(3 if mode == "learn" else 1):
                    svals.append([float(s)]); modes.append(mode); seeds.append(sd)
        out = train(svals, modes, seeds, t_end=a.t_end or 600.0, lr=0.05, device=a.device)
        out["strengths"] = strengths
    elif a.experiment == "sweep_mode3":
        weak = np.round(np.logspace(np.log10(0.3), np.log10(1.2), 6), 3)
        svals, modes, seeds = [], [], []
        for w in weak:
            for mode in ("learn", "oracle"):
                for sd in range(2 if mode == "learn" else 1):
                    svals.append([3.0, float(w)]); modes.append(mode); seeds.append(sd)
        out = train(svals, modes, seeds, t_end=a.t_end or 300.0, lr=0.05, device=a.device)
        out["weak"] = weak
    else:
        raise SystemExit("unknown experiment")
    np.savez_compressed(CACHE / f"attn_{a.experiment}.npz", **out)
    print("saved", CACHE / f"attn_{a.experiment}.npz")


if __name__ == "__main__":
    main()
