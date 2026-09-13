"""Reproduction runs (CPU, float64). Writes cache/reproduce.npz and cache/semantic.npz.

  OMP_NUM_THREADS=4 .venv/bin/python 03-saxe-dynamics/compute_reproduce.py
"""
import pathlib

import numpy as np
import torch

import saxe_core as sc

torch.set_num_threads(4)
HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

S_TOY = np.array([5.0, 2.0, 1.0, 0.5])
N = 8


def toy_runs():
    Syx, U, _, V = sc.target_from_singular_values(S_TOY, N, N, seed=1)
    r = len(S_TOY)
    Ur, Vr = U[:, :r], V[:, :r]
    lr, T = 5e-5, 14.0
    n = int(T / lr)
    rec = n // 700
    out = {}
    # (1) decoupled balanced init, u0 = 1e-5
    u0 = 1e-5
    Ws = [torch.tensor(w)[None] for w in sc.decoupled_init(U, V, [N, N, N], u0, 2)]
    res = sc.gd_deep_linear(Ws, torch.eye(N), torch.tensor(Syx), lr, n, rec, U=Ur, V=Vr)
    out.update(t=res["t"], dec_modes=res["modes"][:, 0], dec_loss=res["loss"][:, 0], dec_u0=u0)
    # (2) small random Gaussian init, many seeds; record full U^T W V (mode-mixing matrix)
    sigma, seeds = 2e-3, 16
    rng = np.random.default_rng(7)
    inits = [sc.gaussian_init([N, N, N], sigma, rng) for _ in range(seeds)]
    Ws = [torch.tensor(np.stack([w[l] for w in inits])) for l in range(2)]
    res = sc.gd_deep_linear(Ws, torch.eye(N), torch.tensor(Syx), lr, n, rec, U=Ur, V=Vr, record_W=True)
    UWV = np.einsum("ia,rbij,jc->rbac", U, res["W"], V)  # (R, B, N, N) full mode matrix
    u0_eff = np.stack([sc.effective_u0_two_layer(w[0], w[1], U, V, r) for w in inits])  # (B, r)
    out.update(rnd_modes=res["modes"], rnd_loss=res["loss"], rnd_UWV=UWV[:, :, :6, :6].astype(np.float32),
               rnd_u0eff=u0_eff, sigma=sigma)
    out.update(s=S_TOY, Syx=Syx, U=U, V=V, loss_const=0.5 * np.sum(S_TOY ** 2))
    np.savez_compressed(CACHE / "reproduce.npz", **out)
    for a, s in enumerate(S_TOY):
        pred = sc.sigmoid_mode(out["t"], s, u0)
        th_m = sc.first_crossing(out["t"], out["dec_modes"][:, a], s / 2)
        print(f"decoupled s={s}: max|meas-analytic|/s = {np.max(np.abs(out['dec_modes'][:, a] - pred)) / s:.2e}"
              f"  t_half meas {th_m:.4f} pred {sc.t_half(s, u0):.4f}")
        th_r = np.array([sc.first_crossing(out["t"], out["rnd_modes"][:, b, a], s / 2) for b in range(seeds)])
        th_p = sc.t_half(s, u0_eff[:, a])
        print(f"   random  s={s}: t_half rel err mean {np.mean((th_r - th_p) / th_p):+.3%}  max {np.max(np.abs(th_r - th_p) / th_p):.3%}")


def semantic_run():
    d = sc.semantic_dataset()
    P, NF = d["Y"].shape
    H = 16
    lr, T = 5e-4, 18.0
    n = int(T / lr)
    rec = n // 900
    rng = np.random.default_rng(3)
    sigma = 1e-3
    W1, W2 = sc.gaussian_init([P, H, NF], sigma, rng)
    Ws = [torch.tensor(W1)[None], torch.tensor(W2)[None]]
    res = sc.gd_deep_linear(Ws, torch.tensor(d["Sxx"]), torch.tensor(d["Syx"]), lr, n, rec,
                            U=d["U"], V=d["V"], record_W=True, record_layers=(0,))
    u0_eff = sc.effective_u0_two_layer(W1, W2, d["U"], d["V"], P)
    loss_const = 0.5 * np.sum(d["Y"] ** 2) / P
    np.savez_compressed(
        CACHE / "semantic.npz", t=res["t"], modes=res["modes"][:, 0], loss=res["loss"][:, 0] + loss_const,
        W=res["W"][:, 0], W1=res["W1"][:, 0], u0_eff=u0_eff, s=d["s"], U=d["U"], V=d["V"], Y=d["Y"],
        items=np.array(d["items"]), features=np.array(d["features"]), category=np.array(sc.CATEGORY),
        sigma=sigma, H=H)
    for a, s in enumerate(d["s"]):
        th = sc.first_crossing(res["t"], res["modes"][:, 0, a], s / 2)
        print(f"semantic mode {a} s={s:.3f} u0eff={u0_eff[a]:.2e}: t_half meas {th:.3f} pred {sc.t_half(s, u0_eff[a]):.3f}")


if __name__ == "__main__":
    toy_runs()
    semantic_run()
