"""Break Saxe's assumptions one at a time and measure how far the closed form drifts. CPU, float64.

Unified measurement: the k-th singular value sigma_k(t) of the network map W(t) (basis-free, also what we track
in the attention experiment). t_k = first time sigma_k(t) reaches half of its converged value.
Prediction: t_half(s_k, u0) = ln(s_k/u0 - 1) / (2 s_k)  (Saxe 2014 eq. 11 with u_f = s/2).

  OMP_NUM_THREADS=4 .venv/bin/python 03-saxe-dynamics/compute_breaks.py [init|imbalance|whiten|depth|all]
"""
import pathlib
import sys

import numpy as np
import torch

import saxe_core as sc

torch.set_num_threads(4)
HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
S = np.array([5.0, 2.0, 1.0, 0.5])
N = 8


def sv_traj(W):
    """W: (R, B, n, n) -> sorted singular values (R, B, n)."""
    return np.linalg.svd(W, compute_uv=False)


def crossing_times(t, sv, final, r):
    B = sv.shape[1]
    out = np.full((B, r), np.nan)
    for b in range(B):
        for k in range(r):
            out[b, k] = sc.first_crossing(t, sv[:, b, k], 0.5 * final[b, k])
    return out


def run(Ws_list, Sxx, Syx, lr, T, n_rec=600, L=2):
    Ws = [torch.tensor(np.stack([w[l] for w in Ws_list])) for l in range(L)]
    n = int(T / lr)
    res = sc.gd_deep_linear(Ws, torch.as_tensor(Sxx), torch.as_tensor(Syx), lr, n, max(1, n // n_rec), record_W=True)
    return res["t"], res["W"], res["loss"]


def exp_init():
    Syx, U, _, V = sc.target_from_singular_values(S, N, N, seed=1)
    sigmas = np.logspace(-4, 0, 13)
    seeds = 8
    rng = np.random.default_rng(11)
    inits, meta = [], []
    for si, sg in enumerate(sigmas):
        for b in range(seeds):
            w = sc.gaussian_init([N, N, N], sg, rng)
            inits.append(w)
            meta.append((si, sc.effective_u0_two_layer(w[0], w[1], U, V, 4)))
    t, W, loss = run(inits, np.eye(N), Syx, 2e-4, 26.0)
    sv = sv_traj(W)
    final = np.tile(S, (len(inits), 1))
    tk = crossing_times(t, sv, final, 4)
    u0eff = np.stack([m[1] for m in meta])
    # also keep per-mode projections for a few example curves
    modes = np.einsum("ia,rbij,ja->rba", U[:, :4], W, V[:, :4])
    np.savez_compressed(CACHE / "break_init.npz", t=t, sigmas=sigmas, seeds=seeds, tk=tk, u0eff=u0eff,
                        u0nom=N * sigmas ** 2 / 2, sv=sv[:, :, :4].astype(np.float32), loss=loss,
                        modes=modes.astype(np.float32), s=S)
    print("init done")


def exp_imbalance():
    Syx, U, _, V = sc.target_from_singular_values(S, N, N, seed=1)
    rhos = np.logspace(0, 2.5, 11)
    u0 = 1e-5
    inits = [sc.decoupled_init(U, V, [N, N, N], u0, 2, imbalance=r) for r in rhos]
    t, W, loss = run(inits, np.eye(N), Syx, 1e-4, 14.0)
    sv = sv_traj(W)
    tk = crossing_times(t, sv, np.tile(S, (len(rhos), 1)), 4)
    # exact prediction: integrate the scalar 2D flow a' = b(s-ab), b' = a(s-ab)
    exact = np.zeros((len(rhos), 4))
    for i, r in enumerate(rhos):
        for k, s in enumerate(S):
            a, b = r * np.sqrt(u0), np.sqrt(u0) / r
            h, tt = 1e-4, 0.0
            while a * b < s / 2:
                def f(a, b):
                    e = s - a * b
                    return b * e, a * e
                k1 = f(a, b); k2 = f(a + h / 2 * k1[0], b + h / 2 * k1[1])
                k3 = f(a + h / 2 * k2[0], b + h / 2 * k2[1]); k4 = f(a + h * k3[0], b + h * k3[1])
                a_new = a + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                b_new = b + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                if a_new * b_new >= s / 2:
                    frac = (s / 2 - a * b) / (a_new * b_new - a * b)
                    tt += frac * h
                    break
                a, b, tt = a_new, b_new, tt + h
            exact[i, k] = tt
    np.savez_compressed(CACHE / "break_imbalance.npz", t=t, rhos=rhos, u0=u0, tk=tk, exact=exact,
                        sv=sv[:, :, :4].astype(np.float32), loss=loss, s=S)
    print("imbalance done")


def exp_whiten():
    Syx, U, _, V = sc.target_from_singular_values(S, N, N, seed=1)
    kappas = np.array([1, 1.25, 1.6, 2, 3, 5, 10, 30, 100.0])
    seeds = 8
    u0 = 1e-5
    rng = np.random.default_rng(5)
    Sxxs, finals, inits, tags = [], [], [], []
    for variant in ("aligned", "rotated"):
        for kap in kappas:
            for b in range(seeds):
                lam = np.exp(rng.uniform(-0.5, 0.5, N) * np.log(kap))
                lam[0], lam[1] = kap ** 0.5 if b % 2 == 0 else kap ** -0.5, kap ** -0.5 if b % 2 == 0 else kap ** 0.5
                Q = V if variant == "aligned" else np.linalg.qr(rng.standard_normal((N, N)))[0]
                Sxx = Q @ np.diag(lam) @ Q.T
                Sxxs.append(Sxx)
                finals.append(np.linalg.svd(Syx @ np.linalg.inv(Sxx), compute_uv=False)[:4])
                inits.append(sc.decoupled_init(U, V, [N, N, N], u0, 2))
                tags.append((variant, kap, b))
    Ws = [torch.tensor(np.stack([w[l] for w in inits])) for l in range(2)]
    lr, T = 1e-4, 40.0
    n = int(T / lr)
    res = sc.gd_deep_linear(Ws, torch.tensor(np.stack(Sxxs)), torch.tensor(Syx), lr, n, n // 800, record_W=True)
    sv = sv_traj(res["W"])
    tk = crossing_times(res["t"], sv, np.stack(finals), 4)
    modes = np.einsum("ia,rbij,ja->rba", U[:, :4], res["W"], V[:, :4])
    final_modes = np.stack([np.einsum("ia,ij,ja->a", U[:, :4], Syx @ np.linalg.inv(X), V[:, :4]) for X in Sxxs])
    lam_modes = np.stack([np.einsum("ja,jk,ka->a", V[:, :4], X, V[:, :4]) for X in Sxxs])  # v_a^T Sxx v_a
    tm = np.full((len(inits), 4), np.nan)
    for b in range(len(inits)):
        for k in range(4):
            if final_modes[b, k] > 0:
                tm[b, k] = sc.first_crossing(res["t"], modes[:, b, k], 0.5 * final_modes[b, k])
    np.savez_compressed(CACHE / "break_whiten.npz", t=res["t"], kappas=kappas, seeds=seeds, tk=tk, tm=tm, u0=u0,
                        variant=np.array([x[0] for x in tags]), kappa=np.array([x[1] for x in tags]),
                        sv=sv[:, :, :4].astype(np.float32), modes=modes.astype(np.float32), finals=np.stack(finals),
                        final_modes=final_modes, lam_modes=lam_modes, s=S)
    print("whiten done")


def exp_depth():
    """(i) decoupled balanced init at depth L: exact ODE vs 2-layer formula; (ii) random init, many modes, 1/s law."""
    out = {}
    u0 = 1e-3
    Syx, U, _, V = sc.target_from_singular_values(S, N, N, seed=1)
    for L in (2, 3, 4):
        inits = [sc.decoupled_init(U, V, [N] * (L + 1), u0, L)]
        t, W, loss = run(inits, np.eye(N), Syx, 2e-4, 12.0, L=L)
        modes = np.einsum("ia,rbij,ja->rba", U[:, :4], W, V[:, :4])[:, 0]
        out[f"dec_t_L{L}"], out[f"dec_modes_L{L}"], out[f"dec_loss_L{L}"] = t, modes, loss[:, 0]
    # random init, 10 modes with log-spaced strengths, width 16
    n_modes, width = 10, 16
    s_many = np.logspace(np.log10(4.0), np.log10(0.4), n_modes)
    Syx2, U2, _, V2 = sc.target_from_singular_values(s_many, width, width, seed=3)
    seeds = 6
    for L in (2, 3, 4):
        rng = np.random.default_rng(100 + L)
        sigma = ({2: 1e-4, 3: 1e-3, 4: 1e-2}[L] / width ** ((L - 1) / 2)) ** (1.0 / L)
        inits = [sc.gaussian_init([width] * (L + 1), sigma, rng) for _ in range(seeds)]
        T = {2: 18.0, 3: 36.0, 4: 36.0}[L]
        lr = {2: 2e-4, 3: 2e-4, 4: 2e-4}[L]
        t, W, loss = run(inits, np.eye(width), Syx2, lr, T, n_rec=1500, L=L)
        sv = sv_traj(W)
        tk = crossing_times(t, sv, np.tile(s_many, (seeds, 1)), n_modes)
        modes = np.einsum("ia,rbij,ja->rba", U2[:, :n_modes], W, V2[:, :n_modes])
        out[f"rnd_t_L{L}"], out[f"rnd_tk_L{L}"], out[f"rnd_sigma_L{L}"] = t, tk, sigma
        out[f"rnd_modes_L{L}"] = modes.astype(np.float32)
        print("depth L", L, "tk mean", np.nanmean(tk, 0).round(2))
    out.update(s=S, u0=u0, s_many=s_many, width=width)
    np.savez_compressed(CACHE / "break_depth.npz", **out)
    print("depth done")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ("init", "imbalance", "whiten", "depth"):
        if which in (name, "all"):
            globals()[f"exp_{name}"]()
