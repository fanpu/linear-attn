"""Reduced ('Saxe-style') model of the attention head in compute_attention.py.

Assumptions: attention logits depend only on positions (content terms ~ 0), the non-target positions are
symmetric, and the value/output weights only matter through their projections on the target's singular vectors.
Then, with alpha = attention on token 1 and A2 = alpha^2 + (1-alpha)^2/(T-1),

    L = 1/2 sum_k [ s_k^2 - 2 s_k u_k alpha + u_k^2 A2 ],     u_k = a_k . b_k          (OV mode k)
    alpha = e^delta / (e^delta + T - 1),                      delta = q . kappa / sqrt(d_k)   (QK "mode")

and gradient flow on the vectors a_k, b_k (in the d_h-dim hidden space) and q, kappa (in the d_k-dim key space):
    da_k/dt = -dL/du_k b_k,   db_k/dt = -dL/du_k a_k,
    dq/dt = g kappa / sqrt(d_k),   dkappa/dt = g T/(T-1) q / sqrt(d_k),   g = -dL/ddelta.
Initial vectors are read off the *same* random init the real model used (same seeds).

  .venv/bin/python 03-saxe-dynamics/attn_reduced.py      -> cache/attn_reduced.npz
"""
import math
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent


def init_vectors(seed, d, T, dk, dh, sigma, U, V, r):
    D = d + T
    W = {}
    for name, shape in dict(WQ=(dk, D), WK=(dk, D), WV=(dh, D), WO=(d, dh)).items():
        g = torch.Generator().manual_seed(1000 * seed + {"WQ": 1, "WK": 2, "WV": 3, "WO": 4}[name])
        W[name] = torch.randn(*shape, generator=g, dtype=torch.float64).numpy() * sigma
    a = [W["WV"][:, :d] @ V[:, k] for k in range(r)]
    b = [W["WO"].T @ U[:, k] for k in range(r)]
    q = W["WQ"][:, d + T - 1]
    kappa = W["WK"][:, d] - W["WK"][:, d + 1:].mean(1)
    return np.array(a), np.array(b), q, kappa


def integrate(s, a, b, q, kappa, T, dk, t_end, dt=0.01, learn_attn=True, alpha_fixed=None, n_rec=2000):
    s = np.asarray(s, float)
    n = int(t_end / dt)
    every = max(1, n // n_rec)
    ts, us, als = [], [], []
    y = np.concatenate([a.ravel(), b.ravel(), q, kappa])
    r, dh = a.shape

    def unpack(y):
        A = y[: r * dh].reshape(r, dh); B = y[r * dh: 2 * r * dh].reshape(r, dh)
        return A, B, y[2 * r * dh: 2 * r * dh + dk], y[2 * r * dh + dk:]

    def rhs(y):
        A, B, q_, k_ = unpack(y)
        u = np.sum(A * B, 1)
        if alpha_fixed is not None:
            al = alpha_fixed
        else:
            delta = q_ @ k_ / math.sqrt(dk)
            al = 1.0 / (1.0 + (T - 1) * math.exp(-delta)) if delta > -50 else 0.0
        A2 = al ** 2 + (1 - al) ** 2 / (T - 1)
        dLdu = -s * al + u * A2
        dA = -dLdu[:, None] * B
        dB = -dLdu[:, None] * A
        if learn_attn and alpha_fixed is None:
            dLdal = np.sum(-s * u + 0.5 * u ** 2 * (2 * al - 2 * (1 - al) / (T - 1)))
            g = -dLdal * al * (1 - al)
            dq = g * k_ / math.sqrt(dk)
            dk_ = g * T / (T - 1) * q_ / math.sqrt(dk)
        else:
            dq = np.zeros(dk); dk_ = np.zeros(dk)
        return np.concatenate([dA.ravel(), dB.ravel(), dq, dk_]), u, al

    for i in range(n + 1):
        k1, u, al = rhs(y)
        if i % every == 0:
            ts.append(i * dt); us.append(u); als.append(al)
        if i == n:
            break
        k2 = rhs(y + dt / 2 * k1)[0]; k3 = rhs(y + dt / 2 * k2)[0]; k4 = rhs(y + dt * k3)[0]
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(ts), np.array(us), np.array(als)


def run_file(name, t_end, dt):
    d = np.load(HERE / f"cache/attn_{name}.npz")
    U, V = d["U"], d["V"]
    T, dim = int(d["T"]), int(d["d"])
    out = {}
    for i, (att, seed) in enumerate(zip(d["attn"], d["seeds"])):
        s = d["S"][i]
        r = int(np.sum(s > 0))
        a, b, q, kap = init_vectors(int(seed), dim, T, 16, 16, float(d["sigma"]), U, V, r)
        kw = dict(learn_attn=att == "learn", alpha_fixed={"learn": None, "uniform": 1.0 / T, "oracle": 1.0}[str(att)])
        ts, us, als = integrate(s[:r], a, b, q, kap, T, 16, t_end, dt=dt, **kw)
        out[f"u_{i}"], out[f"a1_{i}"] = us, als
        out["t"] = ts
    np.savez_compressed(HERE / f"cache/attn_reduced_{name}.npz", **out)
    print("saved", name)


if __name__ == "__main__":
    run_file("showcase", 150.0, 0.02)
    run_file("sweep_rank1", 600.0, 0.02)
    run_file("sweep_mode3", 300.0, 0.02)
