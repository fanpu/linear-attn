"""Why does Newton-Schulz Muon miss the spectral-norm margin?  Same data/setup as compute_spectral.py.

Variants (all with Nesterov momentum 0.95 unless noted, eta = 0.01, raw-gradient momentum in log scale):
  ns5, ns10, ns20      Muon with 5 / 10 / 20 Newton-Schulz steps (Jordan's quintic)
  ns5_nomom            spectral descent with 5 NS steps, no momentum
Records W and the singular values of the (Frobenius-normalized) matrix fed to NS and of NS's output.
    python compute_ns.py [T]  -> cache/ns.npz
"""
import sys, time
import numpy as np
from core import log_checkpoints
from compute_spectral import make, ns5, k, d, n, Sn

T = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1000000
VARS = {"ns5": (5, True), "ns10": (10, True), "ns20": (20, True), "ns5_nomom": (5, False)}

data = [make(s) for s in range(Sn)]
X = np.stack([D[0] for D in data]); Y = np.stack([D[1] for D in data]); onehot = np.eye(k)[Y]


def grad_hat(W):
    logits = np.einsum("skd,snd->snk", W, X)
    logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    logw = np.where(onehot > 0, -np.inf, logp)
    C = logw.max(axis=(1, 2), keepdims=True)
    Pw = np.exp(logw - C)
    coef = Pw - onehot * Pw.sum(-1, keepdims=True)
    return np.einsum("snk,snd->skd", coef, X), C[:, 0, 0]


Wst = {v: np.zeros((Sn, k, d)) for v in VARS}
mom = {v: np.zeros((Sn, k, d)) for v in VARS}
momC = {v: np.zeros(Sn) for v in VARS}
ts = log_checkpoints(T, 30)
rec = {v: np.empty((len(ts), Sn, k, d)) for v in VARS}
sv_in = {v: np.empty((len(ts), Sn, k)) for v in VARS}
sv_out = {v: np.empty((len(ts), Sn, k)) for v in VARS}
kk = 0
t0 = time.time()
for t in range(1, T + 1):
    for v, (steps, use_mom) in VARS.items():
        G, C = grad_hat(Wst[v])
        if use_mom:
            mom[v] = 0.95 * mom[v] * np.exp(momC[v] - C)[:, None, None] + G
            momC[v] = C
            upd = G + 0.95 * mom[v]
        else:
            upd = G
        out = ns5(upd, steps)
        Wst[v] -= 0.01 * out
        if t == ts[kk]:
            un = upd / np.linalg.norm(upd, axis=(1, 2), keepdims=True)
            sv_in[v][kk] = np.linalg.svd(un, compute_uv=False)
            sv_out[v][kk] = np.linalg.svd(out, compute_uv=False)
            rec[v][kk] = Wst[v]
    if t == ts[kk]:
        kk += 1
        if kk % 30 == 0:
            print(t, f"{time.time() - t0:.0f}s", flush=True)
np.savez("cache/ns.npz", steps=ts, **{f"W_{v}": rec[v] for v in VARS}, **{f"svin_{v}": sv_in[v] for v in VARS},
         **{f"svout_{v}": sv_out[v] for v in VARS})
print("done", time.time() - t0)
