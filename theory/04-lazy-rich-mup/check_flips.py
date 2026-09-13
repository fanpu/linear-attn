"""Why does the ReLU tangent kernel move like alpha^-1/2 (not alpha^-1) deep in the lazy regime?

Hypothesis: activation-pattern flips. Theta = K_a + K_W with
    K_a(x,x')  = 1/m sum_j relu(w_j.x) relu(w_j.x')                (continuous in w)
    K_W(x,x')  = (x.x')/m sum_j a_j^2 1[w_j.x>0] 1[w_j.x'>0]        (jumps when a neuron crosses its kink)
Train the centred alpha-model (same setup as compute_ntk.py alpha, CPU, float64 for a clean small-change check)
and measure each part's relative change, plus the fraction of flipped (neuron, input) pairs.

    DEV=cpu .venv/bin/python 04-lazy-rich-mup/check_flips.py
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntk_core import cifar_binary, init_params, net

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
xtr, ytr, _, _ = cifar_binary()
X = torch.tensor(xtr[:400]); Y = torch.tensor(ytr[:400])
m = 1024


def parts(p):
    pre = X @ p["W"].T
    act, on = torch.relu(pre), (pre > 0).double()
    Ka = act @ act.T / m
    Kw = (X @ X.T) * ((on * p["a"] ** 2) @ on.T) / m
    return Ka, Kw, on


out = {}
for alpha in [1.0, 10.0, 100.0, 1000.0]:
    p = init_params(192, m, 0, "cpu", dtype=torch.float64)
    p0 = {k: v.clone() for k, v in p.items()}
    f0 = net(p0, X)
    for t in range(3000):
        W, a = p["W"].requires_grad_(True), p["a"].requires_grad_(True)
        loss = 0.5 * ((alpha * (net({"W": W, "a": a}, X) - f0) - Y) ** 2).mean() / alpha ** 2
        gW, ga = torch.autograd.grad(loss, (W, a))
        with torch.no_grad():
            p = {"W": W - 0.5 * gW, "a": a - 0.5 * ga}
    Ka0, Kw0, on0 = parts(p0)
    Ka1, Kw1, on1 = parts(p)
    rel = lambda A, B: float(torch.linalg.norm(A - B) / torch.linalg.norm(B))
    # split K_W's change into "same pattern, new a^2" and "pattern flips"
    Kw_noflip = (X @ X.T) * ((on0 * p["a"] ** 2) @ on0.T) / m
    out[alpha] = dict(dW=rel(p["W"], p0["W"]), dKa=rel(Ka1, Ka0), dKw=rel(Kw1, Kw0),
                      dKw_flips=rel(Kw1, Kw_noflip) * float(torch.linalg.norm(Kw_noflip) / torch.linalg.norm(Kw0)),
                      flip_frac=float((on1 != on0).double().mean()))
    print(alpha, {k: f"{v:.3e}" for k, v in out[alpha].items()}, flush=True)
json.dump(out, open(f"{HERE}/cache/check_flips.json", "w"), indent=1)
