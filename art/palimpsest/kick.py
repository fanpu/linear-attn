"""Scraping or overwriting? The matched random-kick control.

At each phase-2 snapshot the network has moved a distance ||dW_l|| (per weight tensor) from where
phase 1 left it. Take the phase-1 network and move every tensor the *same* distance in a random
Gaussian direction instead. If A's bands vanish as fast under the random kick as under training
on B, forgetting is generic: any displacement of that size scrapes A. If A survives the random
kick much better, training on B removes A specifically.

The measured quantity is the least-squares amplitude of A's band-k content in the output,
<P_k f, P_k A>/<P_k A, P_k A> (no B to subtract: the kicked network was never shown B).
CPU only: one forward pass per snapshot per draw.

    python kick.py cache/runs <run-name> [ndraws]
"""
import json
import os
import sys

import numpy as np
import torch

from metrics import Ghost
from pages import load
from train import make_model, coords

HERE = os.path.dirname(os.path.abspath(__file__))


class NS:
    def __init__(self, d):
        self.__dict__.update(d)


def main():
    root, name = sys.argv[1], sys.argv[2]
    ndraws = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    d = np.load(os.path.join(root, name + ".npz"))
    a = NS(json.loads(str(d["args"])))
    n = a.res
    P = load(n)
    torch.manual_seed(a.seed)
    model = make_model(a, torch.Generator().manual_seed(a.seed))
    s1 = torch.load(os.path.join(root, name + "_phase1.pt"))
    names = [str(x) for x in d["param_names"]]
    X = coords(n, a)
    G = Ghost({k: P[k] for k in ["A", "D1", "D2", "D3", "D4"]}, n)
    dw = d["snap_dw"]  # (T, L)
    steps = d["snap_steps"]
    rng = torch.Generator().manual_seed(1234)
    out = np.zeros((ndraws, len(steps), 5, len(G.labels)), np.float32)
    kick_f = np.zeros((len(steps), n, n), np.float16)  # draw 0's outputs, for plates
    dirs = []
    for r in range(ndraws):
        dirs.append({k: torch.randn(s1[k].shape, generator=rng) for k in names})
    for r in range(ndraws):
        for t in range(len(steps)):
            sd = dict(s1)
            for li, k in enumerate(names):
                u = dirs[r][k]
                sd[k] = s1[k] + u * (float(dw[t, li]) / float(u.norm()))
            model.load_state_dict(sd)
            with torch.no_grad():
                f = model(X).reshape(n, n).numpy()
            g, _ = G(f)
            out[r, t] = g
            if r == 0:
                kick_f[t] = f
        print("draw", r, "done", flush=True)
    np.savez_compressed(os.path.join(root, name + "_kick.npz"), g=out, kick_f=kick_f, steps=steps, labels=np.array(G.labels),
                        names=np.array(G.names))
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    gt = d["ghost"][:, 0]  # trained: A's amplitude in (f - B)
    gk = out[:, :, 0].mean(0)
    for t in range(0, len(steps), max(1, len(steps) // 12)):
        print(f"step {steps[t]:5d} |dW| {dw[t].sum():8.3f}  trained {gt[t]}  kicked {gk[t]}")


if __name__ == "__main__":
    main()
