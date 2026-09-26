"""Why is A erased faster than B is written? Measure the tangent kernel's gain along each page.

The first gradient step on page B changes the output by  df = -lr * K (f - B),  where K = J J^T is
the empirical neural tangent kernel on the pixel grid. Along a unit image v, the rate is the
Rayleigh quotient  R(v) = v^T K v / v^T v = ||J^T v||^2 / ||v||^2, one vector-Jacobian product.
If, after phase 1, R is much larger along A's bands than along B's (or a decoy's), the network's
kernel has aligned to A, and B's gradient scrapes A before it can write B.

    python kernel.py <run_dir> <run-name>        (uses <run-name>_phase1.pt; for a none->B run that
                                                  file is the untrained network: the control)
"""
import json
import os
import sys

import numpy as np
import torch

from metrics import band_masks
from pages import load
from train import make_model, coords


class NS:
    def __init__(self, d):
        self.__dict__.update(d)


def rayleigh(model, X, v):
    model.zero_grad(set_to_none=True)
    f = model(X).reshape(-1)
    (f * v.reshape(-1)).sum().backward()
    g2 = sum(float((p.grad ** 2).sum()) for p in model.parameters() if p.grad is not None)
    return g2 / float((v ** 2).sum())


def main():
    root, name = sys.argv[1], sys.argv[2]
    d = np.load(os.path.join(root, name + ".npz"))
    a = NS(json.loads(str(d["args"])))
    n = a.res
    P = load(n)
    torch.manual_seed(a.seed)
    model = make_model(a, torch.Generator().manual_seed(a.seed))
    mode = sys.argv[3] if len(sys.argv) > 3 else "phase1"
    at_init = mode == "init"
    if mode == "phase1":
        model.load_state_dict(torch.load(os.path.join(root, name + "_phase1.pt")))
    elif mode == "final":
        model.load_state_dict(torch.load(os.path.join(root, name + "_final.pt")))
    X = coords(n, a)
    M, labels = band_masks(n)
    keys = ["A", "B", "C", "D1", "D2", "D3", "D4"]
    R = np.zeros((len(keys), len(labels)))
    for i, k in enumerate(keys):
        img = torch.as_tensor(P[k] - P[k].mean(), dtype=torch.float64)
        F = torch.fft.fft2(img)
        for b in range(len(labels)):
            v = torch.fft.ifft2(F * M[b].double()).real.float()
            R[i, b] = rayleigh(model, X, v)
    np.savez(os.path.join(root, name + {"init": "_kernel_init.npz", "phase1": "_kernel.npz", "final": "_kernel_final.npz"}[mode]), R=R, keys=np.array(keys), labels=np.array(labels))
    np.set_printoptions(precision=3, suppress=False, linewidth=200)
    print(name, {"init": "(untrained)", "phase1": "(end of phase 1)", "final": "(end of phase 2, trained on B)"}[mode], labels)
    for i, k in enumerate(keys):
        print(f"  {k:3s}", np.array2string(R[i], formatter={"float_kind": lambda x: f"{x:9.3g}"}))
    print("  A/B ", np.array2string(R[0] / R[1], precision=2), "  A/decoy", np.array2string(R[0] / R[3:].mean(0), precision=2))


if __name__ == "__main__":
    main()
