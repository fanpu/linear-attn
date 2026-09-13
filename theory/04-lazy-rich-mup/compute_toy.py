"""2D toy runs for the hero animation and the alpha widget (CPU, numpy; see toy_core.py for the model).

    OMP_NUM_THREADS=2 .venv/bin/python 04-lazy-rich-mup/compute_toy.py hero     # rich + lazy, m=1024
    OMP_NUM_THREADS=2 .venv/bin/python 04-lazy-rich-mup/compute_toy.py widget   # alpha ladder, m=384
"""
import os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_core import rings, train_np

HERE = os.path.dirname(os.path.abspath(__file__))
C = f"{HERE}/cache/toy"
os.makedirs(C, exist_ok=True)
X, Y = rings(900, noise=0.0)
R = 1.25


def schedule(T, n):
    u = np.linspace(0, 1, n)
    return np.unique(np.round(T * u ** 2.5).astype(int))


def job(args):
    tag, m, alpha, lr, T, n_frames, grid_res = args
    path = f"{C}/{tag}.npz"
    if os.path.exists(path):
        return tag, "cached"
    g = np.linspace(-R, R, grid_res)
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    r = train_np(X, Y, m=m, alpha=alpha, lr=lr, steps=T, frames=schedule(T, n_frames), grid=G)
    np.savez_compressed(path, x=X, y=Y, alpha=alpha, lr=lr, m=m, R=R, **r)
    return tag, f"loss {r['loss'][-1]:.4f} acc {r['acc'][-1]:.3f}"


if __name__ == "__main__":
    if sys.argv[1] == "hero":
        jobs = [("hero_rich", 1024, 0.5, 0.1, 40000, 400, 256), ("hero_lazy", 1024, 1000.0, 0.1, 40000, 400, 256)]
    elif sys.argv[1] == "widget":
        jobs = [(f"widget_a{a:g}", 384, a, 0.1, 40000, 72, 64) for a in [0.5, 2, 8, 32, 128, 1000]]
    with Pool(4) as p:
        for tag, msg in p.imap_unordered(job, jobs):
            print(tag, msg, flush=True)
