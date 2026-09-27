"""CPU worker for train_gpt.py: fresh entry-shuffles, Grams and eigenvalues (single-threaded torch).

For each matrix W (r x c) returns eigenvalues, descending, of the smaller Gram W^T W / max(r, c)
for W itself and for two independent entry-shuffles of W (fresh permutations every checkpoint).
"""
import numpy as np
import torch


def init():
    torch.set_num_threads(1)


def _gram(W):
    r, c = W.shape
    return (W.T @ W if c <= r else W @ W.T) / max(r, c)


def eig(ci, mats, seed):
    """mats: list of (name, float32 numpy matrix) -> (ci, {name: (lam, s1, s2)})."""
    g = torch.Generator().manual_seed(int(seed))
    groups = {}
    for n, W in mats:
        W = torch.from_numpy(W)
        flat = W.reshape(-1)
        Gs = [_gram(W)] + [_gram(flat[torch.randperm(flat.numel(), generator=g)].view_as(W)) for _ in range(2)]
        groups.setdefault(Gs[0].shape, []).append((n, Gs))
    res = {}
    for items in groups.values():
        G = torch.stack([G for _, Gs in items for G in Gs])
        ev = torch.linalg.eigvalsh(G).flip(-1).clamp_min(0).numpy()
        for j, (n, _) in enumerate(items):
            res[n] = (ev[3 * j], ev[3 * j + 1], ev[3 * j + 2])
    return ci, res
