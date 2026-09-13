"""Core math for the lazy/NTK part: data, 2-layer ReLU net, empirical NTK via torch.func, closed forms.

Model (NTK parameterization, Jacot et al. 2018; Lee et al. 2019):
    f(x; W, a) = alpha/sqrt(m) * sum_j a_j relu(w_j . x)   -  (centered at init if alpha-scaling is on)
with W_ij, a_j ~ N(0, 1). Inputs are unit-norm, so pre-activations are O(1).
Loss: (1/alpha^2) * (1/2n) sum_i (f(x_i) - y_i)^2, full-batch gradient descent.
"""
import math
import pickle

import numpy as np
import torch
from torch.func import functional_call, jvp, vjp, vmap

CIFAR = "/home/fzeng/ml/research/art/data/cifar-10-batches-py"


def cifar_binary(c0=0, c1=1, n_train=1000, n_test=1000, pool=4, seed=0):
    """Two CIFAR-10 classes, avg-pooled 32->8 (192 dims), train-mean centered, unit-norm rows. y = +-1."""
    def load(files):
        X, Y = [], []
        for f in files:
            with open(f"{CIFAR}/{f}", "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            X.append(d[b"data"]); Y.append(np.array(d[b"labels"]))
        return np.concatenate(X).astype(np.float64) / 255.0, np.concatenate(Y)

    rng = np.random.default_rng(seed)
    out = []
    for files, n in [([f"data_batch_{i}" for i in range(1, 6)], n_train), (["test_batch"], n_test)]:
        X, Y = load(files)
        idx = np.concatenate([rng.permutation(np.where(Y == c)[0])[: n // 2] for c in (c0, c1)])
        idx = rng.permutation(idx)
        x = X[idx].reshape(-1, 3, 32, 32)
        x = x.reshape(-1, 3, 32 // pool, pool, 32 // pool, pool).mean(axis=(3, 5)).reshape(len(idx), -1)
        out.append((x, np.where(Y[idx] == c1, 1.0, -1.0)))
    (xtr, ytr), (xte, yte) = out
    mu = xtr.mean(0)
    xtr, xte = xtr - mu, xte - mu
    xtr /= np.linalg.norm(xtr, axis=1, keepdims=True)
    xte /= np.linalg.norm(xte, axis=1, keepdims=True)
    return xtr, ytr, xte, yte


def init_params(d, m, seed, device, dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(m, d, generator=g, dtype=dtype)
    a = torch.randn(m, generator=g, dtype=dtype)
    return {"W": W.to(device), "a": a.to(device)}


def net(params, x):
    """Raw NTK-parameterized 2-layer ReLU net, x: (n, d) -> (n,)"""
    m = params["a"].shape[0]
    return torch.relu(x @ params["W"].T) @ params["a"] / math.sqrt(m)


def empirical_ntk(params, x1, x2=None, chunk=None, budget=1.5e9):
    """Theta(x1, x2) = J(x1) J(x2)^T without materializing the full Jacobian of all points.

    For a chunk C of x1 we pull back unit covectors through the net (VJP) to get the rows J_C,
    then push each row forward over x2 (JVP) to get J(x2) J_C^T. Memory: chunk * n_params.
    """
    x2 = x1 if x2 is None else x2
    keys = list(params)
    if chunk is None:  # keep the vmapped JVP activations (chunk * n2 * m numbers) under `budget` bytes
        m = params["a"].shape[0]
        chunk = max(1, int(budget / (x2.shape[0] * m * x2.element_size() * 3)))
    out = torch.empty(x1.shape[0], x2.shape[0], dtype=params[keys[0]].dtype, device=x1.device)

    def f2(*p):
        return net(dict(zip(keys, p)), x2)

    for s in range(0, x1.shape[0], chunk):
        xc = x1[s:s + chunk]
        _, pull = vjp(lambda *p: net(dict(zip(keys, p)), xc), *[params[k] for k in keys])
        rows = vmap(pull)(torch.eye(xc.shape[0], dtype=xc.dtype, device=xc.device))  # tuple of (c, *shape)
        push = lambda *t: jvp(f2, tuple(params[k] for k in keys), t)[1]
        out[s:s + chunk] = vmap(push)(*rows)  # (c, n2)
    return out


def relu_ntk_analytic(x1, x2):
    """Infinite-width NTK of f = 1/sqrt(m) sum a_j relu(w_j.x) with N(0,1) weights (no bias).

    Theta = E[relu(w.x) relu(w.x')] + (x.x') E[relu'(w.x) relu'(w.x')]
          = |x||x'|/(2pi) (sin t + (pi - t) cos t) + (x.x') (pi - t)/(2pi)
    """
    n1 = np.linalg.norm(x1, axis=1)[:, None]
    n2 = np.linalg.norm(x2, axis=1)[None, :]
    dot = x1 @ x2.T
    c = np.clip(dot / (n1 * n2), -1.0, 1.0)
    t = np.arccos(c)
    k1 = n1 * n2 / (2 * np.pi) * (np.sin(t) + (np.pi - t) * c)
    k0 = (np.pi - t) / (2 * np.pi)
    return k1 + dot * k0


def linearized_gd(K_all_tr, K_tr_tr, f0_all, f0_tr, y, lr, steps, record=None):
    """Discrete GD on the model linearized at init, loss (1/2n)|f - y|^2, exactly (float64, numpy).

    f_tr(t+1) = f_tr(t) - lr/n K_tr_tr (f_tr(t) - y);   f_all(t+1) likewise with K_all_tr.
    Solved in the eigenbasis of K_tr_tr: r(t) = (I - lr/n K)^t r(0).
    """
    n = len(y)
    lam, U = np.linalg.eigh(K_tr_tr)
    r0 = U.T @ (f0_tr - y)
    ts = record if record is not None else [steps]
    outs = []
    for t in ts:
        decay = (1 - lr / n * lam) ** t
        # f_all(t) = f0_all - K_all_tr U diag((1 - decay)/lam) r0  (sum of geometric series)
        with np.errstate(divide="ignore", invalid="ignore"):
            coef = np.where(lam > 1e-12, (1 - decay) / lam, lr / n * t)
        outs.append(f0_all - K_all_tr @ (U @ (coef * r0)))
    return outs if record is not None else outs[0]


def kernel_alignment(K, y):
    """Centered-free kernel-target alignment <K, yy^T>_F / (|K|_F |yy^T|_F)."""
    return float(y @ K @ y / (np.linalg.norm(K) * (y @ y)))
