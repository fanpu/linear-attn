"""Model, data, HVP and sketch helpers for the Ribbon GPU jobs.

Copied (not imported, because the source is a script with argparse at module level) from
art/edge-of-stability/eos_batch.py, so that Stage B reproduces the exact setup of main4.npz:
first 5000 CIFAR-10 train images, per-channel standardised with full-CIFAR statistics,
3072-200-200-10 tanh MLP, PyTorch-default-style uniform init from a float64 CPU generator (seed 0),
loss 0.5*||f(x) - onehot||^2 averaged over examples, float32. The count-sketch uses the same
generator seed (1234) and parameter count, so its hash is bit-identical to main4.npz's sketch.
Everything is M-batched (theta has shape (M, P)) as in the source; the Ribbon jobs use M = 1 for
training and M = batch for loss-grid evaluation.
"""
import math

import numpy as np
import torch
import torchvision

SHAPES = [(200, 3072), (200,), (200, 200), (200,), (10, 200), (10,)]
FAN = [3072, 3072, 200, 200, 200, 200]
SIZES = [math.prod(s) for s in SHAPES]
P = sum(SIZES)  # 656,810


def load_data(n=5000, dt=torch.float32, dev="cuda"):
    ds = torchvision.datasets.CIFAR10(root="/home/fzeng/ml/research/art/data", train=True, download=False)
    full = ds.data.astype(np.float64) / 255.0
    mean, std = full.mean(axis=(0, 1, 2)), full.std(axis=(0, 1, 2))
    Xn = np.transpose((full[:n] - mean) / std, (0, 3, 1, 2)).reshape(n, -1)
    X = torch.tensor(Xn, dtype=dt, device=dev)
    yl = torch.tensor(np.array(ds.targets[:n]), device=dev)
    Y = torch.nn.functional.one_hot(yl, 10).to(dt)
    return X, Y, yl


def init_theta(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(torch.rand(z, generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(f)
                      for z, f in zip(SIZES, FAN)])


def unflatten(th):
    ps, o = [], 0
    M = th.shape[0]
    for s, z in zip(SHAPES, SIZES):
        ps.append(th[:, o:o + z].reshape(M, *s))
        o += z
    return ps


def forward(th, X):  # (M, P) -> (M, n, 10)
    ps = unflatten(th)
    h = torch.tanh(X @ ps[0].transpose(1, 2) + ps[1][:, None])
    h = torch.tanh(h @ ps[2].transpose(1, 2) + ps[3][:, None])
    return h @ ps[4].transpose(1, 2) + ps[5][:, None]


def losses(th, X, Y):
    out = forward(th, X)
    return 0.5 * ((out - Y) ** 2).sum(-1).mean(-1), out


def grad_hvps(th, V, X, Y):
    """Gradient (M,P) and HVPs (M,k,P) of the block-diagonal sum objective (source: eos_batch.py)."""
    t = th.detach().requires_grad_(True)
    lv, out = losses(t, X, Y)
    if V is None:
        gr = torch.autograd.grad(lv.sum(), t)[0]
        return lv.detach(), out.detach(), gr, None
    gr = torch.autograd.grad(lv.sum(), t, create_graph=True)[0]
    kk = V.shape[1]
    HV = torch.stack([torch.autograd.grad(gr, t, V[:, j], retain_graph=(j < kk - 1))[0]
                      for j in range(kk)], 1)
    return lv.detach(), out.detach(), gr.detach(), HV


def rayleigh_ritz(V, HV):
    dev, dt = V.device, V.dtype
    Tm = torch.einsum("mip,mjp->mij", V, HV)
    ev, Q = torch.linalg.eigh((0.5 * (Tm + Tm.transpose(1, 2))).cpu().double())
    ev, Q = ev.flip(-1).to(dev, dt), Q.flip(-1).to(dev, dt)
    V = torch.einsum("mji,mjp->mip", Q, V)
    HV = torch.einsum("mji,mjp->mip", Q, HV)
    return ev, V, HV


def orth(W):
    return torch.linalg.qr(W.transpose(1, 2))[0].transpose(1, 2).contiguous()


def make_sketch(S, dt=torch.float32, dev="cuda"):
    """Count-sketch with the same hash as eos_batch.py (generator seed 1234)."""
    sg = torch.Generator().manual_seed(1234)
    idx = torch.randint(0, S, (P,), generator=sg).to(dev)
    sign = (torch.randint(0, 2, (P,), generator=sg) * 2 - 1).to(dev, dt)

    def sketch(v):  # (M,P) -> (M,S)
        return torch.zeros(v.shape[0], S, dtype=dt, device=dev).index_add_(1, idx, v * sign)

    return sketch
