"""Shells: shared pieces for the 3-direction loss volume (art/ml-art-3d.md §2).

Reuses art/loss-landscape/common.py by import (never modified): the Li et al. CIFAR
ResNets, the fixed 1000-image training subset (fixed_subset(1000), seed 0), BN in eval
mode, fp32 with TF32 off, and filter-normalised random directions (seeds 1, 2; biasbn
zeroed).  Adds a third direction (seed 3), top-3 PCA directions of the checkpoint
trajectory, and a batched evaluator that computes K grid points per forward pass with
torch.func.functional_call + vmap.

Coordinates: w = w* + a*d1 + b*d2 + c*d3.  Volumes are stored as loss[k_c, j_b, i_a]
(z = c, row = b, col = a), matching loss-landscape's 2D loss[j_b, i_a].
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
LL_ROOT = "/home/fzeng/ml/research/art/loss-landscape"
sys.path.insert(0, LL_ROOT)
import common as LL  # noqa: E402  (art/loss-landscape/common.py)

N_SUBSET = 1000
SEEDS = (1, 2, 3)


def grid_axis(n=27, h=0.08):
    """Odd n, spacing h, centred on 0 (see NOTES.md Decision: grid)."""
    assert n % 2 == 1
    return (np.arange(n) - n // 2) * h


def load_net(name="resnet20", epoch=None, device="cuda"):
    return LL.load_model(name, epoch, device=device)


def random_dirs3(net, seeds=SEEDS, device="cuda"):
    params = [p for p in net.parameters()]
    return [LL.random_direction(params, s, device=device) for s in seeds]


def pca_dirs3(name="resnet20", device="cuda"):
    d = torch.load(os.path.join(CACHE, "dirs", f"{name}_pca3.pt"), weights_only=False)
    return [[t.to(device) for t in d[k]] for k in ("d1", "d2", "d3")], d


def load_subset(device="cuda", n=N_SUBSET):
    x, y = LL.load_cifar(True, device="cpu")
    idx = torch.tensor(LL.fixed_subset(n))
    return x[idx].to(device), y[idx].to(device)


class SeqEvaluator:
    """The loss-landscape path (LL.LossEvaluator, one point per pass), extended to 3 dirs."""

    def __init__(self, net, x, y, dirs, batch=2500):
        self.ev = LL.LossEvaluator(net, x, y, dirs[0], dirs[1], batch=batch)
        self.d3 = [d.float() for d in dirs[2]]

    @torch.no_grad()
    def __call__(self, a, b, c):
        self.ev.set_point(a, b)
        for p, u in zip(self.ev.params, self.d3):
            p.add_(u, alpha=float(c))
        tot = torch.zeros((), device=self.ev.x.device, dtype=torch.float64)
        cor = torch.zeros((), device=self.ev.x.device, dtype=torch.long)
        X, Y, B = self.ev.x, self.ev.y, self.ev.batch
        for i in range(0, X.shape[0], B):
            out = self.ev.net(X[i:i + B])
            tot += F.cross_entropy(out.double(), Y[i:i + B], reduction="sum")
            cor += (out.argmax(1) == Y[i:i + B]).sum()
        n = X.shape[0]
        return (tot / n).item(), (cor.double() / n).item()


class VmapEvaluator:
    """K points per pass: parameters are stacked as w0 + A d1 + B d2 + C d3 (K copies) and
    the net is vmapped over them with functional_call.  BN buffers are shared (eval mode).
    Always evaluates exactly K points (callers pad), so batch shapes stay fixed."""

    def __init__(self, net, x, y, dirs, K=8, img_batch=1000):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.net = net.float().eval()
        self.names = [n for n, _ in net.named_parameters()]
        self.w0 = [p.detach().clone() for _, p in net.named_parameters()]
        self.dirs = [[t.float() for t in d] for d in dirs]
        self.buffers = {n: b for n, b in net.named_buffers()}
        self.x, self.y = x.float(), y
        self.K, self.img_batch = K, img_batch

        def f(params, xb):
            return torch.func.functional_call(self.net, (params, self.buffers), (xb,))
        self.fv = torch.vmap(f, in_dims=(0, None))

    @torch.no_grad()
    def __call__(self, abc):
        abc = torch.as_tensor(np.asarray(abc, dtype=np.float64), dtype=torch.float32, device=self.x.device)
        assert abc.shape == (self.K, 3), abc.shape
        params = {}
        for n, w, u, v, s in zip(self.names, self.w0, *self.dirs):
            view = (self.K,) + (1,) * w.dim()
            params[n] = (w.unsqueeze(0) + abc[:, 0].view(view) * u.unsqueeze(0)
                         + abc[:, 1].view(view) * v.unsqueeze(0) + abc[:, 2].view(view) * s.unsqueeze(0))
        tot = torch.zeros(self.K, device=self.x.device, dtype=torch.float64)
        cor = torch.zeros(self.K, device=self.x.device, dtype=torch.long)
        X, Y, B = self.x, self.y, self.img_batch
        for i in range(0, X.shape[0], B):
            out = self.fv(params, X[i:i + B])  # (K, b, 10)
            yb = Y[i:i + B]
            lp = torch.log_softmax(out.double(), -1)
            tot -= lp.gather(-1, yb.view(1, -1, 1).expand(self.K, -1, 1)).squeeze(-1).sum(1)
            cor += (out.argmax(-1) == yb.view(1, -1)).sum(1)
        n = X.shape[0]
        return (tot / n).cpu().numpy(), (cor.double() / n).cpu().numpy()
