"""dense-landscape shared pieces: CIFAR-10 on GPU, a small 4-conv BN net, Li et al.
filter-normalised directions, and a *batched* evaluator that runs P perturbed copies of
the net in one forward pass (grouped convolutions), with BatchNorm either in eval mode
(running statistics) or in train mode (statistics of each minibatch of `gs` images).
"""
import os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
GALLERY = os.path.join(ROOT, "gallery")
DATA = "/home/fzeng/ml/research/art/data/cifar-10-batches-py"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
WIDTHS = (16, 32, 64, 64)
EPS = 1e-5


def load_cifar(train=True, device="cuda", dtype=torch.float32):
    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    xs, ys = [], []
    for f in files:
        with open(os.path.join(DATA, f), "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        xs.append(d["data"]); ys.extend(d["labels"])
    x = torch.tensor(np.concatenate(xs)).view(-1, 3, 32, 32).to(device=device, dtype=dtype) / 255.0
    m = torch.tensor(MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return (x - m) / s, torch.tensor(ys, device=device, dtype=torch.long)


class ConvNet4(nn.Module):
    """conv3x3-BN-ReLU x4 (16@32, 32@16, 64@8, 64@4, 2x2 maxpool between), GAP, linear.
    ~3.4 M multiply-adds per image, 69k parameters."""

    def __init__(self, widths=WIDTHS):
        super().__init__()
        cin = 3
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for c in widths:
            self.convs.append(nn.Conv2d(cin, c, 3, 1, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(c)); cin = c
        self.fc = nn.Linear(cin, 10)

    def forward(self, x):
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            x = F.relu(b(c(x)))
            if i < len(self.convs) - 1:
                x = F.max_pool2d(x, 2)
        return self.fc(x.mean((2, 3)))


def random_direction(net, seed):
    """Li et al. 'weights' direction, filter-normalised, ignore='biasbn': one N(0,1) tensor
    per parameter; dim<=1 tensors (BN gamma/beta, fc bias) zeroed; each filter (row) of
    a dim>=2 tensor rescaled to the norm of the same filter of the trained weights."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, w in net.named_parameters():
        d = torch.randn(w.shape, generator=g, dtype=torch.float64)
        wd = w.detach().cpu().double()
        if d.dim() <= 1:
            d.zero_()
        else:
            dn = d.flatten(1).norm(dim=1); wn = wd.flatten(1).norm(dim=1)
            d.mul_((wn / (dn + 1e-10)).view(-1, *([1] * (d.dim() - 1))))
        out[name] = d.float()
    return out


class BatchedNet:
    """P copies of ConvNet4 at w0 + a_p dx + b_p dy, evaluated in one pass."""

    def __init__(self, net, dx, dy, device="cuda"):
        sd = {k: v.detach().to(device).float() for k, v in net.state_dict().items()}
        self.nl = len(net.convs)
        self.w0 = [sd[f"convs.{i}.weight"] for i in range(self.nl)] + [sd["fc.weight"]]
        self.dx = [dx[f"convs.{i}.weight"].to(device) for i in range(self.nl)] + [dx["fc.weight"].to(device)]
        self.dy = [dy[f"convs.{i}.weight"].to(device) for i in range(self.nl)] + [dy["fc.weight"].to(device)]
        self.bn = [(sd[f"bns.{i}.weight"], sd[f"bns.{i}.bias"], sd[f"bns.{i}.running_mean"],
                    sd[f"bns.{i}.running_var"]) for i in range(self.nl)]
        self.fcb = sd["fc.bias"]

    def weights(self, a, b):
        """a, b: (P,) tensors -> list of (P, *shape) weights."""
        out = []
        for w, u, v in zip(self.w0, self.dx, self.dy):
            sh = (-1,) + (1,) * w.dim()
            out.append(w[None] + a.view(sh) * u[None] + b.view(sh) * v[None])
        return out

    def forward(self, x, a, b, bn_train=False, gs=None, shared_input=True):
        """x: (N,3,32,32) shared by all P nets, or (P, N, 3, 32, 32) per-net images.
        P nets run as ONE grouped convolution (groups=P) on a (N, P*C, H, W) layout.
        (A linearity-trick variant, conv(h,w0)+a conv(h,dx)+b conv(h,dy), measured 10x slower
        on the GB10: it triples memory traffic, and this net is memory-bound.)
        BN train mode: statistics over each consecutive group of `gs` images (default N),
        per net and channel.  Returns logits (N, P, 10)."""
        P = a.shape[0]
        W = self.weights(a, b)
        if not shared_input:
            x = x.permute(1, 0, 2, 3, 4).reshape(x.shape[1], P * 3, *x.shape[3:])
        h = x
        for i in range(self.nl):
            w = W[i]
            cout = w.shape[1]
            w = w.reshape(P * cout, *w.shape[2:])
            groups = 1 if (i == 0 and shared_input) else P
            h = F.conv2d(h, w, padding=1, groups=groups)
            g_, b_, rm, rv = self.bn[i]
            if bn_train:
                N, C, H, Wd = h.shape
                k = gs or N
                hv = h.view(N // k, k, C, H, Wd)
                mu = hv.mean((1, 3, 4), keepdim=True)
                var = hv.var((1, 3, 4), keepdim=True, unbiased=False)
                h = ((hv - mu) * torch.rsqrt(var + EPS)).view(N, C, H, Wd)
                h = h * g_.repeat(P).view(1, -1, 1, 1) + b_.repeat(P).view(1, -1, 1, 1)
            else:
                scale = (g_ * torch.rsqrt(rv + EPS))
                h = h * scale.repeat(P).view(1, -1, 1, 1) + (b_ - rm * scale).repeat(P).view(1, -1, 1, 1)
            h = F.relu(h)
            if i < self.nl - 1:
                h = F.max_pool2d(h, 2)
        f = h.mean((2, 3)).view(h.shape[0], P, -1)        # (N, P, C)
        return torch.einsum("npc,pkc->npk", f, W[-1]) + self.fcb


def grid_axis(res, lo=-1.0, hi=1.0):
    return np.linspace(lo, hi, res)


def load_trained(device="cuda"):
    net = ConvNet4().to(device)
    sd = torch.load(os.path.join(CACHE, "convnet4.pt"), map_location=device)
    net.load_state_dict(sd["state_dict"]); net.eval()
    return net


def fixed_indices(n, seed=0, total=50000):
    return np.sort(np.random.default_rng(seed).permutation(total)[:n])
