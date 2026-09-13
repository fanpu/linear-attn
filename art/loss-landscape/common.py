"""Shared pieces: CIFAR-10 on GPU, the Li et al. CIFAR ResNets (with / without
shortcuts), filter-normalized random directions, and a fast loss evaluator.

The architectures and the direction recipe follow github.com/tomgoldstein/loss-landscape
(cifar10/models/resnet.py and net_plotter.py):
  * ResNet_cifar: 16-32-64 channels, 3 stages of BasicBlocks, 1x1-conv+BN projection
    shortcut on shape change; the `noshort` variant simply drops the shortcut.
  * direction = one N(0,1) tensor per entry of net.parameters(); for every tensor with
    dim <= 1 (BN gamma/beta, the linear bias) the direction is set to zero ('biasbn');
    for dim >= 2 each filter d[i] is rescaled to ||w[i]|| (the linear layer's rows
    count as filters). BN running statistics are untouched (dir_type='weights').
  * loss is evaluated with net.eval() (BN in eval mode).
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
STD = (0.2023, 0.1994, 0.2010)  # the values used in Li et al.'s dataloader


# ----------------------------------------------------------------------------- data
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
    x = (x - m) / s
    y = torch.tensor(ys, device=device, dtype=torch.long)
    return x, y


def fixed_subset(n, seed=0):
    """Indices of the fixed training subset used for every landscape evaluation."""
    g = np.random.default_rng(seed)
    return np.sort(g.permutation(50000)[:n])


# ----------------------------------------------------------------------------- models
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, shortcut=True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.use_short = shortcut
        self.shortcut = nn.Sequential()
        if shortcut and (stride != 1 or cin != cout):
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_short:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, depth=20, shortcut=True, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for stage, c in enumerate([16, 32, 64]):
            blocks = []
            for b in range(n):
                s = 2 if (b == 0 and stage > 0) else 1
                blocks.append(BasicBlock(cin, c, s, shortcut)); cin = c
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = layers
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.avg_pool2d(out, 8).flatten(1)
        return self.linear(out)


def model_name(depth, shortcut):
    return f"resnet{depth}" + ("" if shortcut else "_noshort")


def parse_name(name):
    shortcut = not name.endswith("_noshort")
    depth = int(name.replace("resnet", "").replace("_noshort", ""))
    return depth, shortcut


def load_model(name, epoch=None, device="cuda"):
    depth, shortcut = parse_name(name)
    net = ResNetCifar(depth, shortcut).to(device)
    tag = "final" if epoch is None else f"ep{epoch:03d}"
    sd = torch.load(os.path.join(CACHE, "ckpt", f"{name}_{tag}.pt"), map_location=device)
    net.load_state_dict(sd["state_dict"])
    net.eval()
    return net


# ----------------------------------------------------------------------------- directions
def random_direction(params, seed, device="cuda", dtype=torch.float32):
    """Li et al. 'weights' direction with filter normalization and ignore='biasbn'.
    Raw Gaussians are drawn on CPU from a fixed seed (so the same raw direction is
    reproducible and shared across checkpoints of the same architecture)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for w in params:
        d = torch.randn(w.shape, generator=g, dtype=torch.float64)
        wd = w.detach().to("cpu", torch.float64)
        if d.dim() <= 1:
            d.zero_()
        else:
            dn = d.flatten(1).norm(dim=1)
            wn = wd.flatten(1).norm(dim=1)
            d.mul_((wn / (dn + 1e-10)).view(-1, *([1] * (d.dim() - 1))))
        out.append(d.to(device=device, dtype=dtype))
    return out


def get_directions(net, seed_x=1, seed_y=2, dtype=torch.float32):
    params = [p for p in net.parameters()]
    return random_direction(params, seed_x, dtype=dtype), random_direction(params, seed_y, dtype=dtype)


# ----------------------------------------------------------------------------- evaluation
class LossEvaluator:
    """Evaluates mean cross-entropy (and accuracy) of net at w0 + a*dx + b*dy on a fixed
    set of examples, BN in eval mode. TF32 is disabled so conv arithmetic is true fp32
    (or fp64 when dtype=torch.float64)."""

    def __init__(self, net, x, y, dx, dy, batch=2500, dtype=torch.float32):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.net = net.to(dtype).eval()
        self.params = [p for p in self.net.parameters()]
        for p in self.params:
            p.requires_grad_(False)
        self.w0 = [p.detach().clone() for p in self.params]
        self.dx = [d.to(dtype) for d in dx]
        self.dy = [d.to(dtype) for d in dy]
        self.x, self.y = x.to(dtype), y
        self.batch = batch
        self.dtype = dtype

    @torch.no_grad()
    def set_point(self, a, b):
        for p, w, u, v in zip(self.params, self.w0, self.dx, self.dy):
            p.copy_(w).add_(u, alpha=float(a)).add_(v, alpha=float(b))

    @torch.no_grad()
    def __call__(self, a, b):
        self.set_point(a, b)
        tot = torch.zeros((), device=self.x.device, dtype=torch.float64)
        cor = torch.zeros((), device=self.x.device, dtype=torch.long)
        for i in range(0, self.x.shape[0], self.batch):
            out = self.net(self.x[i:i + self.batch])
            tot += F.cross_entropy(out.to(torch.float64), self.y[i:i + self.batch], reduction="sum")
            cor += (out.argmax(1) == self.y[i:i + self.batch]).sum()
        n = self.x.shape[0]
        return (tot / n).item(), (cor.double() / n).item()
