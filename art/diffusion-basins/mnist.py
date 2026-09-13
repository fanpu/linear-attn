"""MNIST models: unconditional eps-prediction U-Net DDPM, a digit classifier, and a memorizing DDPM.

  python mnist.py clf                     # -> cache/mnist_clf.pt
  python mnist.py ddpm  --steps 30000     # -> cache/mnist_ddpm.pt     (full 60k train set)
  python mnist.py memo  --steps 30000     # -> cache/mnist_memo.pt     (40 images, 4 per digit)
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE, T_MIN, alpha_bar, ddim, gpu_setup

DATA = "/home/fzeng/ml/research/art/data"


def load_mnist(train=True, device="cpu"):
    import torchvision
    ds = torchvision.datasets.MNIST(DATA, train=train, download=False)
    x = ds.data.float().div(255).mul(2).sub(1).unsqueeze(1)  # [-1, 1]
    return x.to(device), ds.targets.to(device)


# ----------------------------------------------------------------------------- U-Net
class TimeEmb(nn.Module):
    def __init__(self, dim, nfreq=32):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(1000), nfreq)))
        self.mlp = nn.Sequential(nn.Linear(2 * nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c):  # c = log(sigma)/4, shape (B,)
        a = c[:, None] * self.freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class Res(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(te)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, 1, C, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.o(h.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    """28 -> 14 -> 7 with attention at 7x7."""

    def __init__(self, ch=(48, 96, 144)):
        super().__init__()
        c1, c2, c3 = ch
        td = 4 * c1
        self.te = TimeEmb(td)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1a, self.d1b = Res(c1, c1, td), Res(c1, c1, td)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2a, self.d2b = Res(c1, c2, td), Res(c2, c2, td)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.m1, self.ma, self.m2 = Res(c2, c3, td), Attn(c3), Res(c3, c3, td)
        self.u2a, self.u2b = Res(c3 + c2, c2, td), Res(c2 + c2, c2, td)
        self.u1a, self.u1b = Res(c2 + c1, c1, td), Res(c1 + c1, c1, td)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, c):
        te = self.te(c)
        h = self.inp(x)
        a = self.d1a(h, te); b = self.d1b(a, te)
        h = self.down1(b)
        c_ = self.d2a(h, te); d = self.d2b(c_, te)
        h = self.down2(d)
        h = self.m2(self.ma(self.m1(h, te)), te)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u2a(torch.cat([h, d], 1), te); h = self.u2b(torch.cat([h, c_], 1), te)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u1a(torch.cat([h, b], 1), te); h = self.u1b(torch.cat([h, a], 1), te)
        return self.out(h)


class PatchUNet(nn.Module):
    """U-Net on the 2x2 pixel-unshuffled image: 14x14 (c1) -> 7x7 (c2, attention) -> 14x14.
    4x fewer high-resolution FLOPs than a 28x28 U-Net; chosen because the GPU is shared by ~20 jobs."""

    def __init__(self, ch=(64, 128)):
        super().__init__()
        c1, c2 = ch
        td = 4 * c1
        self.te = TimeEmb(td)
        self.inp = nn.Conv2d(4, c1, 3, padding=1)
        self.d1a, self.d1b = Res(c1, c1, td), Res(c1, c1, td)
        self.down = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.m1, self.ma, self.m2 = Res(c1, c2, td), Attn(c2), Res(c2, c2, td)
        self.u1a, self.u1b = Res(c2 + c1, c1, td), Res(c1 + c1, c1, td)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 4, 3, padding=1))

    def forward(self, x, c):
        te = self.te(c)
        h = self.inp(F.pixel_unshuffle(x, 2))
        a = self.d1a(h, te); b = self.d1b(a, te)
        h = self.m2(self.ma(self.m1(self.down(b), te)), te)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.u1a(torch.cat([h, b], 1), te); h = self.u1b(torch.cat([h, a], 1), te)
        return F.pixel_shuffle(self.out(h), 2)


def unet_eps_fn(net, dtype=torch.float32, amp=False):
    """eps_fn(x_ve (B,784 or B,1,28,28), sigma) for the samplers in common.py"""
    def fn(x, sigma):
        shp = x.shape
        xv = (x / math.sqrt(1 + sigma ** 2)).reshape(-1, 1, 28, 28).to(dtype)
        c = torch.full((xv.shape[0],), math.log(sigma) / 4, device=x.device, dtype=dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            e = net(xv, c)
        return e.to(x.dtype).reshape(shp)
    return fn


def load_unet(kind, device, dtype=torch.float32):
    ck = torch.load(f"{CACHE}/mnist_{kind}.pt", map_location=device)
    net = PatchUNet(**ck["cfg"]).to(device)
    net.load_state_dict(ck["ema"])
    return net.to(dtype).eval(), ck


# ----------------------------------------------------------------------------- classifier
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 49, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, x):
        return self.f(x)


def load_clf(device):
    ck = torch.load(f"{CACHE}/mnist_clf.pt", map_location=device)
    m = Clf().to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


def train_clf(dev, epochs=4):
    x, y = load_mnist(True, dev)
    xt, yt = load_mnist(False, dev)
    torch.manual_seed(0)
    m = Clf().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    n, bs = len(x), 256
    steps = epochs * (n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        m.train()
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            xb = x[idx]
            # light augmentation: random +-2 px shifts
            dx, dy = np.random.randint(-2, 3, 2)
            xb = torch.roll(xb, (int(dx), int(dy)), (2, 3))
            loss = F.cross_entropy(m(xb), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); it += 1
        m.eval()
        with torch.no_grad():
            acc = torch.cat([m(xt[i:i + 2000]).argmax(1) == yt[i:i + 2000] for i in range(0, len(xt), 2000)]).float().mean()
        print(f"clf epoch {ep} loss {loss.item():.4f} test acc {acc.item():.4f}", flush=True)
    torch.save(dict(state=m.state_dict(), test_acc=acc.item()), f"{CACHE}/mnist_clf.pt")


# ----------------------------------------------------------------------------- DDPM training
def train_ddpm(dev, kind, steps, bs, lr, n_memo=40, seed=0, ch=(64, 128)):
    torch.manual_seed(seed)
    x, y = load_mnist(True, dev)
    if kind == "memo":
        g = torch.Generator().manual_seed(1)
        idx = []
        for d in range(10):
            cand = torch.nonzero(y.cpu() == d).flatten()
            idx += cand[torch.randperm(len(cand), generator=g)[: n_memo // 10]].tolist()
        idx = torch.tensor(idx)
        x, y = x[idx.to(dev)], y[idx.to(dev)]
        train_idx = idx
    else:
        train_idx = None
    torch.backends.cudnn.benchmark = True
    net = PatchUNet(ch).to(dev)
    ema = PatchUNet(ch).to(dev)
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[{kind}] params {sum(p.numel() for p in net.parameters())/1e6:.2f}M, data {len(x)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 1000))
    t0 = time.time()
    n = len(x)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        x0 = x[idx]
        t = T_MIN + (1 - T_MIN) * torch.rand(bs, device=dev)
        ab = alpha_bar(t)[:, None, None, None]
        e = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        c = torch.log(((1 - ab) / ab).sqrt()).flatten() / 4
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(xt, c)
        loss = F.mse_loss(pred.float(), e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        with torch.no_grad():
            dec = min(0.9995, (1 + it) / (10 + it))
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - dec)
        if it % 500 == 0 or it == steps - 1:
            torch.cuda.synchronize()
            print(f"[{kind}] it {it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if (it + 1) % 5000 == 0 or it == steps - 1:
            ck = dict(cfg=dict(ch=ch), ema=ema.state_dict(), steps=it + 1, bs=bs, lr=lr, seed=seed,
                      train_idx=train_idx, wall=time.time() - t0)
            if kind == "memo":
                ck["memo_eval"] = eval_memo(ema, x, dev)
                print(f"[memo] eval {ck['memo_eval']}", flush=True)
            torch.save(ck, f"{CACHE}/mnist_{kind}.pt")


@torch.no_grad()
def eval_memo(ema, xtrain, dev, n=512):
    ema.eval()
    g = torch.Generator(device=dev).manual_seed(5)
    z = torch.randn(n, 784, device=dev, generator=g)
    out = ddim(unet_eps_fn(ema, amp=True), z, 50)
    d = torch.cdist(out, xtrain.reshape(len(xtrain), -1))
    s, _ = d.sort(1)
    ratio = (s[:, 0] / s[:, 1])
    ema.train()
    return dict(frac_ratio_lt_1_3=float((ratio < 1 / 3).float().mean()), median_ratio=float(ratio.median()),
                median_l2_nn=float(s[:, 0].median()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4)
    a = ap.parse_args()
    dev = gpu_setup()
    if a.what == "clf":
        train_clf(dev)
    elif a.what == "clf+ddpm":
        train_clf(dev)
        train_ddpm(dev, "ddpm", a.steps, a.bs, a.lr)
    else:
        train_ddpm(dev, a.what, a.steps, a.bs, a.lr)
