"""2-D Gaussian-mixture toy: mixture definitions, analytic score, MLP eps-net, training.

  python toy.py train            # trains ring8, grid25, scatter12 nets -> cache/toy_<name>.pt
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common import CACHE, T_MIN, alpha_bar, sigma_of_t, gpu_setup


def mixture(name):
    """returns means (k,2), weights (k,), component std s"""
    if name == "ring8":
        k = 8
        ang = 2 * math.pi * np.arange(k) / k
        mu = 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1)
        w = np.ones(k) / k
        s = 0.10
    elif name == "grid25":
        g = np.arange(-2, 3, dtype=float)
        mu = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        w = np.ones(25) / 25
        s = 0.08
    elif name == "scatter12":
        rng = np.random.default_rng(7)
        pts = []
        while len(pts) < 12:  # Poisson-disc-ish rejection so modes stay separated
            p = rng.uniform(-2.2, 2.2, 2)
            if all(np.linalg.norm(p - q) > 0.9 for q in pts):
                pts.append(p)
        mu = np.array(pts)
        w = rng.uniform(0.5, 1.5, 12)
        w /= w.sum()
        s = 0.09
    else:
        raise ValueError(name)
    return mu, w, s


class GMM:
    """exact noised mixture; everything in the VE variable x = x_t / sqrt(ab)"""

    def __init__(self, name, device, dtype=torch.float64):
        mu, w, s = mixture(name)
        self.name, self.s = name, s
        self.mu = torch.tensor(mu, device=device, dtype=dtype)
        self.logw = torch.tensor(np.log(w), device=device, dtype=dtype)
        self.k = len(w)

    def resp(self, x, sigma):
        v = self.s ** 2 + sigma ** 2
        d2 = ((x[:, None, :] - self.mu[None]) ** 2).sum(-1)
        return torch.softmax(self.logw[None] - 0.5 * d2 / v, dim=1)

    def denoise(self, x, sigma):
        """Tweedie E[x0 | x_sigma = x]"""
        v = self.s ** 2 + sigma ** 2
        r = self.resp(x, sigma)
        m = r @ self.mu
        return m + (self.s ** 2 / v) * (x - m)

    def score(self, x, sigma):
        return (self.denoise(x, sigma) - x) / sigma ** 2

    def eps(self, x, sigma):
        """eps prediction in VE units: x = x0 + sigma eps  =>  eps = (x - D) / sigma"""
        return (x - self.denoise(x, sigma)) / sigma

    def sample(self, n, gen):
        dev = gen.device
        idx = torch.multinomial(self.logw.exp().float().to(dev), n, replacement=True, generator=gen)
        eps = torch.randn(n, 2, generator=gen, dtype=self.mu.dtype, device=dev)
        return self.mu[idx.to(self.mu.device)] + self.s * eps.to(self.mu.device)

    def label(self, x0):
        d2 = ((x0[:, None, :] - self.mu[None]) ** 2).sum(-1)
        return d2.argmin(1)


class EpsMLP(nn.Module):
    """eps-prediction MLP. Input: VP-scaled x_t and Fourier features of c = log(sigma)/4."""

    def __init__(self, width=512, depth=4, nfreq=16):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(nfreq) * 0.5)
        layers, d_in = [], 2 + 2 * nfreq + 1
        for i in range(depth):
            layers += [nn.Linear(d_in if i == 0 else width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_vp, c):
        c = c.expand(x_vp.shape[0], 1) if c.dim() < 2 or c.shape[0] == 1 else c
        ang = c * self.freqs[None]
        h = torch.cat([x_vp, c, torch.sin(ang), torch.cos(ang)], 1)
        return self.net(h)


def learned_eps_fn(net):
    def fn(x, sigma):
        dt = next(net.parameters()).dtype
        x_vp = x / math.sqrt(1 + sigma ** 2)
        c = torch.full((1, 1), math.log(sigma) / 4, device=x.device, dtype=dt)
        return net(x_vp.to(dt), c).to(x.dtype)
    return fn


def analytic_eps_fn(gmm):
    return lambda x, sigma: gmm.eps(x, sigma)


def load_net(name, device, dtype=torch.float64):
    ck = torch.load(f"{CACHE}/toy_{name}.pt", map_location=device)
    net = EpsMLP(**ck["cfg"]).to(device)
    net.load_state_dict(ck["ema"])
    return net.to(dtype).eval()


def train(name, device, steps=40000, bs=8192, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    gmm = GMM(name, device, torch.float32)
    cfg = dict(width=512, depth=4, nfreq=16)
    net = EpsMLP(**cfg).to(device)
    ema = EpsMLP(**cfg).to(device)
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    log = []
    for it in range(steps):
        x0 = gmm.sample(bs, gen)
        t = T_MIN + (1 - T_MIN) * torch.rand(bs, 1, device=device, generator=gen)
        ab = alpha_bar(t)
        e = torch.randn(x0.shape, device=device, generator=gen)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * e
        c = torch.log(((1 - ab) / ab).sqrt()) / 4
        loss = ((net(xt, c) - e) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            dec = min(0.999, (1 + it) / (10 + it))
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - dec)
        if it % 2000 == 0 or it == steps - 1:
            torch.cuda.synchronize()
            print(f"[{name}] it {it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    # excess loss over the Bayes-optimal (analytic) eps, measured on fresh data, float64
    ema = ema.double().eval()
    g64 = GMM(name, device, torch.float64)
    gen2 = torch.Generator(device=device).manual_seed(123)
    rows = []
    with torch.no_grad():
        for tt in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
            n = 200000
            x0 = g64.sample(n, gen2)
            ab = alpha_bar(tt)
            sig = math.sqrt((1 - ab) / ab)
            e = torch.randn(n, 2, generator=gen2, dtype=torch.float64, device=device)
            xve = x0 + sig * e
            l_net = ((learned_eps_fn(ema)(xve, sig) - e) ** 2).sum(1).mean().item()
            l_opt = ((g64.eps(xve, sig) - e) ** 2).sum(1).mean().item()
            gap = ((learned_eps_fn(ema)(xve, sig) - g64.eps(xve, sig)) ** 2).sum(1).mean().item()
            rows.append((tt, sig, l_net, l_opt, gap))
            print(f"[{name}] t={tt:<6} sigma={sig:8.4f} loss_net={l_net:.5f} loss_bayes={l_opt:.5f} "
                  f"|eps_net-eps*|^2={gap:.2e}", flush=True)
    torch.save(dict(cfg=cfg, ema=ema.float().state_dict(), eval=rows, steps=steps, bs=bs, lr=lr,
                    seed=seed, wall=time.time() - t0), f"{CACHE}/toy_{name}.pt")


if __name__ == "__main__":
    dev = gpu_setup()
    if sys.argv[1] == "train":
        names = sys.argv[2:] or ["ring8", "grid25", "scatter12"]
        for nm in names:
            train(nm, dev)
