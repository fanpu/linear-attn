"""2D toy for the hero: a two-layer ReLU net whose hidden neurons are drawn as particles.

    f(x) = alpha/m * sum_j a_j relu(w_j . x + b_j),     loss = 1/alpha^2 * 1/(2n) sum (f - y)^2
Mean-field scaling (Chizat & Bach 2018; Mei, Montanari & Nguyen 2018) with Chizat-Oyallon-Bach output scale alpha.
Symmetric ("doubling") init: neurons come in pairs with equal (w, b) and opposite a, so f = 0 at init.
Gradient descent uses step size lr * m per particle (the mean-field time scale), so for every alpha the function
moves at the same speed at the start; only how far the *particles* must travel changes (by 1/alpha).
"""
import numpy as np
import torch


def rings(n, seed=0, noise=0.04):
    """Three concentric annuli: inner disk +1, middle ring -1, outer ring +1 (radii scaled to ~[-1, 1])."""
    g = np.random.default_rng(seed)
    r = np.sqrt(g.uniform(0, 1, n)) * 1.0
    th = g.uniform(0, 2 * np.pi, n)
    y = np.where(r < 0.36, 1.0, np.where(r < 0.7, -1.0, 1.0))
    # push points away from the two class boundaries so the classes are cleanly separable
    keep = (np.abs(r - 0.36) > 0.05) & (np.abs(r - 0.7) > 0.05)
    x = np.stack([r * np.cos(th), r * np.sin(th)], 1) + noise * g.standard_normal((n, 2))
    return x[keep], y[keep]


def spiral(n, seed=0, turns=1.6, noise=0.03):
    g = np.random.default_rng(seed)
    t = np.sqrt(g.uniform(0.02, 1, n))
    lab = g.integers(0, 2, n)
    ang = 2 * np.pi * turns * t + np.pi * lab
    x = np.stack([t * np.cos(ang), t * np.sin(ang)], 1) + noise * g.standard_normal((n, 2))
    return x, np.where(lab == 1, 1.0, -1.0)


def init(m, seed=0, dtype=torch.float64, radius=1.2):
    """Kink lines {x : w.x + b = 0} spread uniformly over the data disk: random direction, |w| = 1 (then
    jittered), offset uniform in [-radius, radius]. Output weights |N(0,1)| with a +/- twin for each neuron."""
    g = torch.Generator().manual_seed(seed)
    h = m // 2
    th = torch.rand(h, generator=g, dtype=dtype) * 2 * torch.pi
    s = 1 + 0.2 * torch.randn(h, generator=g, dtype=dtype)
    w = torch.stack([th.cos(), th.sin()], 1) * s[:, None]
    b = -(torch.rand(h, generator=g, dtype=dtype) * 2 - 1) * radius * s
    a = torch.randn(h, generator=g, dtype=dtype).abs()
    return dict(w=torch.cat([w, w]), b=torch.cat([b, b]), a=torch.cat([a, -a]))


def forward(p, x, alpha):
    m = p["a"].shape[0]
    return alpha / m * torch.relu(x @ p["w"].T + p["b"]) @ p["a"]


def train(x, y, m=400, alpha=1.0, lr=1.0, steps=4000, frames=None, seed=0, grid=None, device="cpu",
          dtype=torch.float64):
    """Returns per-frame particles (w, b, a), losses and grid predictions."""
    X, Y = torch.tensor(x, device=device, dtype=dtype), torch.tensor(y, device=device, dtype=dtype)
    p = {k: v.to(device) for k, v in init(m, seed, dtype).items()}
    frames = set(frames if frames is not None else [])
    rec = dict(step=[], w=[], b=[], a=[], loss=[], grid=[], acc=[])
    G = torch.tensor(grid, device=device, dtype=dtype) if grid is not None else None
    for t in range(steps + 1):
        if t in frames or t == steps:
            with torch.no_grad():
                f = forward(p, X, alpha)
                rec["step"].append(t); rec["loss"].append(0.5 * ((f - Y) ** 2).mean().item())
                rec["acc"].append(((f > 0) == (Y > 0)).double().mean().item())
                for k in "wba":
                    rec[k].append(p[k].cpu().numpy().copy())
                if G is not None:
                    rec["grid"].append(forward(p, G, alpha).cpu().numpy())
        if t == steps:
            break
        for k in p:
            p[k].requires_grad_(True)
        loss = 0.5 * ((forward(p, X, alpha) - Y) ** 2).mean() / alpha ** 2
        gr = torch.autograd.grad(loss, [p[k] for k in "wba"])
        with torch.no_grad():
            p = {k: p[k] - lr * m * g for k, g in zip("wba", gr)}
    return {k: np.array(v) for k, v in rec.items()}
