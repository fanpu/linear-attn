"""Shared pieces for Combed (art/ml-art-3d.md §8): data, closed-form and trained flow fields,
RK4 sampler, and the Gu et al. memorisation criterion.

Conventions (Bertrand et al. arXiv:2506.03719, Prop. 1 / eq. 6): t = 0 is noise N(0, I), t = 1 is data,
path x_t = (1 - t) x0 + t x1.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
NS = (16, 64, 256, 1024, 4096)
DATA_SEED = 0
NOISE_SEED = 1
N_SAMPLES = 20_000
T_STOP = 1.0 - 1e-3
MEM_RATIO = 1.0 / 3.0  # Gu et al. (after Yoon et al. 2023): memorised if d1 < d2 / 3, l2 norm


# ----------------------------------------------------------------------------- data
def trefoil(s: np.ndarray) -> np.ndarray:
    """x(s) = (sin s + 2 sin 2s, cos s - 2 cos 2s, -sin 3s) / 3."""
    return np.stack([np.sin(s) + 2 * np.sin(2 * s), np.cos(s) - 2 * np.cos(2 * s), -np.sin(3 * s)], -1) / 3.0


def training_set(n: int, n_max: int = max(NS), seed: int = DATA_SEED) -> np.ndarray:
    """Nested: the first n of n_max points drawn with s ~ U[0, 2pi) from one seeded generator."""
    assert n <= n_max
    s = np.random.default_rng(seed).uniform(0.0, 2 * math.pi, n_max)
    return trefoil(s)[:n]


def noise_seeds(n: int = N_SAMPLES, seed: int = NOISE_SEED) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 3))


# ----------------------------------------------------------------------------- closed-form field
def closed_form_logits(x: torch.Tensor, t: float, data: torch.Tensor) -> torch.Tensor:
    """-||x - t x1_i||^2 / (2 (1 - t)^2), shape (B, N)."""
    d2 = (x * x).sum(1, keepdim=True) - 2 * t * (x @ data.T) + (t * t) * (data * data).sum(1)
    return -d2.clamp_min(0) / (2 * (1 - t) ** 2)


def closed_form_velocity(x: torch.Tensor, t: float, data: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """v*(x, t) = sum_i w_i (x1_i - x) / (1 - t), w = softmax(logits). Exact optimum of the CFM loss
    for the empirical data distribution (Bertrand et al. eq. 6)."""
    out = torch.empty_like(x)
    for a in range(0, x.shape[0], chunk):
        xb = x[a:a + chunk]
        w = torch.softmax(closed_form_logits(xb, t, data), 1)
        out[a:a + chunk] = (w @ data - xb) / (1 - t)
    return out


def mixture_density(x: np.ndarray, t: float, data: np.ndarray) -> np.ndarray:
    """p_t(x) = (1/N) sum_i N(x; t x1_i, (1 - t)^2 I) in R^3 (float64, direct form for tests)."""
    s2 = (1 - t) ** 2
    d2 = ((x[:, None, :] - t * data[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * s2)).mean(1) / (2 * math.pi * s2) ** 1.5


# ----------------------------------------------------------------------------- trained field
class TimeEmbed(torch.nn.Module):
    """Sinusoidal features of t: sin/cos at 16 frequencies geometric in [1, 1000]."""

    def __init__(self, n_freq: int = 16, f_max: float = 1000.0):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, math.log(f_max), n_freq)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * self.freqs[None]
        return torch.cat([a.sin(), a.cos()], 1)


class VelocityMLP(torch.nn.Module):
    """[x, emb(t)] -> 4 hidden layers of width 256, SiLU -> v in R^3."""

    def __init__(self, width: int = 256, depth: int = 4, n_freq: int = 16):
        super().__init__()
        self.emb = TimeEmbed(n_freq)
        layers, d_in = [], 3 + 2 * n_freq
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, width), torch.nn.SiLU()]
            d_in = width
        layers.append(torch.nn.Linear(d_in, 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.emb(t)], 1))


def mlp_velocity(model: VelocityMLP, x: torch.Tensor, t: float) -> torch.Tensor:
    with torch.no_grad():
        return model(x, torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device))


# ----------------------------------------------------------------------------- sampler
def time_grid(n_steps: int, t_stop: float = T_STOP) -> np.ndarray:
    return np.linspace(0.0, t_stop, n_steps + 1)


def keep_indices(n_steps: int, n_keep: int = 64) -> np.ndarray:
    idx = np.unique(np.round(np.linspace(0, n_steps, n_keep)).astype(int))
    assert len(idx) == n_keep
    return idx


def rk4(field, x0: torch.Tensor, n_steps: int, t_stop: float = T_STOP, keep: np.ndarray | None = None):
    """Classical RK4 on the uniform grid t_k = k t_stop / n_steps. field(x, t) -> v.
    Returns (x at t_stop, stored states (len(keep), B, 3) or None)."""
    ts = time_grid(n_steps, t_stop)
    x = x0.clone()
    stored = []
    keep_set = set(keep.tolist()) if keep is not None else set()
    if 0 in keep_set:
        stored.append(x.clone())
    for k in range(n_steps):
        t, h = float(ts[k]), float(ts[k + 1] - ts[k])
        k1 = field(x, t)
        k2 = field(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = field(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = field(x + h * k3, t + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k + 1 in keep_set:
            stored.append(x.clone())
    return x, (torch.stack(stored) if keep is not None else None)


def rk4_grid(field, x0: torch.Tensor, ts: np.ndarray) -> torch.Tensor:
    """Classical RK4 on an arbitrary increasing time grid ts (endpoints included)."""
    x = x0.clone()
    for k in range(len(ts) - 1):
        t, h = float(ts[k]), float(ts[k + 1] - ts[k])
        k1 = field(x, t)
        k2 = field(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = field(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = field(x + h * k3, t + h)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def tail_grid(n_steps: int, eps_from: float = 1e-3, eps_to: float = 1e-6) -> np.ndarray:
    """t grid from 1 - eps_from to 1 - eps_to, geometric in (1 - t): constant h / (1 - t)."""
    return 1.0 - np.geomspace(eps_from, eps_to, n_steps + 1)


# ----------------------------------------------------------------------------- memorisation
def nn_two(points: np.ndarray, data: np.ndarray):
    """Indices and l2 distances of the nearest and second-nearest training points."""
    from scipy.spatial import cKDTree

    d, i = cKDTree(data).query(points, k=2)
    return i[:, 0], d[:, 0], d[:, 1]


def memorised(points: np.ndarray, data: np.ndarray, ratio: float = MEM_RATIO) -> np.ndarray:
    _, d1, d2 = nn_two(points, data)
    return d1 < ratio * d2
