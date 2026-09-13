"""Core library: common-random-number (CRN) deep random nets over a 2D
hyperparameter grid, mean-field theory by quadrature, and box counting.

Conventions (match D'Inverno et al. 2025, code github.com/jon-dong/fractal-deep-info-prop):
    z^l = sigma_w * W^l h^{l-1} / sqrt(N) + sigma_b * b^l,   h^l = phi(z^l)
    W^l_ij ~ N(0,1), b^l_i ~ N(0,1), resampled independently per layer.
CRN: W^l and b^l are generated from a seed that depends only on (seed, N, l),
so every pixel of every map, every zoom and every resolution sees the *same*
standard-normal draws, just scaled by that pixel's sigma_w and sigma_b.
"""
import math
import numpy as np
import torch

ACT = {"erf": torch.erf, "tanh": torch.tanh}


def layer_draw(seed, N, l, device="cuda", dtype=torch.float64):
    """Standard-normal (W, b) for layer l (1-indexed); always drawn in float64 on CPU
    so float32 and float64 runs share bit-identical underlying draws."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(N) * 7919 + int(l))
    W = torch.randn(N, N, generator=g, dtype=torch.float64)
    b = torch.randn(N, generator=g, dtype=torch.float64)
    return W.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def inputs_draw(seed, N, n=2, device="cuda", dtype=torch.float64):
    """n independent Gaussian inputs normalised so that |h0|^2 / N = 1
    (paper: unit-norm x, with activations carried as h/sqrt(N))."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(N) * 7919 - 1)
    X = torch.randn(n, N, generator=g, dtype=torch.float64)
    X = X / X.norm(dim=1, keepdim=True) * math.sqrt(N)
    return X.to(device=device, dtype=dtype)


class LayerCache:
    """Cache layer draws on device for reuse across chunks."""

    def __init__(self, seed, N, D, device="cuda", dtype=torch.float64):
        self.seed, self.N, self.D, self.device, self.dtype = seed, N, D, device, dtype
        self.store = None
        mem = D * N * N * (8 if dtype == torch.float64 else 4)
        if mem < 1.5e9:
            self.store = [layer_draw(seed, N, l, device, dtype) for l in range(1, D + 1)]

    def __call__(self, l):
        if self.store is not None:
            return self.store[l - 1]
        return layer_draw(self.seed, self.N, l, self.device, self.dtype)


_STEP = {}


def _step_fn(act):
    if act not in _STEP:
        phi = ACT[act]

        def step(h, Wt, b, w2, bb2):
            return phi(torch.mm(h, Wt) * w2 + bb2 * b)

        _STEP[act] = torch.compile(step, dynamic=False)
    return _STEP[act]


@torch.no_grad()
def frontier_grid(sw, sb, N, D, seed=0, act="erf", dtype=torch.float64, chunk=65536,
                  n_avg=20, record_layers=(), device="cuda", layers=None, log=None,
                  input_perturb=0.0):
    """Pixelwise L^l = |x1^l - x2^l|^2 with x = h / sqrt(N) (the paper's convention).

    sw, sb: 1D float64 arrays (length P) of sigma_w, sigma_b per pixel.
    Returns dict with L_D (last layer), L_avg (mean over last n_avg layers, as in the
    paper's text) and L_rec (len(record_layers), P).
    input_perturb: relative Gaussian perturbation of the inputs (roundoff proxy)."""
    step = _step_fn(act)
    P = len(sw)
    layers = layers or LayerCache(seed, N, D, device, dtype)
    X0 = inputs_draw(seed, N, 2, device, dtype)
    if input_perturb:
        g = torch.Generator(device="cpu").manual_seed(12345)
        X0 = X0 + input_perturb * torch.randn(X0.shape, generator=g, dtype=torch.float64).to(device, dtype)
    L_D = np.zeros(P)
    L_avg = np.zeros(P)
    rec_idx = {l: i for i, l in enumerate(record_layers)}
    L_rec = np.zeros((len(record_layers), P), dtype=np.float64)
    inv = 1.0 / math.sqrt(N)
    chunk = min(chunk, P)
    Wts = {}
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        n = e - s
        w = np.zeros(chunk); w[:n] = sw[s:e]
        bbv = np.zeros(chunk); bbv[:n] = sb[s:e]
        w = torch.as_tensor(w, device=device, dtype=dtype)[:, None]
        bbv = torch.as_tensor(bbv, device=device, dtype=dtype)[:, None]
        w2 = torch.cat([w, w]); bb2 = torch.cat([bbv, bbv])
        h = torch.cat([X0[0].expand(chunk, N), X0[1].expand(chunk, N)]).contiguous()
        acc = torch.zeros(chunk, device=device, dtype=torch.float64)
        for l in range(1, D + 1):
            W, b = layers(l)
            if layers.store is not None:
                if l not in Wts:
                    Wts[l] = (W.T * inv).contiguous()
                Wt = Wts[l]
            else:
                Wt = (W.T * inv).contiguous()
            h = step(h, Wt, b, w2, bb2)
            if l > D - n_avg or l in rec_idx:
                d = ((h[:chunk] - h[chunk:]) ** 2).sum(1).double() / N
                if l > D - n_avg:
                    acc += d
                if l in rec_idx:
                    L_rec[rec_idx[l], s:e] = d[:n].cpu().numpy()
        L_D[s:e] = d[:n].cpu().numpy()
        L_avg[s:e] = (acc / n_avg)[:n].cpu().numpy()
        if log:
            log(f"  chunk {e}/{P}")
    return dict(L_D=L_D, L_avg=L_avg, L_rec=L_rec)


def grid_axes(x0, x1, y0, y1, res):
    """Pixel-centre coordinates. Image rows = y (sigma_b), cols = x (sigma_w)."""
    rx = res if np.isscalar(res) else res[0]
    ry = res if np.isscalar(res) else res[1]
    xs = x0 + (np.arange(rx) + 0.5) * (x1 - x0) / rx
    ys = y0 + (np.arange(ry) + 0.5) * (y1 - y0) / ry
    return xs, ys


# ----------------------------------------------------------------- mean field
def gh(n):
    """Probabilists' Gauss-Hermite nodes/weights for E_{z~N(0,1)}."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


def erf_cov(q11, q22, q12):
    """E[erf(u)erf(v)] for (u,v) ~ N(0, [[q11,q12],[q12,q22]]) (closed form)."""
    return (2 / math.pi) * np.arcsin(np.clip(2 * q12 / np.sqrt((1 + 2 * q11) * (1 + 2 * q22)), -1, 1))


def meanfield_erf_L(sw, sb, D, c0=0.0, n_avg=20):
    """Infinite-width prediction of L^l = |x1^l - x2^l|^2 (post-activation, x = h/sqrt(N))
    for erf, via the exact closed-form covariance recursion. Inputs have |h0|^2/N = 1 and
    overlap c0 (independent Gaussian inputs: c0 ~ 0)."""
    sw2, sb2 = np.asarray(sw, dtype=np.float64) ** 2, np.asarray(sb, dtype=np.float64) ** 2
    q = sw2 * 1.0 + sb2          # pre-activation variance at layer 1
    q12 = sw2 * c0 + sb2         # pre-activation covariance at layer 1
    acc = np.zeros_like(q)
    for l in range(1, D + 1):
        Ephi2 = erf_cov(q, q, q)
        Ephiphi = erf_cov(q, q, q12)
        L = 2 * (Ephi2 - Ephiphi)
        if l > D - n_avg:
            acc += L
        q, q12 = sw2 * Ephi2 + sb2, sw2 * Ephiphi + sb2
    return L, acc / n_avg
