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


NMAX = 1024


def layer_draw(seed, N, l, device="cuda", dtype=torch.float64):
    """Standard-normal (W, b) for layer l (1-indexed); always drawn in float64 on CPU
    so float32 and float64 runs share bit-identical underlying draws.
    Width-nested CRN: one NMAX x NMAX master draw per (seed, layer); the width-N network uses
    its top-left N x N block (and first N bias entries), so widths share draws too."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(l))
    W = torch.randn(NMAX, NMAX, generator=g, dtype=torch.float64)[:N, :N].contiguous()
    b = torch.randn(NMAX, generator=g, dtype=torch.float64)[:N].contiguous()
    return W.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


def inputs_draw(seed, N, n=2, device="cuda", dtype=torch.float64):
    """n independent Gaussian inputs normalised so that |h0|^2 / N = 1
    (paper: unit-norm x, with activations carried as h/sqrt(N))."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 - 1)
    X = torch.randn(n, NMAX, generator=g, dtype=torch.float64)[:, :N]
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
                  input_perturb=0.0, tau_hit=None, hit_every=5):
    """Pixelwise L^l = |x1^l - x2^l|^2 with x = h / sqrt(N) (the paper's convention).

    sw, sb: 1D float64 arrays (length P) of sigma_w, sigma_b per pixel.
    Returns dict with L_D (last layer), L_avg (mean over last n_avg layers, as in the
    paper's text) and L_rec (len(record_layers), P).
    input_perturb: relative Gaussian perturbation of the inputs (roundoff proxy).
    tau_hit: if set, also return t_hit = first checked layer (every hit_every) with L < tau_hit
             (D+1 if never): the ordered-side 'convergence depth'."""
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
    t_hit = np.full(P, D + 1, dtype=np.int32)
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
        hit = torch.full((chunk,), D + 1, device=device, dtype=torch.int32)
        for l in range(1, D + 1):
            W, b = layers(l)
            if layers.store is not None:
                if l not in Wts:
                    Wts[l] = (W.T * inv).contiguous()
                Wt = Wts[l]
            else:
                Wt = (W.T * inv).contiguous()
            h = step(h, Wt, b, w2, bb2)
            if tau_hit is not None and l % hit_every == 0:
                dd = ((h[:chunk] - h[chunk:]) ** 2).sum(1) / N
                hit = torch.where((hit > D) & (dd < tau_hit), torch.full_like(hit, l), hit)
            if l > D - n_avg or l in rec_idx:
                d = ((h[:chunk] - h[chunk:]) ** 2).sum(1).double() / N
                if l > D - n_avg:
                    acc += d
                if l in rec_idx:
                    L_rec[rec_idx[l], s:e] = d[:n].cpu().numpy()
        L_D[s:e] = d[:n].cpu().numpy()
        L_avg[s:e] = (acc / n_avg)[:n].cpu().numpy()
        t_hit[s:e] = hit[:n].cpu().numpy()
        if log:
            log(f"  chunk {e}/{P}")
    return dict(L_D=L_D, L_avg=L_avg, L_rec=L_rec, t_hit=t_hit)


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


# ----------------------------------------------------------------- geometry
def boundary_mask(B):
    """Boundary pixels of a binary image: a 2x2 cell containing both classes marks all its
    pixels (same rule as the paper's extract_edges on a signed field), output same shape."""
    B = B.astype(bool)
    m = np.zeros_like(B)
    c = (B[1:, 1:] != B[:-1, 1:]) | (B[1:, 1:] != B[1:, :-1]) | (B[1:, 1:] != B[:-1, :-1])
    m[1:, 1:] |= c; m[:-1, 1:] |= c; m[1:, :-1] |= c; m[:-1, :-1] |= c
    return m


def edge_cells(B):
    """Paper's extract_edges: (H-1, W-1) mask of 2x2 cells containing both classes."""
    B = B.astype(bool)
    Y = np.stack((B[1:, 1:], B[:-1, 1:], B[1:, :-1], B[:-1, :-1]), -1)
    return Y.any(-1) & ~Y.all(-1)


def box_counts(E, sizes):
    """Number of occupied s x s boxes of a boolean image for each s (trailing partial boxes kept)."""
    out = []
    for s in sizes:
        H, W = E.shape
        ph, pw = (-H) % s, (-W) % s
        Ep = np.pad(E, ((0, ph), (0, pw)))
        out.append(int(Ep.reshape(Ep.shape[0] // s, s, Ep.shape[1] // s, s).any((1, 3)).sum()))
    return np.array(out)


def pick_zoom_center(B, frac, margin=0.25):
    """Choose the sub-window (side = frac * window) with the strongest ordered/chaotic mixing:
    score = (#boundary pixels) * 4 p (1-p), p = chaotic fraction, restricted away from edges.
    Returns centre in fractional image coords (cx, cy) in [0,1]."""
    H, W = B.shape
    k = max(2, int(round(frac * W)))
    E = boundary_mask(B).astype(np.float64)
    Bf = B.astype(np.float64)
    from scipy.ndimage import uniform_filter
    eb = uniform_filter(E, k, mode="constant")
    pb = uniform_filter(Bf, k, mode="constant")
    score = eb * 4 * pb * (1 - pb)
    lo_y, hi_y = int(margin * H), int((1 - margin) * H)
    lo_x, hi_x = int(margin * W), int((1 - margin) * W)
    sub = score[lo_y:hi_y, lo_x:hi_x]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return (ix + lo_x + 0.5) / W, (iy + lo_y + 0.5) / H


_LSTEP = {}


def _lyap_step(act):
    if act not in _LSTEP:
        phi = ACT[act]
        dphi = {"erf": lambda z: (2 / math.sqrt(math.pi)) * torch.exp(-z * z),
                "tanh": lambda z: 1 - torch.tanh(z) ** 2}[act]

        def step(h, v, Wt, b, w, bb):
            z = torch.mm(h, Wt) * w + bb * b
            v = dphi(z) * (torch.mm(v, Wt) * w)
            n = v.norm(dim=1, keepdim=True)
            return phi(z), v / n, n.squeeze(1)

        _LSTEP[act] = torch.compile(step, dynamic=False)
    return _LSTEP[act]


@torch.no_grad()
def lyapunov_grid(sw, sb, N, D, seed=0, act="erf", dtype=torch.float32, chunk=65536, burn=200,
                  record_layers=(), device="cuda", layers=None):
    """Finite-time maximal Lyapunov exponent of the input-to-layer map along the trajectory of input A:
    lambda = mean_{burn < l <= D} log |J^l v| with v renormalised each layer (tangent propagation).
    lambda < 0: nearby inputs merge (order); lambda > 0: they separate (chaos).
    Also returns lambda over (burn, l] for each l in record_layers."""
    step = _lyap_step(act)
    P = len(sw)
    layers = layers or LayerCache(seed, N, D, device, dtype)
    X0 = inputs_draw(seed, N, 2, device, dtype)
    g = torch.Generator(device="cpu").manual_seed(4242)
    v0 = torch.randn(N, generator=g, dtype=torch.float64).to(device, dtype)
    v0 = v0 / v0.norm()
    lam = np.zeros(P)
    rec = np.zeros((len(record_layers), P))
    rix = {l: i for i, l in enumerate(record_layers)}
    inv = 1.0 / math.sqrt(N)
    chunk = min(chunk, P)
    Wts = {}
    for s in range(0, P, chunk):
        e = min(P, s + chunk); n = e - s
        w = np.zeros(chunk); w[:n] = sw[s:e]
        bbv = np.zeros(chunk); bbv[:n] = sb[s:e]
        w = torch.as_tensor(w, device=device, dtype=dtype)[:, None]
        bbv = torch.as_tensor(bbv, device=device, dtype=dtype)[:, None]
        h = X0[0].expand(chunk, N).contiguous()
        v = v0.expand(chunk, N).contiguous()
        acc = torch.zeros(chunk, device=device, dtype=torch.float64)
        for l in range(1, D + 1):
            W, b = layers(l)
            if l not in Wts:
                Wts[l] = (W.T * inv).contiguous()
            h, v, nrm = step(h, v, Wts[l], b, w, bbv)
            if l > burn:
                acc += torch.log(nrm.double().clamp_min(1e-300))
            if l in rix:
                rec[rix[l], s:e] = (acc / max(l - burn, 1))[:n].cpu().numpy()
        lam[s:e] = (acc / (D - burn))[:n].cpu().numpy()
    return lam, rec
