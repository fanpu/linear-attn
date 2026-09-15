"""Batched trainability engine for Solid Edge (3D hyperparameter volumes).

Extends art/trainability-fractal/tfractal.py (imported read-only, never modified):
  net2  - the source network exactly (tfractal._make_step lossgrad reused), with an
          optional init scale sigma multiplying both weight matrices at init.
  net3  - two hidden layers, width 16, tanh(sqrt2 z), three learning rates:
          h0 = phi(X W0 / sqrt n), h1 = phi(h0 W1 / sqrt n), y_hat = h1 W2 / n.
          Same data recipe as the source: X, Y ~ N(0,1), N = #params (= 528).
  quad2 - the source's own quadratic null (tfractal.train_chunk_quadratic): y = F0 a + F1 b,
          F0 = X/sqrt n, F1 = tanh(sqrt2 X W0/sqrt n)/n frozen, init a = W0[0], b = W1; the
          optional sigma scales both inits (it cannot move a linear-GD stability boundary).
  quad3 - null model: the exactly quadratic loss of y_hat = F0 a + F1 b + F2 c with
          frozen features F0 = X/sqrt n, F1 = h0/n, F2 = h1/n taken from net3's init,
          one learning rate per block. GD is linear, so its stability boundary is
          the smooth algebraic surface rho(I - P Hess) = 1.

Convergence measure: his `convergence_measure` as in tfractal.train_chunk
(v_t = min(l_t/l_0, 1e6), converged iff mean(v over last 20) < 1, value -sum v if
converged else +sum 1/v), with ONE declared change (2026-09-15, M2): a non-finite loss gives
v_t = 1e6 instead of min(1e6/l_0, 1e6). They coincide when l_0 <= 1 and give the same label when
l_0 <= 1e6; for l_0 > 1e6 (init scale sigma >~ 10^2.5) his version labels overflowed runs converged.

Early exit follows tfractal.train_chunk (freeze a row's v once its loss is non-finite
or > 1e100, but only when > 2 % of live rows die at a check). The difference is
**bucketed compaction**: the batch is only shrunk to a power-of-two bucket
(>= bucket_min), padded with already-frozen rows whose outputs are never rewritten.
So a run only ever uses shapes from {chunk, chunk/2, ..., bucket_min}, which keeps the
shared unified-memory pool from fragmenting. Per-row arithmetic is unchanged.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/home/fzeng/ml/research/art/trainability-fractal')
import tfractal as tf  # noqa: E402  (source engine, read-only)

MAX_VAL = tf.MAX_VAL
LAST = tf.LAST
S2 = math.sqrt(2.0)
N_HID = 16


# --------------------------------------------------------------------------- problems
def make_problem(kind, seed=0, width=N_HID, device='cuda', dtype=torch.float32):
    """All draws in float64 from a CPU torch Generator, then cast (as tfractal does)."""
    if kind == 'net2':
        old = tf.DT
        tf.DT = dtype
        try:
            p = tf.make_problem(seed, width=width, nonlin='tanh', device=device)
        finally:
            tf.DT = old
        return dict(kind=kind, width=width, Ws=[p['W0'], p['W1']], X=p['X'], Y=p['Y'], data=(p['X'], p['Y']))
    if kind == 'quad2':
        p = make_problem('net2', seed, width, device, torch.float64)
        X, W0 = p['X'], p['Ws'][0]
        H = torch.tanh(X @ W0 * (S2 / math.sqrt(width)))
        c = lambda t: t.to(device, dtype)
        return dict(kind=kind, width=width, Ws=[c(W0[0].reshape(width, 1)), c(p['Ws'][1])], X=c(X), Y=c(p['Y']),
                    data=(c(X / math.sqrt(width)), c(H / width), c(p['Y'])))
    n = width
    g = torch.Generator(device='cpu').manual_seed(seed)
    W0 = torch.randn(n, n, generator=g, dtype=torch.float64)
    W1 = torch.randn(n, n, generator=g, dtype=torch.float64)
    W2 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    n_params = 2 * n * n + n
    X = torch.randn(n_params, n, generator=g, dtype=torch.float64)
    Y = torch.randn(n_params, 1, generator=g, dtype=torch.float64)
    if kind == 'net3':
        c = lambda t: t.to(device, dtype)
        return dict(kind=kind, width=n, Ws=[c(W0), c(W1), c(W2)], X=c(X), Y=c(Y), data=(c(X), c(Y)))
    if kind == 'quad3':
        a0 = 1.0 / math.sqrt(n)
        h0 = torch.tanh(X @ W0 * (S2 * a0))
        h1 = torch.tanh(h0 @ W1 * (S2 * a0))
        F0, F1, F2 = X * a0, h0 / n, h1 / n
        c = lambda t: t.to(device, dtype)
        Ws = [c(W0[0].reshape(n, 1)), c(W1[0].reshape(n, 1)), c(W2.reshape(n, 1))]
        return dict(kind=kind, width=n, Ws=Ws, X=c(X), Y=c(Y), data=(c(F0), c(F1), c(F2), c(Y)))
    raise ValueError(kind)


# --------------------------------------------------------------------------- loss + grad
def _lossgrad_net3(n):
    a0 = 1.0 / math.sqrt(n)

    def lossgrad(Ws, data):
        W0, W1, W2 = Ws
        X, Y = data
        N = X.shape[0]
        P = W0.shape[0]
        V = W0.permute(1, 0, 2).reshape(n, P * n)
        h0 = torch.tanh(torch.mm(X, V).view(N, P, n).transpose(0, 1) * (a0 * S2))   # (P,N,n)
        h1 = torch.tanh(torch.matmul(h0, W1) * (a0 * S2))                          # (P,N,n)
        out = torch.matmul(h1, W2).squeeze(-1) / n                                  # (P,N)
        r = out - Y.view(1, N)
        loss = (r * r).mean(1)
        r = r * (2.0 / N / n)
        gW2 = torch.matmul(h1.transpose(1, 2), r.unsqueeze(-1))                     # (P,n,1)
        dZ1 = (r.unsqueeze(-1) * W2.view(P, 1, n)) * (S2 * a0) * (1.0 - h1 * h1)   # (P,N,n)
        gW1 = torch.matmul(h0.transpose(1, 2), dZ1)                                 # (P,n,n)
        dZ0 = torch.matmul(dZ1, W1.transpose(1, 2)) * (S2 * a0) * (1.0 - h0 * h0)  # (P,N,n)
        gW0 = torch.mm(X.t(), dZ0.transpose(0, 1).reshape(N, P * n)).view(n, P, n).permute(1, 0, 2)
        return loss, [gW0, gW1, gW2]

    return lossgrad


def _lossgrad_net2(n):
    lg, _, _ = tf._make_step('tanh', n)   # the source's hand-written fwd/bwd, unchanged

    def lossgrad(Ws, data):
        loss, gW0, gW1 = lg(Ws[0], Ws[1], data[0], data[1])
        return loss, [gW0, gW1]

    return lossgrad


def _lossgrad_quad3(n):
    def lossgrad(Ws, data):
        Fs, Y = data[:-1], data[-1]
        N = Y.shape[0]
        r = -Y
        for F, W in zip(Fs, Ws):
            r = r + torch.matmul(F, W)                                              # (P,N,1)
        loss = (r * r).mean(dim=(1, 2))
        r = r * (2.0 / N)
        return loss, [torch.matmul(F.t(), r) for F in Fs]

    return lossgrad


_STEPS = {}


def get_step(kind, n, compiled=True):
    key = (kind, n, compiled)
    if key not in _STEPS:
        lg = {'net2': _lossgrad_net2, 'net3': _lossgrad_net3, 'quad3': _lossgrad_quad3, 'quad2': _lossgrad_quad3}[kind](n)

        def step(Ws, lrs, data):
            loss, gs = lg(Ws, data)
            return loss, [W - lr * g for W, lr, g in zip(Ws, lrs, gs)]

        _STEPS[key] = (lg, torch.compile(step, dynamic=True) if compiled else step)
    return _STEPS[key]


# --------------------------------------------------------------------------- trainer
def _bucket(nl, bucket_min):
    return max(bucket_min, 1 << max(0, (nl - 1).bit_length()))


def train_chunk(prob, lrs, sigma=None, steps=500, early_exit=True, check_every=25,
                exit_loss=1e100, bucket_min=1024, compiled=True):
    """lrs: list of (P,) learning-rate tensors, one per parameter block.
    sigma: optional (P,) init scale multiplying every weight block at init.
    Returns dict(measure=(P,) tensor, shapes=set of batch sizes used)."""
    n = prob['width']
    dev = lrs[0].device
    dt = prob['X'].dtype
    _, step = get_step(prob['kind'], n, compiled)
    data = prob['data']
    P0 = lrs[0].shape[0]
    Ws = [W.expand(P0, *W.shape).clone() for W in prob['Ws']]
    if sigma is not None:
        for W in Ws:
            W.mul_(sigma.to(dt).view(P0, 1, 1))
    a_lr = [lr.to(dt).view(P0, 1, 1).clone() for lr in lrs]
    idx = torch.arange(P0, device=dev)
    live = torch.ones(P0, dtype=torch.bool, device=dev)
    oS = torch.zeros(P0, dtype=dt, device=dev)
    oSi = torch.zeros_like(oS)
    oL = torch.zeros_like(oS)
    aS = torch.zeros(P0, dtype=dt, device=dev)
    aSi = torch.zeros_like(aS)
    aW = torch.zeros_like(aS)
    v0 = None
    shapes = {P0}
    for t in range(steps):
        loss, Ws = step(Ws, a_lr, data)
        fin = torch.isfinite(loss)
        l = torch.where(fin, loss, torch.full_like(loss, MAX_VAL))
        if t == 0:
            v0 = l.clone()
        # Non-finite loss -> v = MAX_VAL (the clamp ceiling). His measure sets l = 1e6 *before*
        # normalising, which scores an overflowed run as converged whenever l0 > 1e6 (large sigma;
        # float32 overflows at 3e38 long before the 1e100 early exit). Labels are unchanged when l0 <= 1e6.
        v = torch.where(fin, torch.clamp(l / v0, max=MAX_VAL), torch.full_like(l, MAX_VAL))
        aS.add_(v)
        aSi.add_(1.0 / v)
        if t >= steps - LAST:
            aW.add_(v)
        if early_exit and (t + 1) % check_every == 0 and t + 1 < steps:
            dead = ((~torch.isfinite(loss)) | (loss > exit_loss)) & live
            nd = int(dead.sum())
            nlive = int(live.sum())
            if nd > 0 and nd > 0.02 * nlive:
                rem = steps - (t + 1)
                started = max(0, (t + 1) - (steps - LAST))
                di = idx[dead]
                vd = v[dead]
                oS[di] = aS[dead] + rem * vd
                oSi[di] = aSi[dead] + rem / vd
                oL[di] = aW[dead] + (LAST - started) * vd
                live = live & ~dead
                nl = nlive - nd
                if nl == 0:
                    break
                B = _bucket(nl, bucket_min)
                if B < live.numel():
                    li = torch.nonzero(live).squeeze(1)
                    pad = torch.nonzero(~live).squeeze(1)[:B - nl]
                    sel = torch.cat([li, pad])
                    idx = idx[sel]; live = live[sel]
                    Ws = [W[sel] for W in Ws]; a_lr = [x[sel] for x in a_lr]
                    aS = aS[sel]; aSi = aSi[sel]; aW = aW[sel]; v0 = v0[sel]
                    shapes.add(B)
    if bool(live.any()):
        li = idx[live]
        oS[li] = aS[live]; oSi[li] = aSi[live]; oL[li] = aW[live]
    conv = (oL / LAST) < 1
    return dict(measure=torch.where(conv, -oS, oSi), shapes=shapes)


# --------------------------------------------------------------------------- grids
BASE_RES = 1024          # the source overview window: c = (1.5, 1.5), hw = 4.5 decades
BASE_C, BASE_HW = 1.5, 4.5


def overview_lr_axis(res, device='cpu'):
    """Learning-rate values at every (1024/res)-th pixel centre of the 1024^2 overview,
    pixel index i = stride*j + stride//2, computed exactly as tfractal.log_grid does
    (float64, 10**c * 10**(off*hw)). Returns (values float64 tensor, pixel indices)."""
    stride = BASE_RES // res
    pix = torch.arange(res) * stride + stride // 2
    off = (pix.to(torch.float64) + 0.5) / BASE_RES * 2.0 - 1.0
    ten = torch.tensor(10.0, dtype=torch.float64)
    vals = (10.0 ** torch.tensor(BASE_C, dtype=torch.float64)) * torch.pow(ten, off * BASE_HW)
    return vals.to(device), pix.numpy()


def sigma_axis(res, device='cpu'):
    """log10 sigma_k = (k - k1) * (6/res) with k1 = round(26 res / 64): 6 decades like the
    source sigma x eta plate ([-2.44, 3.47] at res 64), sigma = 1 exactly on plane k1."""
    k1 = int(round(26 * res / 64))
    d = 6.0 / res
    lg = (torch.arange(res, dtype=torch.float64) - k1) * d
    return torch.pow(torch.tensor(10.0, dtype=torch.float64), lg).to(device), lg.numpy(), k1
