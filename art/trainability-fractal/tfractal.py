"""Batched torch reimplementation of Sohl-Dickstein (2024) "The boundary of neural
network trainability is fractal" (github.com/Sohl-Dickstein/fractal, JAX colab).

Every pixel of a 2D hyperparameter grid is an independent tiny network, stored as
a batched tensor (P, 16, 16) + (P, 16, 1) and trained with manual forward/backward
passes (matmul/bmm), all in float64.

Setup mirrored from the colab (width=16, depth=2, dataset_param_multiple=1,
target_dim=1, readout='loss'):
  net:  h = phi(X W0 / sqrt(n)),  phi(z)=tanh(sqrt2 z)  or sqrt2*relu(z)
        y_hat = h W1 / n                           (mean-field parameterization)
  no biases; W0 ~ N(0,1)^{16x16}, W1 ~ N(0,1)^{16x1}
  data: X ~ N(0,1)^{272x16}, Y ~ N(0,1)^{272x1}  (272 = #params)
  loss: mean squared error; full-batch steepest GD, per-layer learning rates
  convergence measure (his `convergence_measure`):
        l_t -> 1e6 if non-finite; v_t = l_t / l_0; v_t = min(v_t, 1e6)
        converged  = mean(v_{T-20:T}) < 1
        value      = -sum_t v_t      if converged   (negative)
                   = +sum_t 1/v_t   if diverged    (positive)
The init and data are shared across all pixels (fixed seed), exactly as in his code.
Only the RNG differs (torch vs JAX), so the exact picture differs but the object is
the same family.
"""
import math
import time

import numpy as np
import torch

DT = torch.float64
N_HID = 16
MAX_VAL = 1e6
LAST = 20


def make_problem(seed=0, width=N_HID, nonlin='tanh', device='cuda', data_mult=1.0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    W0 = torch.randn(width, width, generator=g, dtype=DT)
    W1 = torch.randn(width, 1, generator=g, dtype=DT)
    n_params = width * width + width
    if nonlin == 'quadratic':
        n_data = n_params
    else:
        n_data = int(n_params * data_mult)
    X = torch.randn(n_data, width, generator=g, dtype=DT)
    Y = torch.randn(n_data, 1, generator=g, dtype=DT)
    return dict(W0=W0.to(device), W1=W1.to(device), X=X.to(device), Y=Y.to(device),
                width=width, nonlin=nonlin, seed=seed)


def _forward(W0, W1, X, nonlin, n):
    """W0 (P,n,n) W1 (P,n,1) X (N,n) -> Z0 (P,N,n), h (P,N,n), out (P,N,1)"""
    Z0 = torch.matmul(X, W0)
    Z0.mul_(1.0 / math.sqrt(n))
    if nonlin == 'tanh':
        h = torch.tanh(Z0 * math.sqrt(2.0))
    elif nonlin == 'relu':
        h = torch.relu(Z0) * math.sqrt(2.0)
    elif nonlin == 'identity':
        h = Z0
    else:
        raise ValueError(nonlin)
    out = torch.matmul(h, W1)
    out.mul_(1.0 / n)
    return Z0, h, out


def _loss_and_grad(W0, W1, X, Y, nonlin, n):
    Z0, h, out = _forward(W0, W1, X, nonlin, n)
    r = out.sub_(Y)                       # (P,N,1)
    Nd = X.shape[0]
    loss = (r * r).mean(dim=(1, 2))       # (P,)
    r.mul_(2.0 / Nd / n)                  # dL/d(h W1)
    gW1 = torch.matmul(h.transpose(1, 2), r)          # (P,n,1)
    dh = torch.matmul(r, W1.transpose(1, 2))          # (P,N,n)
    if nonlin == 'tanh':
        # h = tanh(sqrt2 Z0): dZ0 = dh * sqrt2 * (1-h^2)
        dh.mul_(1.0 - h * h).mul_(math.sqrt(2.0))
    elif nonlin == 'relu':
        dh.mul_((Z0 > 0).to(DT)).mul_(math.sqrt(2.0))
    dh.mul_(1.0 / math.sqrt(n))
    gW0 = torch.matmul(X.t(), dh)                     # (P,n,n)
    return loss, gW0, gW1


def _full_loss(W0, W1, X, Y, nonlin, n):
    _, _, out = _forward(W0, W1, X, nonlin, n)
    r = out.sub_(Y)
    return (r * r).mean(dim=(1, 2))


S2 = math.sqrt(2.0)


def _make_step(nonlin, n):
    """Forward/backward in an (N, P*n) activation layout so both big matmuls are single
    2D GEMMs (X @ V, X^T @ dZ) with no broadcast copies; tanh and its derivative are
    applied in place. Bitwise-equivalent math to the naive bmm version (checked to
    4e-16 relative)."""
    a0 = 1.0 / math.sqrt(n)

    def lossgrad(W0, W1, X, Y):
        N = X.shape[0]
        P = W0.shape[0]
        V = W0.permute(1, 0, 2).reshape(n, P * n)            # V[d, p*n+e] = W0[p,d,e]
        Z = torch.mm(X, V).view(N, P, n)
        Z.mul_(a0)
        w1 = W1.view(1, P, n)
        if nonlin == 'tanh':
            h = torch.tanh(Z * S2)
        elif nonlin == 'relu':
            h = torch.relu(Z) * S2
        elif nonlin == 'sin':
            h = torch.sin(Z * S2)
        elif nonlin == 'identity':
            h = Z
        else:
            raise ValueError(nonlin)
        out = (h * w1).sum(-1) / n                             # (N,P)
        r = out - Y.view(N, 1)
        loss = (r * r).mean(0)
        r = r * (2.0 / N / n)
        gW1 = torch.einsum('npe,np->pe', h, r).view(P, n, 1)
        if nonlin == 'tanh':
            dphi = S2 * (1.0 - h * h)
        elif nonlin == 'relu':
            dphi = S2 * (Z > 0).to(Z.dtype)
        elif nonlin == 'sin':
            dphi = S2 * torch.cos(Z * S2)
        else:
            dphi = torch.ones_like(Z)
        dZ = dphi * (r.unsqueeze(-1) * w1) * a0
        gW0 = torch.mm(X.t(), dZ.reshape(N, P * n)).view(n, P, n).permute(1, 0, 2)
        return loss, gW0, gW1

    def full_loss(W0, W1, X, Y):
        N = X.shape[0]
        P = W0.shape[0]
        V = W0.permute(1, 0, 2).reshape(n, P * n)
        Z = torch.mm(X, V).view(N, P, n) * a0
        if nonlin == 'tanh':
            h = torch.tanh(Z * S2)
        elif nonlin == 'relu':
            h = torch.relu(Z) * S2
        elif nonlin == 'sin':
            h = torch.sin(Z * S2)
        else:
            h = Z
        r = (h * W1.view(1, P, n)).sum(-1) / n - Y.view(N, 1)
        return (r * r).mean(0)

    def step_fb(W0, W1, lr0, lr1, wd, X, Y):
        loss, gW0, gW1 = lossgrad(W0, W1, X, Y)
        W0n = W0 - lr0 * (gW0 + wd * W0)
        W1n = W1 - lr1 * (gW1 + wd * W1)
        return loss, W0n, W1n

    def step_mb(W0, W1, lr0, lr1, wd, X, Y, Xb, Yb):
        loss = full_loss(W0, W1, X, Y)
        _, gW0, gW1 = lossgrad(W0, W1, Xb, Yb)
        W0n = W0 - lr0 * (gW0 + wd * W0)
        W1n = W1 - lr1 * (gW1 + wd * W1)
        return loss, W0n, W1n

    return lossgrad, step_fb, step_mb


_COMPILED = {}


def get_steps(nonlin, n, compiled=True):
    key = (nonlin, n, compiled)
    if key not in _COMPILED:
        lg, fb, mb = _make_step(nonlin, n)
        if compiled:
            fb = torch.compile(fb, dynamic=True)
            mb = torch.compile(mb, dynamic=True)
        _COMPILED[key] = (lg, fb, mb)
    return _COMPILED[key]


def train_chunk(prob, lr0, lr1, steps=500, sigma0=None, sigma1=None, wd=None,
                minibatch=None, early_exit=True, check_every=25, exit_loss=1e100,
                checkpoints=None, compiled=True):
    """Train P independent nets (one per pixel), all float64.
    lr0/lr1: (P,) learning rates of input/output layer. sigma0/sigma1: optional (P,)
    init-scale multipliers. wd: optional (P,) L2 coefficient added to the gradient
    (tracked loss stays plain MSE). checkpoints: optional sorted list of step counts
    T_k <= steps at which his convergence measure is also evaluated (as if training had
    stopped at T_k); returned as 'measure_T' (K,P).

    Early exit: once a pixel's loss is non-finite or > exit_loss (1e100 x typical l0),
    its clamped v_t is frozen at the value it has at exit for all remaining steps (the
    remaining 1/v terms are <=1e-6 each). Verified at 128^2: 0 sign flips vs exact,
    max relative change of the measure 1.4e-5."""
    dev = lr0.device
    n = prob['width']
    X, Y = prob['X'], prob['Y']
    _, step_fb, step_mb = get_steps(prob['nonlin'], n, compiled)
    P0 = lr0.shape[0]
    P = P0
    W0 = prob['W0'].expand(P, n, n).clone()
    W1 = prob['W1'].expand(P, n, 1).clone()
    if sigma0 is not None:
        W0.mul_(sigma0.view(P, 1, 1))
    if sigma1 is not None:
        W1.mul_(sigma1.view(P, 1, 1))
    cps = sorted(set(list(checkpoints or []) + [steps]))
    K = len(cps)
    idx = torch.arange(P, device=dev)
    # final outputs (full size)
    outS = torch.zeros(K, P0, dtype=DT, device=dev)
    outSinv = torch.zeros(K, P0, dtype=DT, device=dev)
    outLast = torch.zeros(K, P0, dtype=DT, device=dev)
    # alive accumulators
    aS = torch.zeros(P, dtype=DT, device=dev)
    aSinv = torch.zeros(P, dtype=DT, device=dev)
    aWin = torch.zeros(P, dtype=DT, device=dev)     # sum of v over current last-20 window
    a_lr0 = lr0.view(P, 1, 1).clone()
    a_lr1 = lr1.view(P, 1, 1).clone()
    a_wd = (wd.view(P, 1, 1).clone() if wd is not None
            else torch.zeros(P, 1, 1, dtype=DT, device=dev))
    a_v0 = None
    gmb = torch.Generator(device='cpu').manual_seed(42) if minibatch else None
    ck = 0
    for t in range(steps):
        if minibatch:
            mb = torch.randint(0, X.shape[0], (minibatch,), generator=gmb).to(dev)
            loss, W0, W1 = step_mb(W0, W1, a_lr0, a_lr1, a_wd, X, Y, X[mb], Y[mb])
        else:
            loss, W0, W1 = step_fb(W0, W1, a_lr0, a_lr1, a_wd, X, Y)
        l = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, MAX_VAL))
        if t == 0:
            a_v0 = l.clone()
        v = torch.clamp(l / a_v0, max=MAX_VAL)
        aS.add_(v)
        aSinv.add_(1.0 / v)
        T = cps[ck]
        if t >= T - LAST:
            aWin.add_(v)
        if t == T - 1:
            outS[ck, idx] = aS; outSinv[ck, idx] = aSinv; outLast[ck, idx] = aWin
            aWin.zero_()
            ck += 1
            if ck == K:
                break
        if early_exit and (t + 1) % check_every == 0:
            dead = (~torch.isfinite(loss)) | (loss > exit_loss)
            nd = int(dead.sum())
            if nd > 0 and nd > 0.02 * P:
                di = idx[dead]
                vd = v[dead]
                for j in range(ck, K):
                    rem = cps[j] - (t + 1)
                    outS[j, di] = aS[dead] + rem * vd
                    outSinv[j, di] = aSinv[dead] + rem / vd
                    # window sum: part already accumulated (if window started) + frozen
                    if j == ck:
                        started = max(0, (t + 1) - (cps[j] - LAST))
                        outLast[j, di] = aWin[dead] + (LAST - started) * vd
                    else:
                        outLast[j, di] = LAST * vd
                keep = ~dead
                idx = idx[keep]; W0 = W0[keep]; W1 = W1[keep]
                a_lr0 = a_lr0[keep]; a_lr1 = a_lr1[keep]; a_wd = a_wd[keep]
                aS = aS[keep]; aSinv = aSinv[keep]; aWin = aWin[keep]; a_v0 = a_v0[keep]
                P = idx.numel()
                if P == 0:
                    break
    converged = (outLast / LAST) < 1
    measure = torch.where(converged, -outS, outSinv)
    res = dict(measure=measure[-1])
    if checkpoints:
        res['measure_T'] = measure
    return res


# ---------------------------------------------------------------------------
# Null model: a genuinely quadratic loss with the same two-learning-rate structure.
# y_hat = (X a)/sqrt(n) + (H b)/n with H = phi(X W0/sqrt n) frozen at init.
# a (n,) gets eta0, b (n,) gets eta1. Loss is quadratic in (a,b): GD is linear, the
# stability boundary is the smooth algebraic curve rho(I - P Hess) = 1.
# ---------------------------------------------------------------------------
def train_chunk_quadratic(prob, lr0, lr1, steps=500, **kw):
    dev = lr0.device
    n = prob['width']
    X, Y = prob['X'], prob['Y']
    with torch.no_grad():
        H = torch.tanh(X @ prob['W0'] * (math.sqrt(2.0) / math.sqrt(n)))
    F0 = X / math.sqrt(n)             # (N,n)
    F1 = H / n                        # (N,n)
    P = lr0.shape[0]
    a = torch.zeros(P, n, 1, dtype=DT, device=dev)
    b = prob['W1'].expand(P, n, 1).clone()
    a.add_(prob['W0'][0].view(1, n, 1))       # nonzero init for both blocks
    Nd = X.shape[0]
    S = torch.zeros(P, dtype=DT, device=dev); Sinv = torch.zeros_like(S); Slast = torch.zeros_like(S)
    l0 = None
    for t in range(steps):
        r = torch.matmul(F0, a) + torch.matmul(F1, b) - Y        # (P,N,1)
        loss = (r * r).mean(dim=(1, 2))
        l = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, MAX_VAL))
        if t == 0:
            l0 = l.clone()
        v = torch.clamp(l / l0, max=MAX_VAL)
        S.add_(v); Sinv.add_(1.0 / v)
        if t >= steps - LAST:
            Slast.add_(v)
        r.mul_(2.0 / Nd)
        a.sub_(torch.matmul(F0.t(), r) * lr0.view(P, 1, 1))
        b.sub_(torch.matmul(F1.t(), r) * lr1.view(P, 1, 1))
    converged = (Slast / LAST) < 1
    return dict(measure=torch.where(converged, -S, Sinv))


# ---------------------------------------------------------------------------
# grid driver
# ---------------------------------------------------------------------------
def log_grid(c0, c1, half_w, res, dev='cuda'):
    """Pixel-centre grid in log10 space, centred at (c0,c1) (log10 of the x-axis and
    y-axis hyperparameters), half-width half_w decades (scalar, or (hw_x, hw_y)).
    Returns hx (res*res,), hy (res*res,): hx varies along columns, hy along rows
    (row 0 = bottom). The value is formed as 10**c * 10**offset, which keeps full
    float64 relative resolution around the centre at deep zoom."""
    hwx, hwy = (half_w, half_w) if np.isscalar(half_w) else half_w
    off = (torch.arange(res, dtype=DT) + 0.5) / res * 2.0 - 1.0
    ten = torch.tensor(10.0, dtype=DT)
    ex = (10.0 ** torch.tensor(c0, dtype=DT)) * torch.pow(ten, off * hwx)
    ey = (10.0 ** torch.tensor(c1, dtype=DT)) * torch.pow(ten, off * hwy)
    EY, EX = torch.meshgrid(ey, ex, indexing='ij')
    return EX.reshape(-1).to(dev), EY.reshape(-1).to(dev)


def run_grid(prob, h0, h1, steps=500, chunk=32768, trainer=None, verbose=True,
             checkpoints=None, **kw):
    """h0,h1: flat (P,) hyperparameter tensors (eta0, eta1). Extra (P,) tensors in kw
    (sigma0, sigma1, wd) are chunked alongside. Returns numpy float64 measure (P,), or
    (measure, measure_T (K,P) float32) if checkpoints is given."""
    trainer = trainer or {'quadratic': train_chunk_quadratic, 'liu': train_chunk_liu}.get(prob['nonlin'], train_chunk)
    P = h0.numel()
    out = np.empty(P, dtype=np.float64)
    outT = None
    t0 = time.time()
    extras = {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)}
    tens = {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)}
    if checkpoints:
        extras['checkpoints'] = checkpoints
    nch = math.ceil(P / chunk)
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        sub = {k: v[s:e] for k, v in tens.items()}
        with torch.no_grad():
            r = trainer(prob, h0[s:e], h1[s:e], steps=steps, **sub, **extras)
        out[s:e] = r['measure'].cpu().numpy()
        if checkpoints:
            mt = r['measure_T'].to(torch.float32).cpu().numpy()
            if outT is None:
                outT = np.empty((mt.shape[0], P), dtype=np.float32)
            outT[:, s:e] = mt
        if verbose and ((s // chunk) % 4 == 0 or e == P):
            el = time.time() - t0
            print(f'  chunk {s//chunk+1}/{nch}  {el:.0f}s  ({e/el:.0f} px/s)', flush=True)
    if checkpoints:
        return out, outT
    return out


def setup_gpu(frac=0.10):
    torch.cuda.set_per_process_memory_fraction(frac)
    torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------
# Liu (2024)-style trivially non-convex 2-parameter loss, same measure (see liu_toy.py)
# ---------------------------------------------------------------------------
def train_chunk_liu(prob, lr0, lr1, steps=500, eps=0.05, lam=0.2, rho=0.3, **kw):
    a = torch.ones_like(lr0); b = torch.ones_like(lr0)
    k = 2 * math.pi / lam
    S = torch.zeros_like(a); Sinv = torch.zeros_like(a); Slast = torch.zeros_like(a)
    l0 = None
    for t in range(steps):
        u = a - b
        L = a * a + 2 * rho * a * b + b * b + eps * (1 + torch.cos(k * u))
        L = torch.where(torch.isfinite(L), L, torch.full_like(L, MAX_VAL))
        if t == 0:
            l0 = L.clone()
        v = torch.clamp(L / l0, max=MAX_VAL)
        S += v; Sinv += 1 / v
        if t >= steps - LAST:
            Slast += v
        s = -eps * k * torch.sin(k * u)
        ga = 2 * a + 2 * rho * b + s
        gb = 2 * b + 2 * rho * a - s
        a = a - lr0 * ga
        b = b - lr1 * gb
    return dict(measure=torch.where(Slast / LAST < 1, -S, Sinv))
