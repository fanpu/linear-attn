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


def train_chunk(prob, lr0, lr1, steps=500, sigma0=None, sigma1=None, wd=None,
                minibatch=None, early_exit=True, check_every=25, exit_loss=1e100,
                return_traj=False):
    """Train P independent nets. lr0/lr1: (P,) float64 tensors on device.
    sigma0/sigma1: optional (P,) init-scale multipliers. wd: optional (P,) L2 coefficient
    (added to the gradient as wd*W; the tracked loss stays the plain MSE).
    Returns measure (P,) float64 (his convergence measure) and extras."""
    dev = lr0.device
    n = prob['width']
    nonlin = prob['nonlin']
    X, Y = prob['X'], prob['Y']
    P = lr0.shape[0]
    W0 = prob['W0'].expand(P, n, n).clone()
    W1 = prob['W1'].expand(P, n, 1).clone()
    if sigma0 is not None:
        W0.mul_(sigma0.view(P, 1, 1))
    if sigma1 is not None:
        W1.mul_(sigma1.view(P, 1, 1))
    idx = torch.arange(P, device=dev)            # alive -> original index
    S = torch.zeros(P, dtype=DT, device=dev)
    Sinv = torch.zeros(P, dtype=DT, device=dev)
    Slast = torch.zeros(P, dtype=DT, device=dev)
    v0 = None
    a_lr0 = lr0.view(P, 1, 1).clone()
    a_lr1 = lr1.view(P, 1, 1).clone()
    a_wd = wd.view(P, 1, 1).clone() if wd is not None else None
    # alive-view accumulators (compacted)
    aS = S.clone(); aSinv = Sinv.clone(); aSlast = Slast.clone()
    if minibatch is not None:
        gmb = torch.Generator(device='cpu').manual_seed(42)
    traj = [] if return_traj else None
    dead_S = None
    for t in range(steps):
        if minibatch is None:
            loss, gW0, gW1 = _loss_and_grad(W0, W1, X, Y, nonlin, n)
        else:
            # same minibatch indices for every pixel at a given step (as in his vmap)
            mb = torch.randint(0, X.shape[0], (minibatch,), generator=gmb).to(dev)
            loss = _full_loss(W0, W1, X, Y, nonlin, n)
            _, gW0, gW1 = _loss_and_grad(W0, W1, X[mb], Y[mb], nonlin, n)
        l = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, MAX_VAL))
        if t == 0:
            v0 = l.clone()
            a_v0 = v0.clone()
        v = torch.clamp(l / a_v0, max=MAX_VAL)
        aS.add_(v)
        aSinv.add_(1.0 / v)
        if t >= steps - LAST:
            aSlast.add_(v)
        if return_traj:
            full = torch.full((P,), float('nan'), dtype=DT, device=dev)
            full[idx] = v
            traj.append(full.cpu())
        if a_wd is not None:
            gW0.add_(W0 * a_wd)
            gW1.add_(W1 * a_wd)
        W0.sub_(gW0.mul_(a_lr0))
        W1.sub_(gW1.mul_(a_lr1))
        if early_exit and (t + 1) % check_every == 0 and t + 1 < steps - LAST:
            # a pixel whose loss is non-finite, or astronomically above l0 (>1e100),
            # contributes exactly v=min(1e6/l0,1e6) (or 1e6) for every remaining step.
            dead = (~torch.isfinite(loss)) | (loss > exit_loss)
            nd = int(dead.sum())
            if nd > 0 and nd > 0.05 * dead.numel():
                rem = steps - (t + 1)
                vd = torch.where(torch.isfinite(loss[dead]),
                                 torch.full_like(loss[dead], MAX_VAL),
                                 torch.clamp(MAX_VAL / a_v0[dead], max=MAX_VAL))
                # after exit the loss stays huge/non-finite: non-finite -> 1e6/l0 clamp,
                # finite >1e100 -> clamp 1e6. Use the non-finite branch value if the
                # params overflow later; both are ~1e-6 per step in 1/v and >1 in v.
                S[idx[dead]] = aS[dead] + rem * vd
                Sinv[idx[dead]] = aSinv[dead] + rem / vd
                Slast[idx[dead]] = LAST * vd
                keep = ~dead
                idx = idx[keep]; W0 = W0[keep]; W1 = W1[keep]
                a_lr0 = a_lr0[keep]; a_lr1 = a_lr1[keep]
                if a_wd is not None:
                    a_wd = a_wd[keep]
                aS = aS[keep]; aSinv = aSinv[keep]; aSlast = aSlast[keep]; a_v0 = a_v0[keep]
                P = idx.numel()
                if P == 0:
                    break
    if P > 0:
        S[idx] = aS; Sinv[idx] = aSinv; Slast[idx] = aSlast
    converged = (Slast / LAST) < 1
    measure = torch.where(converged, -S, Sinv)
    out = dict(measure=measure)
    if return_traj:
        out['traj'] = torch.stack(traj, 1)
    return out


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
    """Pixel-centre grid in log10 space, centred at (c0,c1) (log10 eta0, log10 eta1),
    window half-width half_w decades. Returns eta0 (res*res,), eta1 (res*res,) with
    eta0 varying along columns (x) and eta1 along rows (y, row 0 = bottom).
    Offsets are formed first and the exponentiation is done as
    10**c * 10**offset, which keeps full float64 relative resolution at the centre."""
    off = (torch.arange(res, dtype=DT) + 0.5) / res * 2.0 - 1.0
    off = off * half_w
    base0 = 10.0 ** torch.tensor(c0, dtype=DT)
    base1 = 10.0 ** torch.tensor(c1, dtype=DT)
    e0 = base0 * torch.pow(torch.tensor(10.0, dtype=DT), off)
    e1 = base1 * torch.pow(torch.tensor(10.0, dtype=DT), off)
    E1, E0 = torch.meshgrid(e1, e0, indexing='ij')
    return E0.reshape(-1).to(dev), E1.reshape(-1).to(dev)


def run_grid(prob, h0, h1, steps=500, chunk=32768, trainer=None, verbose=True, **kw):
    """h0,h1: flat (P,) hyperparameter tensors. Returns numpy float64 measure (P,)."""
    trainer = trainer or (train_chunk_quadratic if prob['nonlin'] == 'quadratic' else train_chunk)
    P = h0.numel()
    out = np.empty(P, dtype=np.float64)
    t0 = time.time()
    extras = {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)}
    tens = {k: v for k, v in kw.items() if isinstance(v, torch.Tensor)}
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        sub = {k: v[s:e] for k, v in tens.items()}
        with torch.no_grad():
            r = trainer(prob, h0[s:e], h1[s:e], steps=steps, **sub, **extras)
        out[s:e] = r['measure'].cpu().numpy()
        if verbose and (s // chunk) % 8 == 0:
            el = time.time() - t0
            print(f'  chunk {s//chunk+1}/{math.ceil(P/chunk)}  {el:.0f}s  '
                  f'({e/el:.0f} px/s)', flush=True)
    return out


def setup_gpu(frac=0.10):
    torch.cuda.set_per_process_memory_fraction(frac)
    torch.backends.cuda.matmul.allow_tf32 = False
