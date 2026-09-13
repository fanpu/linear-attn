"""Batched full-batch GD engines: every pixel of a 2D slice is an independent tiny network.

All state is float64 on the GPU.  Gradients are written out by hand (no autograd) and the
per-step update is fused by torch.compile, so a 1M-pixel image costs ~a few ms per step.
"""
import math, time
import numpy as np
import torch

DEV = 'cuda'
DT = torch.float64


def gpu_setup(frac=0.10):
    torch.cuda.set_per_process_memory_fraction(frac)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)


# ----------------------------------------------------------------------------------------
# Problem 1: two-layer tanh MLP  (d_in -> H -> 1), MSE, full batch
#   theta layout per pixel: W (H*d), b (H), a (H), c (1)
# ----------------------------------------------------------------------------------------
class MLP:
    def __init__(self, X, t, H):
        self.X = torch.as_tensor(X, dtype=DT, device=DEV)       # (N,d)
        self.t = torch.as_tensor(t, dtype=DT, device=DEV)       # (N,)
        self.N, self.d = self.X.shape
        self.H = H
        self.D = H * self.d + 2 * H + 1
        self._step = torch.compile(self._step_impl, dynamic=True)

    def unpack(self, th):
        H, d = self.H, self.d
        W = th[:, :H * d].reshape(-1, H, d)
        b = th[:, H * d:H * d + H]
        a = th[:, H * d + H:H * d + 2 * H]
        c = th[:, -1]
        return W, b, a, c

    def forward(self, th):
        W, b, a, c = self.unpack(th)
        z = torch.einsum('phd,nd->pnh', W, self.X) + b[:, None, :]
        h = torch.tanh(z)
        y = torch.einsum('pnh,ph->pn', h, a) + c[:, None]
        return z, h, y

    def loss(self, th):
        _, _, y = self.forward(th)
        return 0.5 * ((y - self.t) ** 2).mean(1)

    def _step_impl(self, th, eta):
        H, d, N = self.H, self.d, self.N
        W = th[:, :H * d].reshape(-1, H, d)
        b = th[:, H * d:H * d + H]
        a = th[:, H * d + H:H * d + 2 * H]
        c = th[:, -1]
        z = torch.einsum('phd,nd->pnh', W, self.X) + b[:, None, :]
        h = torch.tanh(z)
        y = torch.einsum('pnh,ph->pn', h, a) + c[:, None]
        e = (y - self.t) / N                                   # (P,N)
        loss = 0.5 * N * (e * e).sum(1)
        ga = torch.einsum('pn,pnh->ph', e, h)
        gc = e.sum(1)
        dz = e[:, :, None] * a[:, None, :] * (1 - h * h)       # (P,N,H)
        gW = torch.einsum('pnh,nd->phd', dz, self.X).reshape(-1, H * d)
        gb = dz.sum(1)
        g = torch.cat([gW, gb, ga, gc[:, None]], 1)
        return th - eta * g, loss

    def step(self, th, eta):
        return self._step(th, eta)

    def jacobian(self, th):
        """d yhat_n / d theta, shape (P,N,D)."""
        W, b, a, c = self.unpack(th)
        z, h, y = self.forward(th)
        s = a[:, None, :] * (1 - h * h)                          # (P,N,H)
        P = th.shape[0]
        jW = (s[:, :, :, None] * self.X[None, :, None, :]).reshape(P, self.N, -1)
        jb = s
        ja = h
        jc = torch.ones(P, self.N, 1, dtype=DT, device=DEV)
        return torch.cat([jW, jb, ja, jc], 2)


# ----------------------------------------------------------------------------------------
# Problem 2: scalar factorisation  f = 1/4 (prod_i x_i - 1)^2
# ----------------------------------------------------------------------------------------
class ScalarFact:
    def __init__(self, depth):
        self.D = depth
        self._step = torch.compile(self._step_impl, dynamic=True)

    def _step_impl(self, th, eta):
        p = th.prod(1, keepdim=True)
        r = p - 1
        # d/dx_i prod = prod_{j!=i} x_j ; compute without division
        D = th.shape[1]
        cols = []
        for i in range(D):
            others = torch.ones_like(th[:, 0])
            for j in range(D):
                if j != i:
                    others = others * th[:, j]
            cols.append(others)
        g = 0.5 * r * torch.stack(cols, 1)
        return th - eta * g, 0.25 * (r[:, 0] ** 2)

    def step(self, th, eta):
        return self._step(th, eta)

    def loss(self, th):
        return 0.25 * (th.prod(1) - 1) ** 2


# ----------------------------------------------------------------------------------------
# Generic driver with active-set compaction
# ----------------------------------------------------------------------------------------
STATUS_CONV, STATUS_NOCONV, STATUS_DIV = 0, 1, 2


def run_gd(prob, th0, eta, T, tol=1e-10, check=50, div_thresh=1e6, chunk=2_000_000,
           cycle_probe=64, verbose=False):
    """Run GD on each row of th0 (P,D) for up to T steps.

    Returns dict of numpy arrays:
      theta   final parameters (for converged: first time loss<tol held over `check` steps)
      loss    final loss
      tconv   step at which loss first went below tol and stayed (T if never)
      status  0 converged, 1 not converged by T, 2 diverged (non-finite or |theta|>div_thresh)
      period  for non-converged pixels: smallest p in 1..cycle_probe with |th_T - th_{T-p}|<1e-7
    """
    P, D = th0.shape
    out_th = np.empty((P, D)); out_loss = np.empty(P); out_t = np.full(P, T, dtype=np.int32)
    out_s = np.empty(P, dtype=np.int8); out_per = np.zeros(P, dtype=np.int16)
    eta_t = torch.tensor(float(eta), dtype=DT, device=DEV)
    for s0 in range(0, P, chunk):
        th = torch.as_tensor(th0[s0:s0 + chunk], dtype=DT, device=DEV).clone()
        n = th.shape[0]
        idx = torch.arange(n, device=DEV)
        first = torch.full((n,), -1, dtype=torch.int64, device=DEV)   # first step below tol
        res_th = torch.empty_like(th); res_loss = torch.empty(n, dtype=DT, device=DEV)
        res_t = torch.full((n,), T, dtype=torch.int64, device=DEV)
        res_s = torch.full((n,), STATUS_NOCONV, dtype=torch.int8, device=DEV)
        t = 0
        while t < T and idx.numel() > 0:
            k = min(check, T - t)
            below_all = torch.ones(idx.numel(), dtype=torch.bool, device=DEV)
            for j in range(k):
                th, loss = prob.step(th, eta_t)
                below = loss < tol
                below_all &= below
                newly = below & (first < 0)
                first = torch.where(newly, torch.full_like(first, t + j), first)
                first = torch.where(below, first, torch.full_like(first, -1))
            t += k
            bad = ~torch.isfinite(th).all(1) | (th.abs().amax(1) > div_thresh)
            conv = below_all & ~bad & (first >= 0)
            done = bad | conv
            if done.any():
                di = idx[done]
                res_th[di] = th[done]; res_loss[di] = loss[done]
                res_s[di] = torch.where(bad[done], torch.tensor(STATUS_DIV, dtype=torch.int8, device=DEV),
                                        torch.tensor(STATUS_CONV, dtype=torch.int8, device=DEV))
                res_t[di] = torch.where(bad[done], torch.full_like(first[done], t), first[done])
                keep = ~done
                th, idx, first = th[keep], idx[keep], first[keep]
            if verbose and (t // check) % 40 == 0:
                print(f'  t={t} active={idx.numel()}', flush=True)
        per = torch.zeros(n, dtype=torch.int16, device=DEV)
        if idx.numel() > 0:
            # remaining: not converged. probe for a cycle.
            _, loss = prob.step(th, eta_t)
            ref = th.clone(); cur = th
            p_found = torch.zeros(idx.numel(), dtype=torch.int16, device=DEV)
            for p in range(1, cycle_probe + 1):
                cur, _ = prob.step(cur, eta_t)
                hit = ((cur - ref).abs().amax(1) < 1e-7) & (p_found == 0)
                p_found = torch.where(hit, torch.full_like(p_found, p), p_found)
            res_th[idx] = th; res_loss[idx] = loss; per[idx] = p_found
        out_th[s0:s0 + n] = res_th.cpu().numpy(); out_loss[s0:s0 + n] = res_loss.cpu().numpy()
        out_t[s0:s0 + n] = res_t.cpu().numpy(); out_s[s0:s0 + n] = res_s.cpu().numpy()
        out_per[s0:s0 + n] = per.cpu().numpy()
    return dict(theta=out_th, loss=out_loss, tconv=out_t, status=out_s, period=out_per)


def slice_grid(center, u, v, lo_a, hi_a, lo_b, hi_b, R, Rb=None):
    """theta = center + alpha u + beta v on an R x Rb grid (row = beta, col = alpha), float64."""
    Rb = Rb or R
    al = np.linspace(lo_a, hi_a, R)
    be = np.linspace(lo_b, hi_b, Rb)
    A, B = np.meshgrid(al, be)
    th = center[None, :] + A.reshape(-1, 1) * u[None, :] + B.reshape(-1, 1) * v[None, :]
    return th
