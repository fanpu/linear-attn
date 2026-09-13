"""Batched full-batch GD engines: every pixel of a 2D slice is an independent tiny model.

State is float64 on the GPU. Gradients are hand-written (no autograd); `check` GD steps are
unrolled into one torch.compile graph so a window of steps is a handful of fused kernels.
"""
import math, time
import numpy as np
import torch

DEV = 'cuda'
DT = torch.float64


def gpu_setup(frac=0.10):
    torch.cuda.set_per_process_memory_fraction(frac)
    torch.set_grad_enabled(False)


class MLP:
    """d_in=2 -> H tanh -> 1 linear, loss = 1/(2N) sum (y - t)^2.
    theta per pixel: [W00 W01 W10 W11 ... (H rows of 2)] + b (H) + a (H) + c (1)."""

    def __init__(self, X, t, H):
        self.X = torch.as_tensor(X, dtype=DT, device=DEV)
        self.t = torch.as_tensor(t, dtype=DT, device=DEV)
        self.N = self.X.shape[0]
        assert self.X.shape[1] == 2
        self.H = H
        self.D = 4 * H + 1

    def split(self, th):
        H = self.H
        return th[:, 0:2 * H:2], th[:, 1:2 * H:2], th[:, 2 * H:3 * H], th[:, 3 * H:4 * H], th[:, 4 * H:4 * H + 1]

    def pre(self, th):
        W0, W1, b, a, c = self.split(th)
        x0, x1 = self.X[:, 0], self.X[:, 1]
        return W0[:, None, :] * x0[None, :, None] + W1[:, None, :] * x1[None, :, None] + b[:, None, :]

    def output(self, th, Xq=None):
        W0, W1, b, a, c = self.split(th)
        Xq = self.X if Xq is None else torch.as_tensor(Xq, dtype=DT, device=DEV)
        z = W0[:, None, :] * Xq[None, :, 0, None] + W1[:, None, :] * Xq[None, :, 1, None] + b[:, None, :]
        return (torch.tanh(z) * a[:, None, :]).sum(2) + c

    def loss(self, th):
        e = self.output(th) - self.t
        return 0.5 * (e * e).mean(1)

    def step(self, th, eta):
        H, N = self.H, self.N
        W0, W1, b, a, c = self.split(th)
        x0, x1 = self.X[:, 0], self.X[:, 1]
        z = W0[:, None, :] * x0[None, :, None] + W1[:, None, :] * x1[None, :, None] + b[:, None, :]
        h = torch.tanh(z)
        y = (h * a[:, None, :]).sum(2) + c
        e = (y - self.t) / N
        loss = 0.5 * N * (e * e).sum(1)
        ga = (e[:, :, None] * h).sum(1)
        gc = e.sum(1, keepdim=True)
        dz = e[:, :, None] * a[:, None, :] * (1 - h * h)
        gW0 = (dz * x0[None, :, None]).sum(1)
        gW1 = (dz * x1[None, :, None]).sum(1)
        gb = dz.sum(1)
        gW = torch.stack([gW0, gW1], 2).reshape(-1, 2 * H)
        return th - eta * torch.cat([gW, gb, ga, gc], 1), loss

    def jacobian(self, th):
        W0, W1, b, a, c = self.split(th)
        z = self.pre(th); h = torch.tanh(z)
        s = a[:, None, :] * (1 - h * h)                       # (P,N,H)
        x0, x1 = self.X[:, 0], self.X[:, 1]
        jW = torch.stack([s * x0[None, :, None], s * x1[None, :, None]], 3).reshape(th.shape[0], self.N, -1)
        return torch.cat([jW, s, h, torch.ones_like(h[:, :, :1])], 2)

    def gn_sharpness(self, th):
        """Largest eigenvalue of the Gauss-Newton matrix (1/N) J^T J (= Hessian at zero loss)."""
        J = self.jacobian(th)
        K = J @ J.transpose(1, 2) / self.N
        return torch.linalg.eigvalsh(K)[:, -1]


class ScalarFact:
    """f = 1/4 (prod_i x_i - 1)^2"""

    def __init__(self, depth):
        self.D = depth

    def step(self, th, eta):
        D = self.D
        cols = []
        for i in range(D):
            o = torch.ones_like(th[:, 0])
            for j in range(D):
                if j != i:
                    o = o * th[:, j]
            cols.append(o)
        r = (cols[0] * th[:, 0])[:, None] - 1
        g = 0.5 * r * torch.stack(cols, 1)
        return th - eta * g, 0.25 * r[:, 0] ** 2

    def loss(self, th):
        return 0.25 * (th.prod(1) - 1) ** 2

    def sharpness(self, th):
        cols = []
        for i in range(self.D):
            m = torch.ones(self.D, dtype=torch.bool, device=th.device); m[i] = False
            cols.append(th[:, m].prod(1))
        return 0.5 * (torch.stack(cols, 1) ** 2).sum(1)


STATUS_CONV, STATUS_NOCONV, STATUS_DIV = 0, 1, 2
_compiled = {}


def _window(prob, k):
    key = (id(prob), k)
    if key not in _compiled:
        def f(th, eta):
            for _ in range(k):
                th, loss = prob.step(th, eta)
            return th, loss
        _compiled[key] = torch.compile(f, dynamic=True)
    return _compiled[key]


def run_gd(prob, th0, eta, T, tol=1e-12, check=20, div_thresh=1e4, chunk=1 << 20,
           cycle_probe=0, log=None):
    """GD for up to T steps on every row of th0 (P,D).

    Loss is inspected every `check` steps. A pixel is *converged* once loss < tol at two
    consecutive checks; *diverged* once any |theta| > div_thresh or non-finite.
    tconv is a smoothed first-passage time: linear interpolation of log-loss between the two
    checks that bracket the tol crossing (a declared continuous-colouring choice, like the
    smoothed escape count of Mandelbrot renderers).  For diverged pixels tconv is the
    analogous smoothed escape time on log|theta|.
    """
    P, D = th0.shape
    win = _window(prob, check)
    out = dict(theta=np.empty((P, D)), loss=np.empty(P), tconv=np.full(P, float(T)),
               status=np.full(P, STATUS_NOCONV, np.int8), period=np.zeros(P, np.int16))
    eta_t = torch.tensor(float(eta), dtype=DT, device=DEV)
    ltol, lthr = math.log(tol), math.log(div_thresh)
    for s0 in range(0, P, chunk):
        th = torch.as_tensor(th0[s0:s0 + chunk], dtype=DT, device=DEV).clone()
        n = th.shape[0]
        idx = torch.arange(n, device=DEV)
        r_th = torch.empty_like(th); r_loss = torch.empty(n, dtype=DT, device=DEV)
        r_t = torch.full((n,), float(T), dtype=DT, device=DEV)
        r_s = torch.full((n,), STATUS_NOCONV, dtype=torch.int8, device=DEV)
        prev_ll = torch.full((n,), 50.0, dtype=DT, device=DEV)     # log loss at previous check
        prev_lm = torch.log(th.abs().amax(1))
        tcross = torch.full((n,), -1.0, dtype=DT, device=DEV)
        t = 0
        while t < T and idx.numel() > 0:
            th, loss = win(th, eta_t)
            t += check
            ll = torch.log(loss.clamp_min(1e-300))
            lm = torch.log(th.abs().amax(1))
            bad = ~torch.isfinite(lm) | (lm > lthr)
            below = torch.isfinite(ll) & (ll < ltol)
            frac = ((prev_ll - ltol) / (prev_ll - ll).clamp_min(1e-12)).clamp(0, 1)
            tc = (t - check) + check * frac
            newly = below & (tcross < 0)
            tcross = torch.where(newly, tc, torch.where(below, tcross, torch.full_like(tcross, -1.0)))
            conv = below & (tcross >= 0) & (tcross <= t - check) & ~bad
            fe = ((lthr - prev_lm) / (lm - prev_lm).clamp_min(1e-12)).clamp(0, 1)
            fe = torch.where(torch.isfinite(lm), fe, torch.ones_like(fe))
            te = (t - check) + check * fe
            done = bad | conv
            prev_ll, prev_lm = ll, lm
            if done.any():
                di = idx[done]
                r_th[di] = th[done]; r_loss[di] = loss[done]
                r_s[di] = torch.where(bad[done], torch.tensor(STATUS_DIV, dtype=torch.int8, device=DEV),
                                      torch.tensor(STATUS_CONV, dtype=torch.int8, device=DEV))
                r_t[di] = torch.where(bad[done], te[done], tcross[done])
                keep = ~done
                th, idx, tcross, prev_ll, prev_lm = th[keep], idx[keep], tcross[keep], prev_ll[keep], prev_lm[keep]
            if log and (t // check) % 100 == 0:
                print(f'    t={t} active={idx.numel()}', file=log, flush=True)
        per = torch.zeros(n, dtype=torch.int16, device=DEV)
        if idx.numel() > 0:
            _, loss = prob.step(th, eta_t)
            r_th[idx] = th; r_loss[idx] = loss
            if cycle_probe:
                ref = th.clone(); cur = th
                pf = torch.zeros(idx.numel(), dtype=torch.int16, device=DEV)
                for p in range(1, cycle_probe + 1):
                    cur, _ = prob.step(cur, eta_t)
                    hit = ((cur - ref).abs().amax(1) < 1e-8) & (pf == 0)
                    pf = torch.where(hit, torch.full_like(pf, p), pf)
                per[idx] = pf
        out['theta'][s0:s0 + n] = r_th.cpu().numpy(); out['loss'][s0:s0 + n] = r_loss.cpu().numpy()
        out['tconv'][s0:s0 + n] = r_t.cpu().numpy(); out['status'][s0:s0 + n] = r_s.cpu().numpy()
        out['period'][s0:s0 + n] = per.cpu().numpy()
    return out


def slice_grid(center, u, v, a0, a1, b0, b1, R, Rb=None):
    """theta = center + alpha u + beta v; row index = beta (bottom-up), col = alpha. float64."""
    Rb = Rb or R
    al = np.linspace(a0, a1, R); be = np.linspace(b0, b1, Rb)
    A, B = np.meshgrid(al, be)
    return center[None, :] + A.reshape(-1, 1) * u[None, :] + B.reshape(-1, 1) * v[None, :]
