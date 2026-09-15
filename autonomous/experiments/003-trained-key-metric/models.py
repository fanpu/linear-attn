"""White-box one-layer models for in-context regression whose key metric M = A^T A can be read off directly.

DeltaNet-WB: tokens are interleaved [query x_t, write (x_t, y_t)]. At write tokens the key is Ax/||Ax|| and the value is
y/||Ax||; at query tokens beta = 0 and the query is Ax (unnormalized), so the readout is y_hat = S A x. The gated delta rule
then performs exactly  w <- alpha * (w + beta (y - w.x) A^T A x / x^T A^T A x)  with w = A^T S^T.

LinAttn-WB: y_hat_t = gamma/(t-1) sum_{i<t} y_i (A x_i).(A x_t)   (one-pass Hebbian estimate, Zhang-Frei-Bartlett form).
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))


class DeltaNetWB(nn.Module):
    def __init__(self, d, A0, learn_alpha=False, alpha0=0.99, beta0=0.5, impl="fla"):
        super().__init__()
        self.A = nn.Parameter(A0.clone().float())
        self.b = nn.Parameter(torch.tensor(math.log(beta0 / (2 - beta0))))       # beta = 2 sigmoid(b)
        self.learn_alpha = learn_alpha
        self.a = nn.Parameter(torch.tensor(math.log(alpha0 / (1 - alpha0))), requires_grad=learn_alpha)
        self.impl = impl

    @property
    def beta(self):
        return 2 * torch.sigmoid(self.b)

    @property
    def alpha(self):
        return torch.sigmoid(self.a) if self.learn_alpha else torch.ones((), device=self.A.device)

    def forward(self, x, y):
        B, T, d = x.shape
        kx = x @ self.A.T
        n = kx.norm(dim=-1, keepdim=True)
        Q = kx.new_zeros(B, 2 * T, 1, d); Q[:, 0::2, 0] = kx
        K = kx.new_zeros(B, 2 * T, 1, d); K[:, 1::2, 0] = kx / n
        V = kx.new_zeros(B, 2 * T, 1, 1); V[:, 1::2, 0] = y[..., None] / n
        Bt = kx.new_zeros(B, 2 * T, 1); Bt[:, 1::2, 0] = self.beta
        G = kx.new_zeros(B, 2 * T, 1)
        if self.learn_alpha:
            G[:, 1::2, 0] = torch.log(self.alpha)
        if self.impl == "fla":
            from recurrences import fla_delta
            o, _ = fla_delta(Q, K, V, Bt, G, mode="chunk")
        else:
            from recurrences import delta_ref
            o, _ = delta_ref(Q, K, V, Bt, G)
        return o[:, 0::2, 0, 0]

    def metric(self):
        return (self.A.T @ self.A).detach()


class LinAttnWB(nn.Module):
    def __init__(self, d, A0):
        super().__init__()
        self.A = nn.Parameter(A0.clone().float())
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, y):
        kx = x @ self.A.T                                            # [B,T,d]
        acc = torch.cumsum(y[..., None] * kx, dim=1)                 # sum_{i<=t} y_i A x_i
        prev = torch.cat([torch.zeros_like(acc[:, :1]), acc[:, :-1]], 1)
        cnt = torch.arange(x.shape[1], device=x.device).clamp(min=1).to(x.dtype)
        return self.gamma * (prev * kx).sum(-1) / cnt

    def metric(self):
        return (self.A.T @ self.A).detach()


def spectrum(d, kappa):
    lam = torch.logspace(0, math.log10(kappa), d) if kappa > 1 else torch.ones(d)
    return lam / lam.mean()


def make_batch(B, T, d, lam, q, sigma2, device, gen=None):
    """x ~ N(0, diag lam); w_1 ~ N(0, I/d); w_{t+1} = sqrt(1-q) w_t + sqrt(q/d) xi."""
    kw = dict(device=device, generator=gen)
    x = torch.randn(B, T, d, **kw) * lam.sqrt()
    w = torch.randn(B, d, **kw) / math.sqrt(d)
    if q > 0:
        a = math.sqrt(1 - q)
        steps = torch.randn(B, T, d, **kw) * math.sqrt(q / d)
        ws = [w]
        for t in range(1, T):
            ws.append(a * ws[-1] + steps[:, t])
        W = torch.stack(ws, 1)
    else:
        W = w[:, None].expand(B, T, d)
    y = (W * x).sum(-1) + math.sqrt(sigma2) * torch.randn(B, T, **kw)
    return x, y, W


def fit_exponent(M, lam):
    """Slope s of log diag(M) = c - s log lam, the fit's R^2, and the off-diagonal Frobenius fraction."""
    dg = torch.diagonal(M).clamp(min=1e-12).log().double().cpu()
    ll = lam.log().double().cpu()
    X = torch.stack([torch.ones_like(ll), ll], 1)
    coef = torch.linalg.lstsq(X, dg[:, None]).solution[:, 0]
    pred = X @ coef
    r2 = 1 - ((dg - pred) ** 2).sum() / ((dg - dg.mean()) ** 2).sum()
    off = (M - torch.diag(torch.diagonal(M))).norm() / M.norm()
    return float(-coef[1]), float(r2), float(off)
