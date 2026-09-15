"""Batched float64 simulation: gated delta rule (NLMS form) vs Kalman filter on drifting in-context regression.

Data (streamed one step at a time, so memory is independent of T):
    x_t ~ N(0, diag(lam))  with mean(lam) = 1,     y_t = <w_t, x_t> + sigma * eps_t
    w_1 ~ N(0, I/d),        w_{t+1} = a w_t + sqrt(q/d) xi_t,   a = sqrt(1 - q)

Both memories predict y_t with u_t = (decayed previous estimate), then correct:
    delta rule   u_t = alpha w_hat,   w_hat <- u_t + beta (y_t - u_t.x_t) M x_t / (x_t^T M x_t)
                 (M = I is the plain NLMS / gated delta rule; M = Sigma^{-s} is a static key preconditioner)
    Kalman       u_t = a w_hat,       P^- = a^2 P + (q/d) I,   K = P^- x / (x^T P^- x + sigma^2),
                 w_hat <- u_t + K (y_t - u_t.x_t),   P <- P^- - K (P^- x)^T

We record the excess risk E[(u_t.x_t - w_t.x_t)^2] = E[(u_t - w_t)^T Sigma (u_t - w_t)] at every step
(computed exactly from the error vector, which has much lower variance than squaring sampled residuals).
"""
import math

import torch

torch.set_default_dtype(torch.float64)


def spectrum(d, kappa):
    """Log-uniformly spaced input variances from 1 to kappa, rescaled to mean 1 (so tr Sigma = d)."""
    lam = torch.logspace(0, math.log10(kappa), d) if kappa > 1 else torch.ones(d)
    return lam / lam.mean()


def simulate(d, q, sigma2, T, batch, alphas, betas, kappa=1.0, precond_s=0.0, kalman=True, seed=0,
             device="cpu", chunks=1):
    """Returns dict with per-step excess risk curves:
        'gdn': [G, T] for each (alpha, beta) pair in the flattened grid (G = len(alphas) * len(betas))
        'kf':  [T]
    With chunks > 1 the batch is split into independent chunks and curves get a trailing [.., chunks] axis
    (for error bars). All memories see exactly the same data stream (common random numbers)."""
    g = torch.Generator(device=device).manual_seed(seed)
    lam = spectrum(d, kappa).to(device)
    sd = lam.sqrt()
    a, sig = math.sqrt(1 - q), math.sqrt(sigma2)
    A, Bt = torch.meshgrid(torch.as_tensor(alphas, device=device), torch.as_tensor(betas, device=device), indexing="ij")
    A, Bt = A.reshape(-1, 1, 1), Bt.reshape(-1, 1)
    G = A.shape[0]
    M = lam.pow(-precond_s)  # diagonal static preconditioner applied to the key direction

    w = torch.randn(batch, d, generator=g, device=device) / math.sqrt(d)
    wh = torch.zeros(G, batch, d, device=device)
    risk_g = torch.zeros(G, T, chunks, device=device)
    if kalman:
        wk = torch.zeros(batch, d, device=device)
        P = torch.eye(d, device=device).expand(batch, d, d) / d
        risk_k = torch.zeros(T, chunks, device=device)
        Q = torch.eye(d, device=device) * (q / d)

    for t in range(T):
        x = torch.randn(batch, d, generator=g, device=device) * sd
        y = (w * x).sum(-1) + sig * torch.randn(batch, generator=g, device=device)

        u = A * wh                                              # [G,B,d]
        err = u - w
        risk_g[:, t] = (err.pow(2) * lam).sum(-1).view(G, chunks, -1).mean(-1)
        Mx = M * x
        resid = y - (u * x).sum(-1)                             # [G,B]
        wh = u + (Bt * resid / (x * Mx).sum(-1))[..., None] * Mx

        if kalman:
            uk = a * wk
            risk_k[t] = ((uk - w).pow(2) * lam).sum(-1).view(chunks, -1).mean(-1)
            P = a * a * P + Q
            Px = torch.einsum("bij,bj->bi", P, x)
            s = (x * Px).sum(-1) + sigma2
            wk = uk + (Px / s[:, None]) * (y - (uk * x).sum(-1))[:, None]
            P = P - torch.einsum("bi,bj->bij", Px, Px) / s[:, None, None]

        w = a * w + math.sqrt(q / d) * torch.randn(batch, d, generator=g, device=device)

    if chunks == 1:
        risk_g = risk_g[..., 0]
        if kalman:
            risk_k = risk_k[..., 0]
    out = {"gdn": risk_g.cpu(), "alphas": list(map(float, alphas)), "betas": list(map(float, betas))}
    if kalman:
        out["kf"] = risk_k.cpu()
    return out
