"""Shared dynamics for the game-chaos gallery.

All learning rules are written in *logit* (dual / cumulative-payoff) coordinates, where
they are exact and numerically benign, and every Lyapunov exponent is computed by
propagating a tangent vector with the exact Jacobian (Benettin renormalisation every step).
float64 everywhere.

Systems
-------
cong1d  Two agents, two parallel links with linear costs c1(l)=a*l, c2(l)=b*l, exponential
        multiplicative weights (MWU_e, Palaiopanos-Panageas-Piliouras 2017) with step eta.
        In logits u = log(x/(1-x)) the update is
            u' = u - s (sigma(v) - y*),   v' = v - s (sigma(u) - y*),
        with s = eta (a+b) the effective step size and y* = (2b-a)/(a+b) the mixed-NE load.
        Both players starting from the same mixed strategy stay on the diagonal u=v, which
        gives the 1-D map u' = u - s (sigma(u) - y*)   (the map proven Li-Yorke chaotic).
rps_ewa Two players, generalised rock-paper-scissors (Sato-Akiyama-Farmer 2002 payoffs
        with tie payoffs eps_x, eps_y), experience-weighted attraction
        (Galla-Farmer 2013):  Q' = (1-alpha) Q + beta * payoff,  x = softmax(Q).
        alpha = 0 is plain MWU / FTRL-entropic.
"""
import numpy as np
import torch

DT = torch.float64


def dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gpu_setup(frac=0.10):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)


# ----------------------------------------------------------------------------- 1-D congestion
@torch.no_grad()
def lyap_cong1d(s, ystar, T0=2000, T1=4000, u0=0.1, chunk=1 << 22):
    """Largest (only) Lyapunov exponent of u' = u - s(sigma(u)-y*) per parameter pair.
    s, ystar: tensors (same shape). Returns numpy float64 array (nats / iteration)."""
    out = torch.empty(s.numel(), dtype=DT)
    S = s.reshape(-1); Y = ystar.reshape(-1)
    for i in range(0, S.numel(), chunk):
        ss = S[i:i + chunk].to(dev(), DT); yy = Y[i:i + chunk].to(dev(), DT)
        u = torch.full_like(ss, u0)
        for _ in range(T0):
            u = u - ss * (torch.sigmoid(u) - yy)
        L = torch.zeros_like(ss)
        for _ in range(T1):
            p = torch.sigmoid(u)
            L += torch.log(torch.abs(1 - ss * p * (1 - p)).clamp_min(1e-300))
            u = u - ss * (p - yy)
        out[i:i + chunk] = (L / T1).cpu()
    return out.reshape(s.shape).numpy()


@torch.no_grad()
def orbit_cong1d(s, ystar, T0=2000, T1=512, u0=0.1):
    """Attractor samples x_t = sigma(u_t), t in [T0, T0+T1) for a batch of parameters."""
    ss = s.to(dev(), DT); yy = ystar.to(dev(), DT)
    u = torch.full_like(ss, u0)
    for _ in range(T0):
        u = u - ss * (torch.sigmoid(u) - yy)
    xs = []
    for _ in range(T1):
        u = u - ss * (torch.sigmoid(u) - yy)
        xs.append(torch.sigmoid(u))
    return torch.stack(xs, -1).cpu().numpy()


# ----------------------------------------------------------------------------- RPS payoffs
def rps_apply(e, z):
    """(A(e) z) for SAF matrix A = [[e,-1,1],[1,e,-1],[-1,1,e]]; z (...,3), e broadcastable."""
    z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2]
    return torch.stack([e * z0 - z1 + z2, z0 + e * z1 - z2, -z0 + z1 + e * z2], -1)


def rps_matrix(e):
    return np.array([[e, -1, 1], [1, e, -1], [-1, 1, e]], float)


@torch.no_grad()
def lyap_rps_ewa(beta, alpha, ex, ey, T0=3000, T1=5000, x0=(0.5, 0.3, 0.2),
                 y0=(0.2, 0.3, 0.5), chunk=1 << 20, seed=0, return_state=False):
    """Largest Lyapunov exponent of discrete EWA on generalised RPS, per parameter tuple.
    All parameter args are tensors of a common shape (or python floats)."""
    shape = torch.broadcast_shapes(*[torch.as_tensor(a).shape for a in (beta, alpha, ex, ey)])
    B, A, EX, EY = [torch.as_tensor(a, dtype=DT).expand(shape).reshape(-1) for a in (beta, alpha, ex, ey)]
    n = B.numel(); out = torch.empty(n, dtype=DT); spread = torch.empty(n, dtype=DT)
    g = torch.Generator().manual_seed(seed)
    for i in range(0, n, chunk):
        b = B[i:i + chunk].to(dev())[:, None]; a = A[i:i + chunk].to(dev())[:, None]
        ex_ = EX[i:i + chunk].to(dev()); ey_ = EY[i:i + chunk].to(dev())
        m = b.shape[0]
        Qx = torch.log(torch.tensor(x0, dtype=DT, device=dev())).expand(m, 3).clone()
        Qy = torch.log(torch.tensor(y0, dtype=DT, device=dev())).expand(m, 3).clone()
        d = torch.randn(6, generator=g, dtype=DT).to(dev())
        dQx = d[:3].expand(m, 3).clone(); dQy = d[3:].expand(m, 3).clone()
        L = torch.zeros(m, dtype=DT, device=dev())
        xmin = torch.full((m,), 1.0, dtype=DT, device=dev())
        for t in range(T0 + T1):
            x = torch.softmax(Qx, -1); y = torch.softmax(Qy, -1)
            dx = x * (dQx - (x * dQx).sum(-1, keepdim=True))
            dy = y * (dQy - (y * dQy).sum(-1, keepdim=True))
            Qx, Qy = (1 - a) * Qx + b * rps_apply(ex_[:, None], y), (1 - a) * Qy + b * rps_apply(ey_[:, None], x)
            dQx, dQy = (1 - a) * dQx + b * rps_apply(ex_[:, None], dy), (1 - a) * dQy + b * rps_apply(ey_[:, None], dx)
            Qx = Qx - Qx.mean(-1, keepdim=True); Qy = Qy - Qy.mean(-1, keepdim=True)
            dQx = dQx - dQx.mean(-1, keepdim=True); dQy = dQy - dQy.mean(-1, keepdim=True)
            nrm = torch.sqrt((dQx ** 2).sum(-1) + (dQy ** 2).sum(-1))
            dQx = dQx / nrm[:, None]; dQy = dQy / nrm[:, None]
            if t >= T0:
                L += torch.log(nrm)
                xmin = torch.minimum(xmin, torch.minimum(x.min(-1).values, y.min(-1).values))
        out[i:i + chunk] = (L / T1).cpu(); spread[i:i + chunk] = xmin.cpu()
    if return_state:
        return out.reshape(shape).numpy(), spread.reshape(shape).numpy()
    return out.reshape(shape).numpy()


# ----------------------------------------------------------------------------- colour helpers
def spectral_split(lam, ref=None, buffer=0.25, zero_tol=0.0):
    """Sohl-Dickstein 'cdf_img' restretch adapted to a Lyapunov exponent.
    lam<0 (ordered) -> rank-normalised into [buffer,1] of the Spectral upper half, the
    slowest contraction (lam just below 0) at the dark purple end;
    lam>0 (chaotic) -> rank-normalised into the lower half, lam just above 0 at deep red.
    Returns values in [0,1] to feed matplotlib 'Spectral'. zero_tol: |lam|<=tol counts as
    chaotic=False (neutral side)."""
    x = np.asarray(lam, float)
    r = x if ref is None else np.asarray(ref, float)
    u = np.sort(r.ravel())
    pos_thr = zero_tol
    nn = int((u <= pos_thr).sum()); npos = u.size - nn
    # negatives: most negative -> -1 ... near zero -> -buffer ; positives: small -> buffer ... large -> 1
    v = np.concatenate([np.linspace(-1, -buffer, nn), np.linspace(buffer, 1, npos)])
    y = np.interp(x, u, v)          # -1..-b (ordered), b..1 (chaotic)
    # Spectral: 0 deep red, 0.5 pale yellow, 1 purple.
    # ordered: near zero (-b) -> purple (1.0) ; most negative (-1) -> pale (0.5+)
    # chaotic: near zero (b) -> deep red (0.0) ; largest (1) -> pale (0.5-)
    c = np.where(y < 0, 1.0 - (-y - buffer) / (1 - buffer) * 0.5 * 0.75,
                 0.0 + (y - buffer) / (1 - buffer) * 0.5 * 0.75)
    return c
