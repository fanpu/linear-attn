"""Shared machinery for the Which-Dog project.

Diffusion convention (used for both the 2-D toy and MNIST):
  continuous-time VP process, linear beta:  beta(t) = b0 + t (b1 - b0), b0 = 0.1, b1 = 20
  alpha_bar(t) = exp(-(b0 t + (b1 - b0) t^2 / 2)),  x_t = sqrt(ab) x0 + sqrt(1 - ab) eps
  VE variable  xt~ = x_t / sqrt(ab),  sigma = sqrt((1 - ab) / ab)
  probability-flow ODE in the VE variable:  d xt~ / d sigma = eps_hat(xt~, sigma)
  DDIM (eta = 0) is exactly explicit Euler in sigma on that ODE.
"""
import math
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
GALLERY = os.path.join(ROOT, "gallery")
B0, B1 = 0.1, 20.0
T_MIN = 1e-3


# ----------------------------------------------------------------------------- schedule
def alpha_bar(t):
    if isinstance(t, torch.Tensor):
        return torch.exp(-(B0 * t + 0.5 * (B1 - B0) * t * t))
    return math.exp(-(B0 * t + 0.5 * (B1 - B0) * t * t))


def sigma_of_t(t):
    ab = alpha_bar(t)
    if isinstance(ab, torch.Tensor):
        return torch.sqrt((1 - ab) / ab)
    return math.sqrt((1 - ab) / ab)


def t_of_sigma(sig):
    """invert sigma(t): -log ab = log(1+sigma^2) = b0 t + (b1-b0)/2 t^2"""
    L = math.log1p(sig * sig)
    a, b = 0.5 * (B1 - B0), B0
    return (-b + math.sqrt(b * b + 4 * a * L)) / (2 * a)


# ----------------------------------------------------------------------------- samplers
# eps_fn(x_ve, sigma_float) -> eps prediction, x_ve is the VE variable.

@torch.no_grad()
def ddim(eps_fn, z, n_steps, t_start=1.0, return_traj=False):
    """DDIM eta=0 with n_steps uniform-in-t steps from t=1 to t=0 (last step lands on sigma=0,
    i.e. it is the Tweedie denoise). z ~ N(0, I) is x_T in VP units."""
    ts = np.linspace(t_start, 0.0, n_steps + 1)
    sig = [sigma_of_t(float(t)) for t in ts]
    x = z * math.sqrt(1 + sig[0] ** 2)
    traj = []
    for i in range(n_steps):
        e = eps_fn(x, sig[i])
        x = x + (sig[i + 1] - sig[i]) * e
        if return_traj:
            traj.append(x)
    return (x, traj) if return_traj else x


@torch.no_grad()
def ode_rk4(eps_fn, z, n_steps, sigma_min=None):
    """Probability-flow ODE, classical RK4 in u = log sigma from sigma(1) to sigma(T_MIN),
    followed by the Tweedie denoise x0 = x - sigma_min * eps."""
    s_max = sigma_of_t(1.0)
    s_min = sigma_of_t(T_MIN) if sigma_min is None else sigma_min
    us = np.linspace(math.log(s_max), math.log(s_min), n_steps + 1)
    x = z * math.sqrt(1 + s_max ** 2)

    def f(x, u):
        s = math.exp(u)
        return s * eps_fn(x, s)

    for i in range(n_steps):
        u, h = us[i], us[i + 1] - us[i]
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, u + 0.5 * h)
        k4 = f(x + h * k3, u + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x - s_min * eps_fn(x, s_min)


@torch.no_grad()
def ddpm_frozen(eps_fn, z, n_steps, noise_seq):
    """Ancestral DDPM with n_steps uniform-in-t steps. noise_seq: tensor (n_steps, *event_shape)
    shared by every pixel (frozen noise). Written in VP units, converted at the boundaries."""
    ts = np.linspace(1.0, 0.0, n_steps + 1)
    x = z.clone()  # VP units
    for i in range(n_steps):
        ab, ab_next = alpha_bar(float(ts[i])), alpha_bar(float(ts[i + 1]))
        a = ab / ab_next
        beta = 1 - a
        sig = math.sqrt((1 - ab) / ab)
        e = eps_fn(x / math.sqrt(ab), sig)
        mean = (x - beta / math.sqrt(1 - ab) * e) / math.sqrt(a)
        if i < n_steps - 1:
            var = (1 - ab_next) / (1 - ab) * beta
            x = mean + math.sqrt(var) * noise_seq[i]
        else:
            x = mean
    return x


# ----------------------------------------------------------------------------- slices
def great_sphere(e0, u, v, alpha, beta, radius):
    """Exponential-map (azimuthal-equidistant) coordinates on the great 2-sphere through the
    orthonormal triple (e0, u, v), scaled to `radius`.  rho = |(alpha, beta)| is the geodesic
    angle from e0 in radians; every point has norm exactly `radius`.
    alpha, beta: 1-D tensors of equal length.  returns (n, d)"""
    rho = torch.sqrt(alpha ** 2 + beta ** 2)
    sinc = torch.where(rho > 1e-12, torch.sin(rho) / torch.clamp(rho, min=1e-300), torch.ones_like(rho))
    return radius * (torch.cos(rho)[:, None] * e0[None] + (sinc * alpha)[:, None] * u[None]
                     + (sinc * beta)[:, None] * v[None])


def orthonormal_triple(d, seed, device="cpu", dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(d, 3, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(m)
    return [q[:, i].to(device=device, dtype=dtype) for i in range(3)]


# ----------------------------------------------------------------------------- box counting
def boundary_mask(lab):
    """pixel is boundary if any 4-neighbour has a different label"""
    b = np.zeros(lab.shape, bool)
    d0 = lab[1:, :] != lab[:-1, :]
    d1 = lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= d0
    b[:-1, :] |= d0
    b[:, 1:] |= d1
    b[:, :-1] |= d1
    return b


def box_count(mask, sizes=None):
    """returns sizes (px), counts of occupied boxes"""
    n = min(mask.shape)
    if sizes is None:
        sizes = [2 ** k for k in range(0, int(math.log2(n)))]
    counts = []
    for s in sizes:
        h, w = (mask.shape[0] // s) * s, (mask.shape[1] // s) * s
        m = mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        counts.append(int(m.sum()))
    return np.array(sizes), np.array(counts)


def fit_dimension(sizes, counts, lo=None, hi=None):
    """least-squares slope of log N vs log(1/eps) over sizes in [lo, hi]; returns D, stderr"""
    sizes, counts = np.asarray(sizes, float), np.asarray(counts, float)
    sel = counts > 0
    if lo is not None:
        sel &= sizes >= lo
    if hi is not None:
        sel &= sizes <= hi
    x, y = np.log(1 / sizes[sel]), np.log(counts[sel])
    if len(x) < 3:
        return float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    dof = max(len(x) - 2, 1)
    s2 = ((y - yhat) ** 2).sum() / dof
    se = math.sqrt(s2 / ((x - x.mean()) ** 2).sum())
    return float(coef[0]), se


# ----------------------------------------------------------------------------- video
def write_video(frames_dir, pattern, out_mp4, out_gif=None, fps=30, gif_fps=15, gif_width=540):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", "-preset", "slow", "-movflags", "+faststart", out_mp4], check=True)
    if out_gif:
        filt = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
                f"[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        os.path.join(frames_dir, pattern), "-vf", filt, out_gif], check=True)


def gpu_setup(frac=0.10):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
