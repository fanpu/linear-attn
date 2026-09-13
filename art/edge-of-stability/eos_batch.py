"""Batched full-batch GD at several step sizes in ONE process (compute step).

Same measurements as eos_train.py (the single-run prototype) but M independent networks, one per
step size, are trained side by side as a batch of parameter vectors theta in R^{M x P}. Because the
total objective is sum_m L(theta_m), its Hessian is block diagonal, so one reverse-over-reverse
Hessian-vector product with a (M, P) direction returns all M per-model HVPs at once. This lets the
whole learning-rate sweep share a single slot on the shared GPU.

Setup (Cohen et al. 2021, arXiv:2103.00065, fc-tanh): first n CIFAR-10 train images, per-channel
standardised with full-CIFAR mean/std, 3072-200-200-10 tanh MLP, PyTorch default Linear init
(identical init for every model: same seed), loss 0.5*||f(x)-onehot||^2 averaged over examples,
constant eta per model, full batch, float32 (float64 is ~17x slower on this GPU; see README).

Logged per model m and step t (arrays shaped (T, M, ...)):
  evals, resid     top-k Hessian eigenvalues at theta_t by warm-started block subspace iteration
                   (1 iteration per refresh, refresh every --eig-every steps; between refreshes the
                   last eigenvectors are reused = per-window eigenvectors) + Rayleigh-Ritz residuals
  loss, acc, gnorm, g_u (gradient along u1..u3)
  x, y, z          oscillation coordinates <theta_t - thetabar_t, u_j(t)>, thetabar_t a centred
                   moving average over 2*half+1 steps, u_j(t) the CURRENT eigenvector (sign-tracked)
  dtheta_u1        <theta_t - theta_{t-1}, u1(t)>
  u1_overlap_prev  |<u1(t), u1(t-E)>| (eigenvector rotation per refresh)
  sketch, u1_sketch  count-sketches of theta_t - theta_0 and u1(t) (optional, --sketch > 0)
Periodically: cold-start subspace iteration (--check-iters iterations from random) as an
independent sharpness check, and snapshots of u1 for an eigenvector-rotation Gram matrix.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torchvision

p = argparse.ArgumentParser()
p.add_argument("--invs", type=str, required=True, help="comma list of 2/eta values")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--k", type=int, default=3)
p.add_argument("--eig-every", type=int, default=1)
p.add_argument("--half", type=int, default=10)
p.add_argument("--sketch", type=int, default=0)
p.add_argument("--check-every", type=int, default=500)
p.add_argument("--check-iters", type=int, default=60)
p.add_argument("--snap-every", type=int, default=100)
p.add_argument("--save-every", type=int, default=250)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dtype", default="float32")
p.add_argument("--mem", type=float, default=0.10)
p.add_argument("--out", required=True)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(args.mem)
dev, dt = "cuda", getattr(torch, args.dtype)
invs = [float(v) for v in args.invs.split(",")]
M = len(invs)
eta = torch.tensor([2.0 / v for v in invs], dtype=dt, device=dev)

# ---------------- data ----------------
ds = torchvision.datasets.CIFAR10(root="/home/fzeng/ml/research/art/data", train=True, download=False)
full = ds.data.astype(np.float64) / 255.0
mean, std = full.mean(axis=(0, 1, 2)), full.std(axis=(0, 1, 2))
Xn = np.transpose((full[: args.n] - mean) / std, (0, 3, 1, 2)).reshape(args.n, -1)
X = torch.tensor(Xn, dtype=dt, device=dev)
yl = torch.tensor(np.array(ds.targets[: args.n]), device=dev)
Y = torch.nn.functional.one_hot(yl, 10).to(dt)
del full

# ---------------- model ----------------
shapes = [(200, 3072), (200,), (200, 200), (200,), (10, 200), (10,)]
fan = [3072, 3072, 200, 200, 200, 200]
sizes = [math.prod(s) for s in shapes]
P = sum(sizes)
g = torch.Generator().manual_seed(args.seed)
th0 = torch.cat([(torch.rand(z, generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(f)
                 for z, f in zip(sizes, fan)])
theta = th0.to(dev, dt).repeat(M, 1).contiguous()  # (M, P)
theta_init = theta.clone()


def forward(th):  # th (M, P) -> (M, n, 10)
    ps, o = [], 0
    for s, z in zip(shapes, sizes):
        ps.append(th[:, o:o + z].reshape(M, *s))
        o += z
    h = torch.tanh(X @ ps[0].transpose(1, 2) + ps[1][:, None])
    h = torch.tanh(h @ ps[2].transpose(1, 2) + ps[3][:, None])
    return h @ ps[4].transpose(1, 2) + ps[5][:, None]


def losses(th):
    out = forward(th)
    return 0.5 * ((out - Y) ** 2).sum(-1).mean(-1), out  # (M,)


def grad_hvps(th, V):
    """Gradient (M,P) and HVPs (M,k,P) of the block-diagonal sum objective."""
    t = th.detach().requires_grad_(True)
    lv, out = losses(t)
    if V is None:
        gr = torch.autograd.grad(lv.sum(), t)[0]
        return lv.detach(), out.detach(), gr, None
    gr = torch.autograd.grad(lv.sum(), t, create_graph=True)[0]
    kk = V.shape[1]
    HV = torch.stack([torch.autograd.grad(gr, t, V[:, j], retain_graph=(j < kk - 1))[0]
                      for j in range(kk)], 1)
    return lv.detach(), out.detach(), gr.detach(), HV


def rayleigh_ritz(V, HV):
    Tm = torch.einsum("mip,mjp->mij", V, HV)
    ev, Q = torch.linalg.eigh((0.5 * (Tm + Tm.transpose(1, 2))).cpu().double())  # tiny: CPU
    ev, Q = ev.flip(-1).to(dev, dt), Q.flip(-1).to(dev, dt)  # descending
    V = torch.einsum("mji,mjp->mip", Q, V)
    HV = torch.einsum("mji,mjp->mip", Q, HV)
    return ev, V, HV


def orth(W):  # (M,k,P) -> orthonormal rows
    return torch.linalg.qr(W.transpose(1, 2))[0].transpose(1, 2).contiguous()


def cold_check(th):
    Vc = orth(torch.randn(M, args.k, P, dtype=dt, device=dev))
    for _ in range(args.check_iters):
        HV = grad_hvps(th, Vc)[3]
        ev, Vc, HV = rayleigh_ritz(Vc, HV)
        Vc = orth(HV)
    return ev.cpu().numpy()


S = args.sketch
if S:
    sg = torch.Generator().manual_seed(1234)
    sk_idx = torch.randint(0, S, (P,), generator=sg).to(dev)
    sk_sign = (torch.randint(0, 2, (P,), generator=sg) * 2 - 1).to(dev, dt)

    def sketch(v):  # (M,P) -> (M,S)
        return torch.zeros(M, S, dtype=dt, device=dev).index_add_(1, sk_idx, v * sk_sign)

T_, k, E = args.steps, args.k, args.eig_every
L = {n_: np.full((T_, M), np.nan, np.float32) for n_ in
     ["loss", "acc", "gnorm", "x", "y", "z", "dtheta_u1", "u1_overlap_prev", "wdist"]}
L["evals"] = np.full((T_, M, k), np.nan, np.float32)
L["resid"] = np.full((T_, M, k), np.nan, np.float32)
L["g_u"] = np.full((T_, M, k), np.nan, np.float32)
L["eig_fresh"] = np.zeros(T_, bool)
if S:
    L["sketch"] = np.zeros((T_, M, S), np.float32)
    L["u1_sketch"] = np.zeros((T_, M, S), np.float32)
checks_t, checks_v, snap_t, snaps = [], [], [], []

W = 2 * args.half + 1
ring_th = torch.zeros(W, M, P, dtype=dt, device=dev)
ring_u = torch.zeros(W, M, k, P, dtype=dt, device=dev)
ring_sum = torch.zeros(M, P, dtype=torch.float64, device=dev)  # float64 running sum (no drift)

V = orth(torch.randn(M, k, P, dtype=dt, device=dev))
for _ in range(60):  # burn-in so step-0 eigenpairs are converged
    HV = grad_hvps(theta, V)[3]
    ev, V, HV = rayleigh_ritz(V, HV)
    V = orth(HV)
u = None
alive = torch.ones(M, dtype=torch.bool, device=dev)
diverged_at = np.full(M, -1)
t0 = time.time()


def save(t, final=False):
    gram = None
    if snaps:
        Sn = torch.stack(snaps)  # (ns, M, P) cpu
        gram = torch.einsum("amp,bmp->mab", Sn, Sn).abs().numpy()
    np.savez_compressed(
        args.out, **L, checks_t=np.array(checks_t), checks_v=np.array(checks_v),
        snap_t=np.array(snap_t), u1_gram=gram if gram is not None else np.zeros(0),
        invs=np.array(invs), eta=eta.cpu().numpy(), diverged_at=diverged_at,
        steps_done=t + 1, final=final,
        meta=json.dumps(vars(args) | {"P": P, "wall_s": time.time() - t0, "M": M}))


for t in range(T_):
    fresh = t % E == 0
    lv, out, gr, HV = grad_hvps(theta, V if fresh else None)
    if fresh:
        ev, Vr, HVr = rayleigh_ritz(V, HV)
        res = (HVr - ev[..., None] * Vr).norm(dim=-1)
        unew = Vr.clone()
        if u is not None:
            sgn = torch.sign(torch.einsum("mjp,mjp->mj", unew, u))
            sgn[sgn == 0] = 1
            unew = unew * sgn[..., None]
            L["u1_overlap_prev"][t] = torch.einsum("mp,mp->m", unew[:, 0], u[:, 0]).abs().cpu().numpy()
        u = unew
        V = orth(HVr)  # warm start for next refresh (one power step)
        L["evals"][t] = ev.cpu().numpy()
        L["resid"][t] = res.cpu().numpy()
        L["eig_fresh"][t] = True
    bad = ~torch.isfinite(lv)
    if bad.any():
        for m in torch.nonzero(bad & alive).flatten().tolist():
            diverged_at[m] = t
            print(f"model {m} (2/eta={invs[m]}) diverged at {t}", flush=True)
        alive &= ~bad
        theta[~alive] = theta_init[~alive]  # park diverged models at init with eta=0 (logged NaN)
        eta[~alive] = 0
        gr[~alive] = 0
        lv = torch.where(alive, lv, torch.nan)
    L["loss"][t] = lv.cpu().numpy()
    L["acc"][t] = (out.argmax(-1) == yl).to(dt).mean(-1).cpu().numpy()
    L["gnorm"][t] = gr.norm(dim=-1).cpu().numpy()
    L["g_u"][t] = torch.einsum("mp,mjp->mj", gr, u).cpu().numpy()
    L["wdist"][t] = (theta - theta_init).norm(dim=-1).cpu().numpy()
    if t > 0:
        L["dtheta_u1"][t] = torch.einsum("mp,mp->m", theta - prev_theta, u[:, 0]).cpu().numpy()
    if S:
        L["sketch"][t] = sketch(theta - theta_init).cpu().numpy()
        L["u1_sketch"][t] = sketch(u[:, 0]).cpu().numpy()
    # centred moving-average oscillation coordinate
    if t >= W:
        ring_sum -= ring_th[t % W].double()
    ring_sum += theta.double()
    ring_th[t % W] = theta
    ring_u[t % W] = u
    if t >= W - 1:
        c = t - args.half
        d = ring_th[c % W] - (ring_sum / W).to(dt)
        xyz = torch.einsum("mp,mjp->mj", d, ring_u[c % W]).cpu().numpy()
        L["x"][c], L["y"][c], L["z"][c] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if t % args.check_every == 0:
        checks_t.append(t)
        checks_v.append(cold_check(theta))
    if args.snap_every and t % args.snap_every == 0:
        snap_t.append(t)
        snaps.append(u[:, 0].float().cpu())
    prev_theta = theta
    theta = theta - eta[:, None] * gr
    if t % 100 == 0:
        lam = " ".join(f"{a:.0f}" for a in L["evals"][t - (t % E), :, 0])
        print(f"t={t} {time.time() - t0:.0f}s lam1=[{lam}] loss={np.nanmean(L['loss'][t]):.4f}", flush=True)
    if (t % args.save_every == 0 and t > 0):
        save(t)
save(T_ - 1, final=True)
print("saved", args.out, f"wall={time.time() - t0:.0f}s")
