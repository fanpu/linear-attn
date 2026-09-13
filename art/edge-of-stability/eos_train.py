"""Full-batch GD on a CIFAR-10 subset with dense sharpness logging (compute step).

Reproduces the Cohen et al. 2021 (arXiv:2103.00065) fc-tanh setup:
  first n CIFAR-10 train images, per-channel standardisation with full-CIFAR mean/std,
  3072-200-200-10 tanh MLP, PyTorch default init, MSE loss 0.5*||f(x)-onehot||^2 averaged
  over examples, constant step size eta, full batch.

Every step we log:
  * top-k Hessian eigenpairs by warm-started block subspace iteration + Rayleigh-Ritz on
    Hessian-vector products (forward-over-reverse autodiff, no explicit Hessian);
  * residual norms ||H u - lambda u|| so the estimate can be audited;
  * loss, accuracy, gradient norm, gradient components along u1, u2;
  * the oscillation coordinate x_t = <theta_t - thetabar_t, u1_t> (and y_t along u2_t),
    thetabar_t a centred moving average over 2*half+1 steps (ring buffer);
  * <theta_t - theta_{t-1}, u1_t>;
  * a count-sketch of theta_t - theta_0 (float64) for sliding-window trajectory PCA;
  * projections of theta_t onto (u1, u2) frozen at a few reference steps (to show why a
    fixed reference frame is only locally meaningful).
Periodically: a cold-start Lanczos estimate of lambda_max for validation, and a snapshot of
u1 for an eigenvector-rotation Gram matrix.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torchvision
from torch.func import grad_and_value, jvp, vmap, grad

p = argparse.ArgumentParser()
p.add_argument("--inv", type=float, required=True, help="2/eta, i.e. the stability threshold")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--k", type=int, default=4, help="block size for subspace iteration")
p.add_argument("--iters", type=int, default=2, help="subspace iterations per GD step")
p.add_argument("--half", type=int, default=10, help="moving-average half window")
p.add_argument("--sketch", type=int, default=1024)
p.add_argument("--lanczos-every", type=int, default=250)
p.add_argument("--lanczos-m", type=int, default=40)
p.add_argument("--snap-every", type=int, default=50)
p.add_argument("--ref-every", type=int, default=500)
p.add_argument("--dtype", default="float64")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--act", default="tanh")
p.add_argument("--out", required=True)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
dev = "cuda"
dt = getattr(torch, args.dtype)
torch.manual_seed(args.seed)
eta = 2.0 / args.inv

# ---------------- data ----------------
ds = torchvision.datasets.CIFAR10(root="/home/fzeng/ml/research/art/data", train=True, download=False)
full = ds.data.astype(np.float64) / 255.0
mean = full.mean(axis=(0, 1, 2))
std = full.std(axis=(0, 1, 2))
Xn = (full[: args.n] - mean) / std  # (n,32,32,3)
Xn = np.transpose(Xn, (0, 3, 1, 2)).reshape(args.n, -1)  # CHW flatten, as nn.Flatten
X = torch.tensor(Xn, dtype=dt, device=dev)
yl = torch.tensor(np.array(ds.targets[: args.n]), device=dev)
Y = torch.nn.functional.one_hot(yl, 10).to(dt)

# ---------------- model (flat parameter vector) ----------------
shapes = [(200, 3072), (200,), (200, 200), (200,), (10, 200), (10,)]
fan = [3072, 3072, 200, 200, 200, 200]
sizes = [math.prod(s) for s in shapes]
P = sum(sizes)
g = torch.Generator().manual_seed(args.seed)
chunks = []
for s, f in zip(shapes, fan):  # PyTorch default Linear init == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    b = 1.0 / math.sqrt(f)
    chunks.append((torch.rand(math.prod(s), generator=g, dtype=torch.float64) * 2 - 1) * b)
theta = torch.cat(chunks).to(dev, dt)
act = {"tanh": torch.tanh, "elu": torch.nn.functional.elu}[args.act]


def forward(th):
    ps, o = [], 0
    for s, z in zip(shapes, sizes):
        ps.append(th[o:o + z].view(s))
        o += z
    h = act(X @ ps[0].T + ps[1])
    h = act(h @ ps[2].T + ps[3])
    return h @ ps[4].T + ps[5]


def loss_fn(th):
    out = forward(th)
    l = 0.5 * ((out - Y) ** 2).sum(1).mean()
    return l, out


gv = grad_and_value(loss_fn, has_aux=True)
gonly = grad(lambda th: loss_fn(th)[0])


def hvp(th, v):
    return jvp(gonly, (th,), (v,))[1]


hvp_batch = vmap(hvp, in_dims=(None, 0))


def lanczos_top(th, m):
    """Cold-start Lanczos with full reorthogonalisation; returns top-3 Ritz values."""
    Q = torch.zeros(m + 1, P, dtype=dt, device=dev)
    q = torch.randn(P, dtype=dt, device=dev)
    Q[0] = q / q.norm()
    al, be = [], []
    for j in range(m):
        w = hvp(th, Q[j])
        a = torch.dot(w, Q[j])
        w = w - Q[: j + 1].T @ (Q[: j + 1] @ w)
        w = w - Q[: j + 1].T @ (Q[: j + 1] @ w)
        b = w.norm()
        al.append(a.item())
        be.append(b.item())
        Q[j + 1] = w / b
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    return ev[:3]


# count sketch
sg = torch.Generator(device="cpu").manual_seed(1234)
sk_idx = torch.randint(0, args.sketch, (P,), generator=sg).to(dev)
sk_sign = (torch.randint(0, 2, (P,), generator=sg) * 2 - 1).to(dev, dt)


def sketch(v):
    return torch.zeros(args.sketch, dtype=dt, device=dev).index_add_(0, sk_idx, v * sk_sign)


# ---------------- buffers ----------------
T_ = args.steps
k = args.k
L = {
    "loss": np.zeros(T_), "acc": np.zeros(T_), "gnorm": np.zeros(T_),
    "evals": np.zeros((T_, k)), "resid": np.zeros((T_, k)),
    "g_u1": np.zeros(T_), "g_u2": np.zeros(T_),
    "dtheta_u1": np.full(T_, np.nan), "u1_overlap_prev": np.full(T_, np.nan),
    "x": np.full(T_, np.nan), "y": np.full(T_, np.nan),
    "sketch": np.zeros((T_, args.sketch)),
    "wnorm": np.zeros(T_),
}
nref = T_ // args.ref_every + 1
ref_steps = []
ref_U = torch.zeros(nref, 2, P, dtype=dt, device=dev)
L["refproj"] = np.full((T_, nref, 2), np.nan)
lanc_steps, lanc_vals = [], []
snap_steps, snaps = [], []

W = 2 * args.half + 1
ring_th = torch.zeros(W, P, dtype=dt, device=dev)
ring_u = torch.zeros(W, 2, P, dtype=dt, device=dev)
ring_sum = torch.zeros(P, dtype=dt, device=dev)
theta0 = theta.clone()

V = torch.linalg.qr(torch.randn(P, k, dtype=dt, device=dev))[0].T  # (k,P)
# burn-in at init so step-0 eigenpairs are converged
for _ in range(50):
    Wm = hvp_batch(theta, V)
    Tm = V @ Wm.T
    ev, Qm = torch.linalg.eigh(0.5 * (Tm + Tm.T))
    V = torch.linalg.qr(Wm.T)[0].T
prev_u1 = None
t0 = time.time()
diverged = False
for t in range(T_):
    # ---- eigenpairs at theta_t ----
    for it in range(args.iters):
        Wm = hvp_batch(theta, V)
        Tm = V @ Wm.T
        ev, Qm = torch.linalg.eigh(0.5 * (Tm + Tm.T))
        order = torch.argsort(ev, descending=True)
        ev, Qm = ev[order], Qm[:, order]
        V = Qm.T @ V
        Wm = Qm.T @ Wm
        if it < args.iters - 1:
            V = torch.linalg.qr(Wm.T)[0].T
    res = (Wm - ev[:, None] * V).norm(dim=1)
    u = V[:2].clone()
    if prev_u1 is not None:
        if torch.dot(u[0], prev_u1[0]) < 0:
            u[0] = -u[0]
        if torch.dot(u[1], prev_u1[1]) < 0:
            u[1] = -u[1]
        L["u1_overlap_prev"][t] = torch.dot(u[0], prev_u1[0]).abs().item()
        L["dtheta_u1"][t] = torch.dot(theta - prev_theta, u[0]).item()
    V = torch.linalg.qr(Wm.T)[0].T  # warm start for next step (one free power step)
    # keep sign continuity in the warm start too
    # ---- gradient ----
    gr, (lv, out) = gv(theta)
    if not torch.isfinite(lv):
        diverged = True
        print("diverged at", t)
        break
    L["loss"][t] = lv.item()
    L["acc"][t] = (out.argmax(1) == yl).to(dt).mean().item()
    L["gnorm"][t] = gr.norm().item()
    L["evals"][t] = ev.cpu().numpy()
    L["resid"][t] = res.cpu().numpy()
    L["g_u1"][t] = torch.dot(gr, u[0]).item()
    L["g_u2"][t] = torch.dot(gr, u[1]).item()
    L["sketch"][t] = sketch(theta - theta0).cpu().numpy()
    L["wnorm"][t] = theta.norm().item()
    # ---- frozen-reference projections ----
    if t % args.ref_every == 0:
        r = len(ref_steps)
        ref_U[r] = u
        ref_steps.append(t)
    nr = len(ref_steps)
    L["refproj"][t, :nr] = (ref_U[:nr] @ theta).cpu().numpy()
    # ---- centred moving-average oscillation coordinate ----
    ring_sum += theta - ring_th[t % W] if t >= W else theta
    ring_th[t % W] = theta
    ring_u[t % W] = u
    if t >= W - 1:
        c = t - args.half
        tb = ring_sum / W
        thc = ring_th[c % W]
        L["x"][c] = torch.dot(thc - tb, ring_u[c % W][0]).item()
        L["y"][c] = torch.dot(thc - tb, ring_u[c % W][1]).item()
    # ---- occasional checks ----
    if t % args.lanczos_every == 0:
        lv3 = lanczos_top(theta, args.lanczos_m)
        lanc_steps.append(t)
        lanc_vals.append(lv3)
    if t % args.snap_every == 0:
        snap_steps.append(t)
        snaps.append(u[0].float().cpu())
    prev_u1 = u
    prev_theta = theta
    # ---- GD step ----
    theta = theta - eta * gr
    if t % 200 == 0 or t == T_ - 1:
        el = time.time() - t0
        print(f"t={t} loss={lv.item():.4f} acc={L['acc'][t]:.3f} lam={ev[0].item():.2f} "
              f"(2/eta={args.inv}) res={res[0].item():.2e} lanczos={lanc_vals[-1][0]:.2f} "
              f"x={L['x'][max(c if t >= W - 1 else 0, 0)]:.2e} {el:.0f}s", flush=True)

S = torch.stack(snaps)
gram = (S @ S.T).abs().numpy()
wall = time.time() - t0
np.savez_compressed(
    args.out, **L, ref_steps=np.array(ref_steps), lanc_steps=np.array(lanc_steps),
    lanc_vals=np.array(lanc_vals), snap_steps=np.array(snap_steps), u1_gram=gram,
    eta=eta, inv=args.inv, diverged=diverged, steps_done=t + (0 if diverged else 1),
    meta=json.dumps(vars(args) | {"P": P, "wall_s": wall}),
)
print("saved", args.out, f"wall={wall:.0f}s")
