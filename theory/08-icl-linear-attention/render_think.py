"""'Watching it think' animation (d = 2): context points arrive one at a time; each in-context algorithm's implied
weight vector moves through w-space.  All trajectories are exact closed forms of the algorithms:
    ridge / RLS (Bayes-optimal here), trained 1-layer LSA = x_q^T Gamma^-1 (1/t) sum y x (ZFB optimum, verified in
    lsa_gradflow.py / train_lsa1.py), DeltaNet rule = delta rule with k = x, v = y (one pass of online SGD / LMS), beta = 0.15.
The bottom-right panel overlays the expected risk of each algorithm (closed form for ridge-MC / LSA; MC for LMS).

  .venv/bin/python 08-icl-linear-attention/render_think.py
"""
import pathlib, subprocess, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import style
from icl_core import zfb_gamma, risk_precond_gd1, rls_prefix_w, lms_prefix_w

HERE = pathlib.Path(__file__).parent
FR = HERE / "_frames" / "think"
FR.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(163)  # a typical draw (chosen so the per-draw ordering matches the average)

d, T, sigma, Ntrain, beta = 2, 24, 0.35, 20, 0.15
th = np.deg2rad(35)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Lam = R @ np.diag([2.2, 0.45]) @ R.T
w_true = np.array([0.9, -1.1])
X = rng.multivariate_normal(np.zeros(2), Lam, size=T)
y = X @ w_true + sigma * rng.standard_normal(T)

Xt, yt = torch.tensor(X)[None], torch.tensor(y)[None]
W_rls = rls_prefix_w(Xt, yt, lam=sigma**2)[0].numpy()                       # [T+1, 2]
W_lms = lms_prefix_w(Xt, yt, beta=beta, normalize=False)[0].numpy()
Gi = np.linalg.inv(zfb_gamma(torch.tensor(Lam), Ntrain).numpy())
cs = np.cumsum(y[:, None] * X, 0) / np.arange(1, T + 1)[:, None]
W_lsa = np.vstack([np.zeros(2), cs @ Gi.T])

# expected excess risk curves E||w_hat - w||_Lam^2 (w ~ N(0, I)): closed form for LSA, Monte Carlo for ridge & LMS
ts = np.arange(1, T + 1)
r_lsa = np.array([risk_precond_gd1(torch.tensor(Gi), torch.tensor(Lam), int(t), sigma=sigma).item() for t in ts])
B = 200_000
L = np.linalg.cholesky(Lam)
Xm = torch.tensor(rng.standard_normal((B, T, 2)) @ L.T)
wm = torch.tensor(rng.standard_normal((B, 2)))
ym = torch.einsum("btd,bd->bt", Xm, wm) + sigma * torch.tensor(rng.standard_normal((B, T)))
Lt = torch.tensor(Lam)
ex = lambda W: torch.einsum("btd,de,bte->bt", W[:, 1:] - wm[:, None], Lt, W[:, 1:] - wm[:, None]).mean(0).numpy()
r_rls, r_lms = ex(rls_prefix_w(Xm, ym, sigma**2)), ex(lms_prefix_w(Xm, ym, beta, normalize=False))
r0 = np.trace(Lam)

ALG = [("Ridge / RLS", W_rls, r_rls, style.NIGHT_INK), ("Linear attention (1 layer, trained)", W_lsa, r_lsa, "#4f9bf0"),
       ("DeltaNet rule (online SGD)", W_lms, r_lms, "#2fcf94")]

style.use_dark()
FPS, PER = 24, 11
HOLD = 36
nframes = T * PER + HOLD
ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))
cmap = style.diverging_dark()
ylim = np.abs(y).max()


def interp(W, s):
    i = int(np.floor(s)); f = s - i
    if i >= T:
        return W[T]
    return (1 - f) * W[i] + f * W[i + 1]


allp = np.vstack([W_rls, W_lsa, W_lms, w_true[None]])
lo, hi = allp.min(0) - 0.35, allp.max(0) + 0.35
ctr, half = (lo + hi) / 2, max(hi - lo) / 2
XL, YL = (ctr[0] - half * 1.7, ctr[0] + half * 1.7), (ctr[1] - half, ctr[1] + half)
grid = np.linspace(-4, 4, 200)
GX, GY = np.meshgrid(grid, grid)
import os
ONLY = [int(v) for v in os.environ.get('ONLY', '').split(',') if v]
for fi in (ONLY or range(nframes)):
    s = min(T, fi / PER)
    k = int(np.floor(s)); u = ease(s - k); sf = k + u if k < T else T
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.text(0.045, 0.93, "Watching a transformer think", fontsize=26, fontweight="bold", color=style.NIGHT_INK)
    fig.text(0.045, 0.895, "Context pairs $(x_t, y_t)$ arrive one by one. Each algorithm turns them into an estimate $\\hat w$ of the hidden regression vector.",
             fontsize=13, color=style.NIGHT_MUTED)
    # ---- data space
    ax = fig.add_axes([0.045, 0.08, 0.40, 0.76])
    wr = interp(W_rls, sf)
    ax.contourf(GX, GY, GX * wr[0] + GY * wr[1], levels=np.linspace(-7, 7, 29), cmap=cmap, alpha=0.32, extend="both")
    ax.contour(GX, GY, GX * wr[0] + GY * wr[1], levels=np.linspace(-7, 7, 15), colors=[style.NIGHT_RULE], linewidths=0.6)
    n_vis = min(T, k + 1)
    for j in range(n_vis):
        a = 1.0 if j < k else u
        ax.scatter(X[j, 0], X[j, 1], s=70, c=[cmap(0.5 + 0.5 * np.clip(y[j] / (0.6 * ylim), -1, 1))], edgecolors=style.NIGHT_INK, linewidths=1.2, alpha=a, zorder=3)
    if k < T:
        ax.scatter(X[k, 0], X[k, 1], s=70 + 900 * u, facecolors="none", edgecolors=style.NIGHT_INK, linewidths=1.5 * (1 - u), alpha=(1 - u), zorder=4)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-2.88, 2.75, "input space  $x \\in \\mathbb{R}^2$", color=style.NIGHT_INK, fontsize=13, fontweight="bold")
    ax.text(-2.88, 2.45, "colour = label $y$;  shading = ridge's current prediction", color=style.NIGHT_MUTED, fontsize=11)
    ax.text(2.88, -2.85, f"t = {min(T, k + (1 if u > 0.5 else 0))} examples", color=style.NIGHT_INK, fontsize=15, ha="right", fontweight="bold")
    # ---- w space
    bx = fig.add_axes([0.52, 0.40, 0.44, 0.46])
    tt = max(1, min(T, k + 1))
    Xc, yc = X[:tt], y[:tt]
    H = Xc.T @ Xc / tt
    wc = np.linalg.solve(H + 1e-9 * np.eye(2), Xc.T @ yc / tt)
    ev, evec = np.linalg.eigh(H)
    for lev in [0.01, 0.04, 0.1, 0.22, 0.45]:
        wid, hei = 2 * np.sqrt(2 * lev / ev)
        ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
        bx.add_patch(Ellipse(wc, wid, hei, angle=ang, fill=False, color="#3a3f4b", lw=0.9))
    bx.scatter(*w_true, marker="*", s=380, color="#ffd08a", zorder=5, edgecolors=style.NIGHT, linewidths=1.0)
    bx.annotate("true $w$", w_true, xytext=(-62, 14), textcoords="offset points", color="#ffd08a", fontsize=12, fontweight="bold")
    for name, W, _, col in ALG:
        m = int(np.floor(sf))
        path = np.vstack([W[: m + 1], interp(W, sf)])
        bx.plot(path[:, 0], path[:, 1], color=col, lw=2.2, alpha=0.9, zorder=4)
        p = interp(W, sf)
        bx.scatter(*p, s=160, color=col, alpha=0.25, zorder=5, linewidths=0)
        bx.scatter(*p, s=46, color=col, zorder=6, edgecolors=style.NIGHT, linewidths=1.0)
    bx.scatter(0, 0, s=18, color=style.NIGHT_MUTED, zorder=3)
    bx.set_xlim(*XL); bx.set_ylim(*YL); bx.set_aspect("equal")
    bx.set_xlabel("$w_1$"); bx.set_ylabel("$w_2$")
    bx.set_title("weight space: where each algorithm thinks $w$ is", color=style.NIGHT_INK, fontsize=13)
    bx.text(0.99, 0.02, "ellipses: level sets of the in-context least-squares loss", transform=bx.transAxes, ha="right",
            color=style.NIGHT_MUTED, fontsize=10)
    # ---- risk
    cx = fig.add_axes([0.56, 0.08, 0.27, 0.23])
    for name, W, r, col in ALG:
        cx.plot(ts, r / r0, color=col, lw=1.2, alpha=0.35)
        m = max(1, int(np.floor(sf)))
        cx.plot(ts[:m], (r / r0)[:m], color=col, lw=2.4)
        cx.annotate(name, (ts[-1], r[-1] / r0), xytext=(6, 0), textcoords="offset points", color=col, fontsize=11, va="center", fontweight="bold",
                    annotation_clip=False)
    cx.axvline(sf, color=style.NIGHT_MUTED, lw=0.8)
    cx.set_yscale("log"); cx.set_xlim(1, T); cx.set_ylim(0.004, 1.5)
    cx.set_xlabel("number of in-context examples $t$")
    cx.set_ylabel("expected error (normalized)")
    cx.set_title("average over all tasks (faint: full curve)", color=style.NIGHT_INK, fontsize=12)
    fig.savefig(FR / f"f{fi:04d}.png", facecolor=style.NIGHT)
    plt.close(fig)
    if fi % 50 == 0:
        print(fi, "/", nframes, flush=True)

if ONLY:
    raise SystemExit
out = HERE / "figures"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-vf", "scale=1600:-2", str(out / "think.mp4")], check=True)
print("wrote think.mp4")
