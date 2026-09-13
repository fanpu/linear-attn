"""Hero animation: a one-layer linear self-attention layer crystallizing into preconditioned gradient descent.

Data: cache/gradflow_ar1_d20_N40_s0.npz -- gradient flow on the exact population loss from Zhang-Frei-Bartlett's
init (lsa_gradflow.py).  Left: the key-query matrix W_KQ.  Middle: value-projection W_PV.  Right: in-context
loss dropping onto the closed-form global minimum, and learned preconditioner entries landing on Gamma^{-1}.

  .venv/bin/python 08-icl-linear-attention/render_hero.py
"""
import os, pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import style

HERE = pathlib.Path(__file__).parent
FR = HERE / "_frames" / "hero"
FR.mkdir(parents=True, exist_ok=True)
z = np.load(HERE / "cache" / "gradflow_ar1_d20_N40_s0.npz")
t, Wkq, Wpv, loss = z["t"], z["Wkq"], z["Wpv"], z["loss"]
Lstar, Wkq_s, Wpv_s = float(z["loss_star"]), z["Wkq_star"], z["Wpv_star"]
d = Wkq.shape[1] - 1
Gi = np.linalg.inv(z["Gam"])
B = Wpv[:, d, d][:, None, None] * Wkq[:, :d, :d]                 # effective preconditioner, -> Gamma^{-1}

style.use_dark()
cmap = style.diverging_dark()
FPS = 30
# frame times: log-uniform in t over the interesting window, then a hold on the converged state
tt = np.concatenate([np.geomspace(0.02, 400, 330), np.full(60, t[-1])])
ONLY = [int(v) for v in os.environ.get("ONLY", "").split(",") if v]
vmax = np.abs(Wkq_s).max()
amp = lambda M, v: np.sign(M) * (np.abs(M) / v) ** 0.55          # perceptual boost so faint noise is visible


def at(arr, tq):
    i = np.searchsorted(t, tq)
    if i <= 0:
        return arr[0]
    if i >= len(t):
        return arr[-1]
    f = (np.log(tq) - np.log(t[i - 1])) / (np.log(t[i]) - np.log(t[i - 1])) if t[i - 1] > 0 else 1.0
    return (1 - f) * arr[i - 1] + f * arr[i]


def matrix(ax, M, v, title, sub):
    ax.imshow(amp(M, v), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    n = M.shape[0]
    for k in range(n + 1):  # hairline cell grid
        ax.axhline(k - 0.5, color="#1b1e26", lw=0.7); ax.axvline(k - 0.5, color="#1b1e26", lw=0.7)
    ax.axhline(n - 1.5, color="#3a3f4b", lw=1.0); ax.axvline(n - 1.5, color="#3a3f4b", lw=1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=style.NIGHT_INK, fontsize=15, loc="left", pad=26)
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=style.NIGHT_MUTED, fontsize=11, va="bottom")


frames = ONLY or range(len(tt))
for fi in frames:
    tq = tt[fi]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.text(0.04, 0.935, "A transformer layer learns to run gradient descent", fontsize=30, fontweight="bold", color=style.NIGHT_INK)
    fig.text(0.04, 0.895, "One linear self-attention layer, trained on random linear-regression prompts ($d=20$, $N=40$ examples, correlated inputs), "
             "flows from random weights to the closed-form optimum of Zhang, Frei & Bartlett (2024).", fontsize=14, color=style.NIGHT_MUTED)
    gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.97, top=0.80, bottom=0.08, width_ratios=[1.25, 0.62, 1.05], wspace=0.14, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    matrix(ax, at(Wkq, tq), vmax, "key–query matrix  $W^{KQ}$", "top-left block $\\to\\ \\Gamma^{-1}$ (a learned preconditioner)")
    bx = fig.add_subplot(gs[0, 1])
    matrix(bx, at(Wpv, tq), np.abs(Wpv_s).max(), "value matrix  $W^{PV}$", "one entry: the output scale")
    # target thumbnail
    cxm = fig.add_subplot(gs[1, 1])
    matrix(cxm, Wkq_s, vmax, "theory: $W^{KQ}_*$", "closed form, Thm 4.1")
    # loss
    lx = fig.add_subplot(gs[0, 2])
    m = t <= tq
    lx.axhline(Lstar, color="#ffd08a", lw=1.2)
    lx.text(2800, Lstar * 0.955, f"closed-form optimum  $L_* = {Lstar:.3f}$", color="#ffd08a", fontsize=12, ha="right", va="top")
    lx.plot(t[1:], loss[1:], color=style.GLOW, lw=1.0, alpha=0.18)
    lx.plot(t[1:][m[1:]], loss[1:][m[1:]], color=style.GLOW, lw=2.6)
    lq = at(loss, tq)
    lx.scatter([tq], [lq], s=160, color=style.GLOW, alpha=0.25, lw=0); lx.scatter([tq], [lq], s=40, color=style.GLOW, lw=0)
    lx.set_xscale("log"); lx.set_yscale("log"); lx.set_xlim(0.02, 3000); lx.set_ylim(Lstar * 0.85, loss[0] * 1.15)
    lx.set_xlabel("training time (gradient flow)", fontsize=12)
    lx.set_title("in-context loss", color=style.NIGHT_INK, fontsize=15, loc="left")
    lx.set_yticks([2.5, 4, 6, 10]); lx.set_yticklabels(["2.5", "4", "6", "10"]); lx.minorticks_off()
    # entries of learned preconditioner vs theory
    sx = fig.add_subplot(gs[1, 2])
    Bq = at(B, tq)
    lim = (Gi.min() * 1.2, Gi.max() * 1.1)
    sx.plot(lim, lim, color="#ffd08a", lw=1.2)
    off = ~np.eye(d, dtype=bool)
    sx.scatter(Gi[off], Bq[off], s=9, color="#5aa9e6", alpha=0.55, lw=0)
    sx.scatter(np.diag(Gi), np.diag(Bq), s=26, color="#ffb070", alpha=0.95, lw=0)
    sx.set_xlim(*lim); sx.set_ylim(lim[0] - 0.25, lim[1] + 0.25)
    sx.set_xlabel("theory  $(\\Gamma^{-1})_{ij}$", fontsize=12); sx.set_ylabel("learned", fontsize=12)
    sx.set_title("every entry of the learned preconditioner", color=style.NIGHT_INK, fontsize=15, loc="left")
    sx.text(0.97, 0.06, "diagonal", color="#ffb070", transform=sx.transAxes, ha="right", fontsize=11, fontweight="bold")
    sx.text(0.97, 0.15, "off-diagonal", color="#5aa9e6", transform=sx.transAxes, ha="right", fontsize=11, fontweight="bold")
    err = np.linalg.norm(Bq - Gi) / np.linalg.norm(Gi)
    fig.text(0.04, 0.035, f"t = {tq:8.2f}      relative distance to theory  {err:6.1%}", fontsize=13, color=style.NIGHT_INK, family="DejaVu Sans Mono")
    fig.savefig(FR / f"f{fi:04d}.png", facecolor=style.NIGHT)
    plt.close(fig)

if not ONLY:
    out = HERE / "figures"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "f%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "scale=1600:-2", str(out / "hero.mp4")], check=True)
    subprocess.run(["cp", str(FR / f"f{len(tt)-1:04d}.png"), str(out / "hero_final.png")], check=True)
    print("wrote hero.mp4")
