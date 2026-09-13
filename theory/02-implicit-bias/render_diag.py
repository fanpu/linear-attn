"""Figures for the diagonal-linear-network section.  Reads cache/diag_*.npz.

  figures/diag_sweep.png     distance to basis pursuit / min-L2 vs alpha, measured flow vs closed-form Q_alpha;
                             recovery error vs number of samples
  figures/diag_2d.mp4/.gif   the d=2, n=1 picture: trajectories for a sweep of alpha + the Q_alpha ball morphing
  figures/diag_stems.mp4/.gif  d=100 solution morphing from dense (kernel) to sparse (rich) as alpha shrinks
"""
import os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import style as S
from core import q_fun

S.light()
only = sys.argv[1:] or ["sweep", "2d", "stems"]
D = np.load("cache/diag_sparse.npz")
FPS = 30


def encode(fdir, name, gif_width=900, gif_fps=20, colors=128):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{fdir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", f"figures/{name}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{fdir}/f%05d.png", "-vf",
                    f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    f"figures/{name}.gif"], check=True)
    print("wrote", name, os.path.getsize(f"figures/{name}.gif") / 1e6, "MB gif")


# ============================================================================ static sweep
if "sweep" in only:
    a, Wf, Wq, bp, l2, ws = D["alphas"], D["W_flow"], D["W_q"], D["bp"], D["l2"], D["w_star"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]
    dq_bp = np.linalg.norm(Wq - bp, axis=1)
    dq_l2 = np.linalg.norm(Wq - l2, axis=1)
    ax.loglog(a, dq_bp, color=S.AQUA, lw=2, zorder=2)
    ax.loglog(a, dq_l2, color=S.BLUE, lw=2, zorder=2)
    sel = slice(0, None, 2)
    for curve, col in [(np.linalg.norm(Wf - bp, axis=1), S.AQUA), (np.linalg.norm(Wf - l2, axis=1), S.BLUE)]:
        ax.scatter(a[sel], np.maximum(curve[sel], 1e-12), s=34, color=col, edgecolor=S.PAPER, linewidth=1.5, zorder=3)
    ax.set_ylim(3e-11, 30)
    ax.set_xlabel(r"initialization scale $\alpha$")
    ax.set_ylabel("distance  $\\|w_\\infty - w_{\\mathrm{ref}}\\|_2$")
    ax.set_title("Where gradient flow lands, as a function of init scale")
    ax.text(1.2e-5, 6e-2, "distance to the min-$L_1$ solution\n(basis pursuit)", color=S.INK2, fontsize=10.5)
    ax.text(3.0, 2e-3, "distance to the\nmin-$L_2$ interpolant", color=S.INK2, fontsize=10.5, ha="left")
    ax.annotate("rich regime", (1.5e-5, 6e-11), color=S.MUTED, fontsize=10)
    ax.annotate("kernel regime", (2.0, 6e-11), color=S.MUTED, fontsize=10, ha="left")
    # legend chips for measured vs theory
    ax.scatter([2.0e-5], [1e-7], s=34, color=S.INK2, edgecolor=S.PAPER, lw=1.5)
    ax.text(3.2e-5, 1e-7, "dots: gradient flow on $w = u\\odot u - v\\odot v$ (measured)", fontsize=9.5, color=S.INK2, va="center")
    ax.plot([1.3e-5, 2.7e-5], [1e-8, 1e-8], color=S.INK2, lw=2)
    ax.text(3.2e-5, 1e-8, "lines: $\\arg\\min\\, Q_\\alpha(w)$ s.t. $Xw=y$ (closed form)", fontsize=9.5, color=S.INK2, va="center")
    err = np.abs(Wf - Wq).max()
    ax.text(3.2e-5, 1.5e-9, f"largest |flow − closed form| over all 57 values of α: {err:.0e}", fontsize=9, color=S.MUTED, va="center")

    ax = axs[1]
    ns, al, rec, bpe = D["ns"], D["a_list"], D["rec_err"], D["bp_err"]
    cols = [S.ALPHA_CMAP(v) for v in np.linspace(0.05, 0.95, len(al))]
    for i in range(len(al)):
        m = rec[i].mean(-1)
        ax.plot(ns, m, color=cols[i], lw=2)
        ax.plot([58, 63], [0.4 - 0.05 * i] * 2, color=cols[i], lw=2.5)
        ax.text(64.5, 0.4 - 0.05 * i, f"α = {al[i]:g}", va="center", fontsize=9.5, color=S.INK2)
    ax.plot(ns, bpe.mean(-1), color=S.INK, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("basis pursuit", (ns[4], bpe.mean(-1)[4]), xytext=(4, 10), textcoords="offset points", fontsize=9.5, color=S.INK2)
    ax.set_xlabel("number of measurements $n$  (d = 100, 5-sparse truth)")
    ax.set_ylabel(r"relative error  $\|w_\infty - w^\star\|/\|w^\star\|$")
    ax.set_title("Small init recovers the sparse truth from few samples")
    ax.set_xlim(10, 80); ax.set_ylim(-0.02, 1.02)
    fig.savefig("figures/diag_sweep.png", bbox_inches="tight")
    print("wrote diag_sweep.png")

# ============================================================================ 2D picture
if "2d" in only:
    T = np.load("cache/diag_toy.npz")
    x = T["x"][0]

    def path(alpha, npts=500):
        c = brentq(lambda c: x @ (2 * alpha ** 2 * np.sinh(c * x)) - 1, 0, 400)
        # parameterize by c in a way that spends points evenly along arclength
        cs = np.linspace(0, c, 6000)
        P = 2 * alpha ** 2 * np.sinh(np.outer(cs, x))
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        return np.array([np.interp(np.linspace(0, sl[-1], npts), sl, P[:, j]) for j in range(2)]).T

    def Qa(W, alpha):
        return alpha ** 2 * q_fun(W / alpha ** 2).sum(-1)

    XL, YL = (-0.1, 1.22), (-0.2, 0.53)
    gx = np.linspace(*XL, 520); gy = np.linspace(*YL, 440)
    GX, GY = np.meshgrid(gx, gy)
    G = np.stack([GX, GY], -1)
    l2pt = x / (x @ x)
    fan_alphas = np.logspace(-3, 1, 17)
    fan = [path(a_) for a_ in fan_alphas]
    # alpha schedule: ping-pong in log alpha
    nF = 300
    la = np.r_[np.linspace(1, -3, nF // 2), np.linspace(-3, 1, nF // 2)]
    la = 1 - 4 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, nF, endpoint=False)))  # smooth 1 -> -3 -> 1
    ends = np.array([path(10 ** v, 50)[-1] for v in np.linspace(-3, 1, 200)])

    fdir = "cache/frames_diag2d"
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    for fi, lv in enumerate(la):
        alpha = 10 ** lv
        fig.clf()
        ax = fig.add_axes([0.04, 0.08, 0.56, 0.84])
        ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(0, color=S.AXIS, lw=0.8, zorder=0); ax.axvline(0, color=S.AXIS, lw=0.8, zorder=0)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.25, 0.5])
        ax.text(XL[1] - 0.01, 0.015, "$w_1$", ha="right", va="bottom", color=S.INK2, fontsize=13)
        ax.text(0.015, YL[1] - 0.01, "$w_2$", ha="left", va="top", color=S.INK2, fontsize=13)
        # unit balls through the two reference solutions
        th = np.linspace(0, 2 * np.pi, 400)
        r2 = np.linalg.norm(l2pt)
        ax.plot(r2 * np.cos(th), r2 * np.sin(th), color=S.BLUE, lw=1.0, alpha=0.55)
        ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=S.AQUA, lw=1.0, alpha=0.75)
        # the solution line x.w = 1
        xs = np.array(XL)
        ax.plot(xs, (1 - x[0] * xs) / x[1], color=S.INK, lw=1.6)
        ax.text(0.86, 0.47, "every point on this line fits the data\n$w_1 + 0.4\\,w_2 = 1$", fontsize=10.5, color=S.INK2,
                ha="left", va="center")
        # fan of paths
        for a_, P in zip(fan_alphas, fan):
            ax.plot(P[:, 0], P[:, 1], color=S.ALPHA_CMAP((np.log10(a_) + 3) / 4), lw=0.9, alpha=0.35)
        # Q_alpha level set through the current limit point
        P = path(alpha)
        wend = P[-1]
        Qv = Qa(G, alpha)
        ax.contour(GX, GY, Qv, levels=[Qa(wend, alpha)], colors=[S.INK], linewidths=1.3, linestyles=[(0, (3, 2.5))])
        col = S.ALPHA_CMAP((lv + 3) / 4)
        S.glow_line(ax, P[:, 0], P[:, 1], col, lw=2.6, layers=4, spread=3, alpha=0.08)
        # particle racing along the path (loops every 50 frames)
        ph = (fi % 50) / 50
        k = int(ph ** 0.6 * (len(P) - 1))
        ax.scatter([P[k, 0]], [P[k, 1]], s=30, color=col, edgecolor=S.PAPER, lw=1.5, zorder=6)
        ax.scatter([wend[0]], [wend[1]], s=90, color=col, edgecolor=S.PAPER, lw=2, zorder=7)
        ax.scatter([l2pt[0]], [l2pt[1]], s=40, color=S.PAPER, edgecolor=S.BLUE, lw=1.6, zorder=5)
        ax.scatter([1], [0], s=40, color=S.PAPER, edgecolor=S.AQUA, lw=1.6, zorder=5)
        ax.annotate("min-$L_2$", l2pt, xytext=(-14, 12), textcoords="offset points", ha="right", color=S.INK2, fontsize=11)
        ax.annotate("min-$L_1$", (1, 0), xytext=(8, -16), textcoords="offset points", ha="left", color=S.INK2, fontsize=11)
        ax.annotate("start: $w=0$", (0, 0), xytext=(-8, -16), textcoords="offset points", ha="right", color=S.INK2, fontsize=10.5)
        ax.text(0.02, -0.17, "dashed: the level set of $Q_\\alpha$ that touches the line", color=S.INK2, fontsize=10)
        # ---- side panel
        fig.text(0.64, 0.87, "INIT SCALE", color=S.MUTED, fontsize=10.5)
        fig.text(0.64, 0.79, f"α = {alpha:.3g}" if alpha >= 0.01 else f"α = {alpha:.1e}", fontsize=30, family=S.MONO, color=S.INK)
        bx = fig.add_axes([0.66, 0.2, 0.3, 0.46])
        lg = np.linspace(-3, 1, 200)
        bx.plot(lg, ends[:, 1], color=S.INK2, lw=1.2)
        for j in range(len(lg) - 1):
            bx.plot(lg[j:j + 2], ends[j:j + 2, 1], color=S.ALPHA_CMAP((lg[j] + 3) / 4), lw=3)
        bx.scatter([lv], [wend[1]], s=80, color=col, edgecolor=S.PAPER, lw=2, zorder=5)
        bx.axhline(l2pt[1], color=S.BLUE, lw=0.8, ls=(0, (3, 3)))
        bx.axhline(0, color=S.AQUA, lw=0.8, ls=(0, (3, 3)))
        bx.text(-3, l2pt[1] + 0.012, "min-$L_2$ answer", color=S.INK2, fontsize=10)
        bx.text(1, 0.012, "min-$L_1$ answer (sparse)", color=S.INK2, fontsize=10, ha="right")
        bx.set_xticks([-3, -2, -1, 0, 1]); bx.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
        bx.set_xlabel("init scale α (log)")
        bx.set_ylabel("$w_2$ of the point GD lands on")
        bx.set_title("Same loss, same data, same optimizer:\nthe init scale picks the answer", fontsize=12.5)
        bx.set_ylim(-0.03, 0.39)
        fig.savefig(f"{fdir}/f{fi:05d}.png", dpi=100)
        if fi == 75:
            fig.savefig("figures/diag_2d_still.png", dpi=180)
    encode(fdir, "diag_2d", gif_width=980)

# ============================================================================ stems morph
if "stems" in only:
    a, Wq, ws, bp, l2 = D["alphas"], D["W_q"], D["w_star"], D["bp"], D["l2"]
    from core import q_alpha_min
    X, y = D["X"], D["y"]
    # display order: interleave the true support among the coordinates so it is not a block at the left
    rng = np.random.default_rng(4)
    perm = rng.permutation(100)
    nF = 330
    lv = 2 - 7 * (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, nF, endpoint=False)))  # 1e2 -> 1e-5 -> 1e2
    cache = {}
    fdir = "cache/frames_stems"
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    fig = plt.figure(figsize=(14, 6.2), dpi=100)
    nu = None
    l1_curve_a = np.logspace(-5, 2, 120)
    l1_curve = []
    for aa in l1_curve_a:
        w, nu = q_alpha_min(X, y, aa, nu0=None)
        l1_curve.append([np.abs(w).sum(), np.linalg.norm(w - ws) / np.linalg.norm(ws)])
    l1_curve = np.array(l1_curve)
    for fi, v in enumerate(lv):
        alpha = 10 ** v
        w = q_alpha_min(X, y, alpha)[0][perm]
        fig.clf()
        ax = fig.add_axes([0.05, 0.14, 0.62, 0.7])
        idx = np.arange(100)
        col = S.ALPHA_CMAP((v + 5) / 7)
        ax.vlines(idx, 0, w, color=col, lw=2.2)
        ax.scatter(idx, w, s=16, color=col, zorder=3)
        wsp = ws[perm]
        sup = np.nonzero(wsp)[0]
        ax.scatter(sup, wsp[sup], s=120, facecolor="none", edgecolor=S.INK, lw=1.3, zorder=4)
        ax.axhline(0, color=S.AXIS, lw=0.8)
        ax.set_ylim(-2.3, 2.3); ax.set_xlim(-2, 101)
        ax.set_xticks([]); ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("coefficient $w_i$")
        ax.set_xlabel("the 100 coordinates of $w$   (rings: the true 5-sparse $w^\\star$)")
        fig.text(0.05, 0.9, "The interpolating solution gradient descent finds, as the init scale shrinks", fontsize=14, color=S.INK)
        fig.text(0.05, 0.855, "d = 100, n = 40 measurements: a 60-dimensional space of perfect fits", fontsize=11, color=S.INK2)
        fig.text(0.72, 0.8, "INIT SCALE", color=S.MUTED, fontsize=10.5)
        fig.text(0.72, 0.72, f"α = {alpha:.0e}" if alpha < 0.1 else f"α = {alpha:.2g}", fontsize=28, family=S.MONO, color=S.INK)
        bx = fig.add_axes([0.745, 0.16, 0.22, 0.44])
        lla = np.log10(l1_curve_a)
        for j in range(len(lla) - 1):
            bx.plot(lla[j:j + 2], l1_curve[j:j + 2, 0], color=S.ALPHA_CMAP((lla[j] + 5) / 7), lw=2.6)
        bx.scatter([v], [np.abs(w).sum()], s=70, color=col, edgecolor=S.PAPER, lw=2, zorder=5)
        bx.axhline(np.abs(bp).sum(), color=S.AQUA, lw=0.9, ls=(0, (3, 3)))
        bx.axhline(np.abs(l2).sum(), color=S.BLUE, lw=0.9, ls=(0, (3, 3)))
        bx.text(-5, np.abs(bp).sum() + 0.3, "basis pursuit", fontsize=9.5, color=S.INK2)
        bx.text(2, np.abs(l2).sum() - 0.4, "min-$L_2$", fontsize=9.5, color=S.INK2, ha="right", va="top")
        bx.set_xticks([-5, -3, -1, 1]); bx.set_xticklabels(["1e-5", "1e-3", "0.1", "10"])
        bx.set_xlabel("α"); bx.set_title("$\\|w\\|_1$ of the solution", fontsize=11.5)
        err = np.linalg.norm(w - wsp) / np.linalg.norm(wsp)
        fig.text(0.965, 0.72, f"error to $w^\\star$: {100 * err:.1f}%", ha="right", fontsize=12, color=S.INK2)
        fig.savefig(f"{fdir}/f{fi:05d}.png", dpi=100)
        if fi == nF // 2:
            fig.savefig("figures/diag_stems_still.png", dpi=180)
    encode(fdir, "diag_stems", gif_width=980)
