"""Deep matrix factorization figures.  Reads cache/mc_*.npz, cache/razin.npz.

  figures/mc_spectra.mp4/.gif   singular values over training, depth 1 / 2 / 3 (m = 2000 observed entries)
  figures/mc_summary.png        (a) recovery error vs depth vs min-nuclear-norm  (b) Arora et al. singular-value ODE check
  figures/razin.png             the Razin & Cohen 2x2 example
"""
import os, shutil, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S

S.light()
only = sys.argv[1:] or ["anim", "summary", "razin"]
FPS = 30


def load(N, m, tag=""):
    return np.load(f"cache/mc_N{N}_m{m}_s0{tag}.npz")


if "summary" in only:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1]))
    ax = axs[0]
    ms = [1250, 2000, 3000]
    xs = np.arange(len(ms))
    series = [("depth 1 (W trained directly)", lambda m: load(1, m)["err"][-1], S.DEPTH_COLORS[1], -0.21),
              ("min nuclear norm (convex)", lambda m: float(np.load(f"cache/mc_nuc_m{m}_s0.npz")["err"]), S.MUTED, -0.07),
              ("depth 2", lambda m: load(2, m)["err"][-1], S.DEPTH_COLORS[2], 0.07),
              ("depth 3", lambda m: (load(3, m, "_slow") if (m == 2000 and os.path.exists("cache/mc_N3_m2000_s0_slow.npz")) else load(3, m))["err"][-1], S.DEPTH_COLORS[3], 0.21)]
    for lab, f, col, off in series:
        vals = [f(m) for m in ms]
        for x, v in zip(xs, vals):
            hv = min(max(v, 0.004), 1.12)
            ax.bar(x + off, hv, width=0.12, color=col, zorder=2)
            txt = f"{v:.2f}" if v >= 0.005 else "0"
            if v > 1.12:
                txt = f"{v:.2f}\n(not\nconverged)"
            ax.text(x + off + (0.02 if v > 1.12 else 0), hv + 0.02, txt, ha="center", fontsize=8.5, color=S.INK2, va="bottom" if v <= 1.12 else "bottom")
        ax.bar([np.nan], [np.nan], color=col, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f"{m} entries\n({m / 100:.1f}% observed)" for m in ms])
    ax.set_ylabel(r"relative error $\|W - W^\star\|_F / \|W^\star\|_F$")
    ax.set_title("Completing a 100×100 rank-5 matrix")
    ax.legend(loc="upper right", fontsize=9.5, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, 1.45)
    ax.grid(axis="x", visible=False)

    ax = axs[1]
    for N, tag, lab in [(2, "", "depth 2"), (3, "_slow", "depth 3")]:
        D = load(N, 2000, tag)
        meas, pred, svv = D["dsv_meas"][:, :5], D["dsv_pred"][:, :5], D["sv"][:, :5]
        ok = (np.abs(pred) > 1e-12)
        big = ok & (svv >= 1e-3)
        small = ok & (svv < 1e-3)
        ax.scatter(np.abs(pred[small]), np.abs(meas[small]), s=9, color=S.AXIS, alpha=0.6, lw=0)
        r = np.median(meas[big] / pred[big])
        ax.scatter(np.abs(pred[big]), np.abs(meas[big]), s=14, color=S.DEPTH_COLORS[N], alpha=0.8, lw=0,
                   label=f"{lab}: median measured/predicted = {r:.4f}")
        print(lab, "median measured/predicted (sigma >= 1e-3)", r, " (sigma < 1e-3)", np.median(meas[small] / pred[small]))
    lim = [1e-10, 1]
    ax.loglog(lim, lim, color=S.INK, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"predicted change per step:  $-\eta\, N\, \sigma_r^{2-2/N} \langle \nabla\ell(W), u_r v_r^\top\rangle$")
    ax.set_ylabel(r"measured change of $\sigma_r$ per GD step")
    ax.set_title("Arora et al.'s singular-value dynamics, checked")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(2e-4, 1e-8, "top-5 singular values at every recorded step\ngray: $\\sigma_r < 10^{-3}$, where the random init is\nnot yet balanced and the equation does not apply", fontsize=9.5, color=S.INK2)
    fig.savefig("figures/mc_summary.png", bbox_inches="tight")
    print("wrote mc_summary.png")

if "razin" in only:
    R = np.load("cache/razin.npz")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.3))
    rows = [("N2_detpos", 2, "depth 2, det W(0) > 0", S.DEPTH_COLORS[2], "-"), ("N3_detpos", 3, "depth 3, det W(0) > 0", S.DEPTH_COLORS[3], "-"),
            ("N2_detneg", 2, "depth 2, det W(0) < 0", S.MUTED, (0, (4, 2))), ("N3_detneg", 3, "depth 3, det W(0) < 0", S.INK2, (0, (1, 1.5)))]
    for tag, N, lab, col, ls in rows:
        t, W, L, sv = R[tag + "_t"], R[tag + "_W"], R[tag + "_loss"], R[tag + "_sv"]
        det = W[:, 0, 0] * W[:, 1, 1] - W[:, 0, 1] * W[:, 1, 0]
        valid = np.ones(len(t), bool)
        if "detpos" in tag:  # the exact flow keeps det > 0 forever; float64 eventually tunnels through det = 0
            bad = np.where(det <= 0)[0]
            if len(bad):
                valid[bad[0]:] = False
                print(tag, "float64 crossed det=0 at t =", t[bad[0]])
        tv = t[valid]
        axs[0].loglog(tv, np.abs(W[valid, 0, 0]), color=col, lw=2.2, ls=ls, label=lab)
        if (~valid).any():
            axs[0].scatter([tv[-1]], [abs(W[valid, 0, 0][-1])], marker="x", s=40, color=col, zorder=4)
        if "detpos" in tag:
            fro = np.linalg.norm(W[valid], axis=(1, 2)); nuc = sv[valid].sum(1)
            sq = np.sqrt(L[valid])
            axs[1].loglog(1 / sq, nuc, color=col, lw=2.2, label=f"nuclear norm, {lab.split(',')[0]}")
            axs[1].loglog(1 / sq, fro, color=col, lw=1.2, ls=(0, (3, 2)))
            dist = sv[valid, 1]  # distance (Frobenius) to the nearest rank-1 matrix
            axs[2].loglog(sq, dist, color=col, lw=2.2, label=lab.split(",")[0])
    x = np.logspace(0, 8, 10)
    axs[1].loglog(x, x, color=S.INK, lw=1, ls=(0, (1, 2)))
    axs[1].text(3e4, 6e4 * 0.25, r"slope 1: norm $\propto 1/\sqrt{\ell}$", fontsize=9.5, color=S.INK2, ha="left")
    q = np.logspace(-8, 0, 10)
    axs[2].loglog(q, 3 * np.sqrt(2) * q, color=S.INK, lw=1.2, ls=(0, (4, 3)))
    axs[2].text(1e-5, 3 * np.sqrt(2) * 1e-5 * 2.5, r"Razin–Cohen bound $3\sqrt{2}\,\sqrt{\ell}$", fontsize=9.5, color=S.INK2, rotation=0)
    axs[0].set_xlabel("gradient-flow time t"); axs[0].set_ylabel(r"$|W_{11}|$, the unobserved entry")
    axs[0].set_title("The unobserved entry runs off to infinity...")
    axs[0].legend(fontsize=9, loc="upper left")
    axs[0].text(0.98, 0.30, "×: float64 loses the det > 0 invariant\n(the exact flow never does)", transform=axs[0].transAxes, ha="right", fontsize=9, color=S.MUTED)
    axs[1].set_xlabel(r"$1/\sqrt{\ell(t)}$  (training progress $\rightarrow$)"); axs[1].set_ylabel("norm of the product matrix")
    axs[1].set_title(r"...so every norm diverges as the loss $\rightarrow$ 0")
    axs[1].text(0.03, 0.9, "solid: nuclear norm   dashed: Frobenius\n(indistinguishable: W is nearly rank 1)", transform=axs[1].transAxes, fontsize=9.5, color=S.INK2)
    axs[2].set_xlabel(r"$\sqrt{\ell(t)}$"); axs[2].set_ylabel(r"$\sigma_2(W)$ = distance to rank 1")
    axs[2].set_title("...while the matrix approaches rank 1")
    axs[2].invert_xaxis()
    axs[2].legend(fontsize=9, loc="lower left")
    fig.savefig("figures/razin.png", bbox_inches="tight")
    print("wrote razin.png")

if "anim" in only:
    m = 2000
    runs = [(1, load(1, m)), (2, load(2, m)), (3, load(3, m, "_slow") if os.path.exists("cache/mc_N3_m2000_s0_slow.npz") else load(3, m))]
    Ws = runs[0][1]["Ws"]
    s_true = np.linalg.svd(Ws, compute_uv=False)[:5]
    TAU0, TAU1 = -0.5, 3.4
    nF = 330
    # time warp: linger where the stages happen (depth 2 around eta*t ~ 0.5, depth 3 around eta*t ~ 10)
    grid = np.linspace(TAU0, TAU1, 4000)
    dens = 1 + 2.0 * np.exp(-0.5 * ((grid + 0.25) / 0.3) ** 2) + 5.0 * np.exp(-0.5 * ((grid - 1.0) / 0.18) ** 2)
    cdf = np.cumsum(dens); cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    taus = np.r_[np.interp(np.linspace(0, 1, nF - 45), cdf, grid), np.full(45, TAU1)]
    fdir = "cache/frames_mc"
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    fig = plt.figure(figsize=(15, 7.6), dpi=100)
    K = 12
    titles = {1: "depth 1:  W trained directly", 2: "depth 2:  W = W₂W₁", 3: "depth 3:  W = W₃W₂W₁"}
    for fi, lt in enumerate(taus):
        fig.clf()
        fig.text(0.04, 0.94, "Matrix completion: the top singular values of W during training", fontsize=15, color=S.INK)
        fig.text(0.04, 0.905, "100×100 matrix of rank 5, 20% of entries observed, tiny random init. Ghost ticks: the true singular values.", fontsize=11, color=S.INK2)
        fig.text(0.96, 0.925, f"time  ηt = {10 ** lt:,.1f}" if lt < 1 else f"time  ηt = {10 ** lt:,.0f}", fontsize=17, family=S.MONO, ha="right", color=S.INK)
        for c, (N, D) in enumerate(runs):
            tau = np.log10(D["steps"] * float(D["lr"]))
            k = np.searchsorted(tau, lt, side="right") - 1
            k = int(np.clip(k, 0, len(tau) - 1))
            sv = D["sv"][k][:K]
            col = S.DEPTH_COLORS[N]
            x0 = 0.05 + c * 0.315
            ax = fig.add_axes([x0, 0.43, 0.27, 0.40])
            ax.bar(np.arange(K), sv, width=0.72, color=col, zorder=2)
            ax.scatter(np.arange(5), s_true, marker="_", s=420, color=S.INK, lw=1.4, zorder=3)
            ax.set_ylim(0, 72); ax.set_xlim(-0.7, K - 0.3)
            ax.set_xticks(np.arange(K)); ax.set_xticklabels([str(i + 1) for i in range(K)], fontsize=8.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(titles[N], fontsize=12.5)
            err = D["err"][k]; er = D["erank"][k]
            ax.text(K - 0.5, 68, f"error {100 * err:.0f}%\neffective rank {er:.1f}", ha="right", va="top", fontsize=10.5, color=S.INK2)
            if c == 0:
                ax.set_ylabel("singular value $\\sigma_r$")
            bx = fig.add_axes([x0, 0.08, 0.27, 0.26])
            L = D["sv"][:, :8]
            for r in range(8):
                bx.plot(tau, L[:, r], color=col, lw=1.6 if r < 5 else 0.9, alpha=1 if r < 5 else 0.5)
            bx.axvline(lt, color=S.INK, lw=1)
            bx.set_xlim(TAU0, TAU1); bx.set_ylim(0, 72)
            bx.set_xticks([0, 1, 2, 3]); bx.set_xticklabels(["1", "10", "100", "1000"])
            bx.set_xlabel("gradient-flow time ηt (log)", fontsize=10)
            if c == 0:
                bx.set_ylabel("$\\sigma_1 \\ldots \\sigma_8$")
        fig.savefig(f"{fdir}/f{fi:05d}.png", dpi=100)
        if fi == 200:
            fig.savefig("figures/mc_spectra_still.png", dpi=170)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{fdir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "figures/mc_spectra.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{fdir}/f%05d.png", "-vf",
                    "fps=20,scale=980:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "figures/mc_spectra.gif"], check=True)
    print("wrote mc_spectra", os.path.getsize("figures/mc_spectra.gif") / 1e6)
