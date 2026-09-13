"""Chizat-Oyallon-Bach alpha sweep on binary CIFAR: static summary + kernel-matrix animation.

    .venv/bin/python 04-lazy-rich-mup/render_alpha.py          # figures/alpha_sweep.png
    .venv/bin/python 04-lazy-rich-mup/render_alpha.py --anim   # figures/kernel_alignment.mp4 (+ still)
"""
import os, subprocess, sys

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as ccm
from matplotlib.colors import LinearSegmentedColormap

import style as S
from analyze_ntk import slope

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(f"{HERE}/cache/ntk_alpha_summary.npz")
A = d["alpha"]
AS = np.unique(A)
REG = LinearSegmentedColormap.from_list("reg", [S.RICH, "#b9a79b", S.LAZY])  # rich -> lazy


def acol(a):
    return REG((np.log10(a) - np.log10(AS[0])) / (np.log10(AS[-1]) - np.log10(AS[0])))


def gmean(key):
    return np.array([np.exp(np.log(d[key][A == a]).mean()) for a in AS])


def static():
    S.use("light")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for key, c, lab, xy in [("dK", S.INK, "tangent kernel", None), ("dW", S.MUTED, "first-layer weights", None)]:
        ax.scatter(A, d[key], s=14, color=c, alpha=.3, lw=0)
        g = gmean(key)
        ax.plot(AS, g, "o-", color=c, ms=6, lw=1.4, mec=S.PAPER, mew=1)
        sel = A >= 1
        s, se, _ = slope(A[sel], d[key][sel])
        ax.text(AS[-1] * 1.25, g[-1], f"{lab}\nslope {s:+.2f}±{se:.2f} (α≥1)", color=c, fontsize=9, va="center")
    xs = np.array([0.3, 150])
    ax.plot(xs, gmean("dK")[AS == 1] * (xs / 1.0) ** -1, ls=(0, (4, 3)), color=S.INK, lw=1, alpha=.7)
    ax.text(0.35, gmean("dK")[AS == 1][0] * 3.5, r"$\propto 1/\alpha$", fontsize=10.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("output scale α"); ax.set_ylabel("relative change during training")
    ax.set_title("Movement falls like 1/α")
    ax.set_xlim(AS[0] / 2, AS[-1] * 2)

    ax = axs[1]
    e = np.array([d["err"][A == a].mean() for a in AS]) * 100
    for a in AS:
        ax.scatter([a] * (A == a).sum(), 100 * d["err"][A == a], color=acol(a), s=18, alpha=.45, lw=0)
    ax.plot(AS, e, "-", color=S.MUTED, lw=1.2, zorder=1)
    ax.scatter(AS, e, color=[acol(a) for a in AS], s=55, edgecolor=S.PAPER, lw=1.2, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("output scale α"); ax.set_ylabel("test error (%)")
    ax.set_title("Test error across the dial")
    ax.text(AS[0], ax.get_ylim()[1], "rich", color=S.RICH, fontweight="bold", va="top")
    ax.text(AS[-1], ax.get_ylim()[1], "lazy", color=S.LAZY, fontweight="bold", va="top", ha="right")

    ax = axs[2]
    st = np.maximum(d["steps"], 1)
    for a in AS:
        tr = d["align_traj"][A == a].mean(0)
        ax.plot(st, tr, color=acol(a), lw=2)
        ax.text(st[-1] * 1.15, tr[-1], f"α={a:g}", color=acol(a), fontsize=8.5, va="center")
    ax.set_xscale("log"); ax.set_xlabel("gradient step"); ax.set_ylabel(r"alignment  $y^\top\Theta_t\,y\,/\,(\|\Theta_t\|_F\,\|y\|^2)$")
    ax.set_title("Only the rich networks align their kernel")
    S.save(fig, f"{HERE}/figures/alpha_sweep.png")


def anim(fps=30):
    S.use("light")
    pick = [(AS[0], "RICH", S.RICH), (AS[-1], "LAZY", S.LAZY)]
    runs = []
    for a, name, col in pick:
        k = np.where((A == a) & (d["seed"] == 0))[0][0]
        K = d["ksub"][k].astype(np.float64)
        K = K / np.mean(np.diagonal(K, axis1=1, axis2=2), axis=1)[:, None, None]  # normalize by mean self-similarity
        runs.append((a, name, col, K, d["align_traj"][k], d["err_traj"][k]))
    st = np.maximum(d["steps"], 1).astype(float)
    F = len(st)
    sub_per = 4
    tl = [(i + j / sub_per) for i in range(F - 1) for j in range(sub_per)] + [F - 1] * int(2.5 * fps)
    W, H = 1600, 760
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = f"{HERE}/figures/kernel_alignment.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.03, right=0.97, top=0.86, bottom=0.12, wspace=0.18)
    cmap = ccm.vik
    for fi, tt in enumerate(tl):
        fig.clear()
        i0 = int(np.floor(tt)); fr = tt - i0; i1 = min(i0 + 1, F - 1)
        for j, (a, name, col, K, al, er) in enumerate(runs):
            ax = fig.add_subplot(gs[j])
            M = (1 - fr) * K[i0] + fr * K[i1]
            ax.imshow(M, cmap=cmap, vmin=-1.2, vmax=1.2, interpolation="nearest")
            ax.axhline(47.5, color=S.PAPER, lw=2); ax.axvline(47.5, color=S.PAPER, lw=2)
            ax.set_xticks([24, 72]); ax.set_xticklabels(["airplanes", "automobiles"], fontsize=10)
            ax.set_yticks([24, 72]); ax.set_yticklabels(["airplanes", "automobiles"], fontsize=10, rotation=90, va="center")
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(f"{name}   α = {a:g}", color=col, fontsize=15, loc="left")
            ax.text(0, -0.12, "", transform=ax.transAxes)
        ax = fig.add_subplot(gs[2])
        tnow = np.exp((1 - fr) * np.log(st[i0]) + fr * np.log(st[i1]))
        for a, name, col, K, al, er in runs:
            ax.plot(st, al, color=col, lw=1.2, alpha=.25)
            m = st <= tnow + 1e-9
            ax.plot(st[m], al[m], color=col, lw=2.6)
            ax.plot(tnow, (1 - fr) * al[i0] + fr * al[i1], "o", color=col, ms=8, mec=S.PAPER, mew=1.5)
        ax.set_xscale("log"); ax.set_xlim(1, st[-1] * 1.05)
        ax.set_ylim(0, max(r[4].max() for r in runs) * 1.12)
        ax.set_xlabel("gradient step"); ax.set_title("kernel–label alignment", loc="left", fontsize=13)
        ax.text(0.03, 0.97, f"step {int(round(tnow)):,}", transform=ax.transAxes, va="top", color=S.MUTED, fontsize=11)
        fig.text(0.03, 0.95, "The tangent kernel $\\Theta_t$ on 96 training images, sorted by class", fontsize=13, color=S.INK,
                 fontweight="bold")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        p.stdin.write(buf.tobytes())
        if fi == len(tl) - 1:
            plt.imsave(f"{HERE}/figures/kernel_alignment_still.png", buf)
    p.stdin.close(); p.wait()
    print(out)


if __name__ == "__main__":
    anim() if "--anim" in sys.argv else static()
