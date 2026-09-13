"""muTransfer figures: validation loss vs learning rate across widths, SP vs muP (static + animated build-up).

    .venv/bin/python 04-lazy-rich-mup/render_mup.py          # figures/mup_transfer.png
    .venv/bin/python 04-lazy-rich-mup/render_mup.py --anim   # figures/mup_buildup.mp4
"""
import os, subprocess, sys

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style as S
from analyze_lm import summary

HERE = os.path.dirname(os.path.abspath(__file__))
RES = summary()["width"]
WIDTHS = [128, 256, 512, 1024]
YLIM = (2.55, 4.6)


def panel(ax, param, reveal=None, show_opt=True, title=True):
    """reveal: dict width -> fraction in [0,1] of the curve to draw (for the animation)."""
    reveal = reveal or {w: 1.0 for w in WIDTHS}
    for i, w in enumerate(WIDTHS):
        key = f"{param}_{w}"
        if key not in RES or reveal.get(w, 0) <= 0:
            continue
        r = RES[key]
        c = S.WIDTH_RAMP[i]
        x = np.log2(r["lrs"]); y = np.array(r["val"], float)
        frac = reveal[w]
        xmax = x[0] + frac * (x[-1] - x[0])
        # draw finite segments up to xmax (interpolated end)
        xs, ys = [], []
        for k in range(len(x)):
            if x[k] <= xmax and np.isfinite(y[k]):
                xs.append(x[k]); ys.append(y[k])
            elif k > 0 and x[k - 1] < xmax < x[k] and np.isfinite(y[k]) and np.isfinite(y[k - 1]):
                t = (xmax - x[k - 1]) / (x[k] - x[k - 1])
                xs.append(xmax); ys.append(y[k - 1] + t * (y[k] - y[k - 1]))
        ax.plot(xs, np.minimum(ys, YLIM[1] + 1), "-", color=c, lw=2.2, zorder=3, solid_capstyle="round")
        ax.plot([a for a in x if a <= xmax + 1e-9], [np.minimum(b, YLIM[1] + 1) for a, b in zip(x, y) if a <= xmax + 1e-9],
                "o", color=c, ms=5, mec=S.PAPER, mew=1, zorder=4)
        for k in range(len(x)):  # diverged runs
            if x[k] <= xmax + 1e-9 and (not np.isfinite(y[k]) or y[k] > YLIM[1]):
                ax.plot(x[k], YLIM[1] - 0.06, marker="x", color=c, ms=6, mew=1.8, zorder=4, clip_on=False)
        if show_opt and frac >= 1 and np.isfinite(r["opt_log2lr"]):
            ax.plot(r["opt_log2lr"], r["opt_val"], marker="*", ms=15, color=c, mec=S.INK, mew=0.8, zorder=6)
            ax.plot([r["opt_log2lr"]] * 2, [YLIM[0], r["opt_val"]], color=c, lw=1.1, ls=(0, (2, 2)), zorder=2, alpha=.9)
            last = np.where(np.isfinite(y))[0]
            if len(last):
                ax.text(x[last[0]] - 0.15, y[last[0]], f"d = {w:,}", color=c, fontsize=9.5, ha="right", va="center",
                        fontweight="bold")
    ax.set_xlim(-12.8, -4.6); ax.set_ylim(*YLIM)
    ticks = np.arange(-12, -4)
    ax.set_xticks(ticks); ax.set_xticklabels([f"$2^{{{t}}}$" for t in ticks])
    ax.set_xlabel("Adam learning rate η  (base value; μP scales hidden layers by 128/d)" if param == "mup" else "Adam learning rate η")
    if title:
        ax.set_title("Standard parameterization (SP)" if param == "sp" else "Maximal update parameterization (μP)")


def static():
    S.use("light")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, param in zip(axs, ["sp", "mup"]):
        panel(ax, param)
    axs[0].set_ylabel("validation loss after 4.1M tokens (nats)")
    S.save(fig, f"{HERE}/figures/mup_transfer.png")


def anim(fps=30):
    S.use("light")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    W, H = 1680, 640
    out = f"{HERE}/figures/mup_buildup.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    per = int(1.6 * fps)  # frames to draw one width's curve
    hold = int(0.5 * fps)
    timeline = []
    for i, w in enumerate(WIDTHS):
        for k in range(per):
            timeline.append((i, (k + 1) / per))
        timeline += [(i, 1.0)] * hold
    timeline += [(len(WIDTHS) - 1, 1.0)] * int(2.5 * fps)
    fig, axs = plt.subplots(1, 2, figsize=(W / 100, H / 100), dpi=100, sharey=True, gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.14)
    for i, frac in timeline:
        reveal = {w: (1.0 if j < i else frac if j == i else 0.0) for j, w in enumerate(WIDTHS)}
        for ax, param in zip(axs, ["sp", "mup"]):
            ax.clear()
            panel(ax, param, reveal=reveal)
        axs[0].set_ylabel("validation loss (nats)")
        axs[1].tick_params(labelleft=False)
        fig.canvas.draw()
        p.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    p.stdin.close(); p.wait()
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    anim() if "--anim" in sys.argv else static()
