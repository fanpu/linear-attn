"""Animation: one softmax attention head escaping two saddles (showcase run, seed 0).

Arcs: measured average attention from the query (last token) to token 1; the remaining weight is split evenly
over the 7 distractors (their individual weights were not recorded; by symmetry they are equal on average).
Bars: measured OV mode strengths u_k / s_k (solid) and the attention-frozen-uniform control (outline).

  .venv/bin/python 03-saxe-dynamics/render_attn_anim.py [--still]
"""
import argparse
import multiprocessing as mp
import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
D = np.load(HERE / "cache/attn_showcase.npz")
T_REC = D["t"]
S = D["S"][0][:3]
A1 = D["a1"][:, 0]
MODES = D["modes"][:, 0, :3]
MODES_U = D["modes"][:, 3, :3]
LOSS = D["loss"][:, 0]
LOSS_U = D["loss"][:, 3]
T_END = 45.0
T_OV = sc.first_crossing(T_REC, MODES[:, 0], S[0] / 2)
T_AT = sc.first_crossing(T_REC, A1, 0.5)


def interp(arr, t):
    i = int(np.clip(np.searchsorted(T_REC, t) - 1, 0, len(T_REC) - 2))
    w = float(np.clip((t - T_REC[i]) / (T_REC[i + 1] - T_REC[i]), 0, 1))
    return arr[i] * (1 - w) + arr[i + 1] * w


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def draw(args):
    t, out = args
    P = style.use("dark")
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0.03, 0.08, 0.55, 0.78])
    ax.set_xlim(-0.5, 8.7); ax.set_ylim(-3.6, 4.4); ax.axis("off")
    fig.text(0.035, 0.92, "One attention head, two saddles", fontsize=22, weight="bold", color=style.NIGHT_INK)
    fig.text(0.035, 0.88, "task: at the last token, output  $M x_1$   (M has singular values 3, 1.5, 0.75)",
             fontsize=13, color=style.NIGHT_MUTED)
    a1 = float(interp(A1, t))
    others = (1 - a1) / 7
    q = np.array([7.5, 0.0])
    accent = "#ff7a59"
    for j in range(8):
        x = 0.5 + j * 1.0 if j < 7 else 7.5
        xc = j * 1.0 + 0.5
        w = a1 if j == 0 else others
        if j < 7:
            p0, p1 = np.array([7.5, 0.55]), np.array([xc, 0.55])
            h = 0.9 + 0.45 * (7 - j)
            path = Path([p0, [(p0[0] + p1[0]) / 2, 0.55 + h], p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            lw = 1 + 16 * w
            for gw, ga in ((lw * 3.5, 0.06), (lw * 2, 0.12)):
                ax.add_patch(PathPatch(path, fc="none", ec=accent, lw=gw, alpha=ga * (0.3 + 0.7 * w / max(a1, 1e-3)), capstyle="round"))
            ax.add_patch(PathPatch(path, fc="none", ec=accent, lw=lw, alpha=0.35 + 0.65 * min(1, w * 4), capstyle="round"))
        box = FancyBboxPatch((xc - 0.36, -0.3), 0.72, 0.72, boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc="#161b27", ec=accent if j in (0, 7) else "#3a4050", lw=2 if j in (0, 7) else 1.2)
        ax.add_patch(box)
        ax.text(xc, 0.06, f"$x_{j + 1}$", ha="center", va="center", fontsize=15, color=style.NIGHT_INK)
    ax.text(7.5, -0.62, "query", ha="center", fontsize=11, color=style.NIGHT_MUTED)
    ax.text(0.5, -0.62, f"attention {a1:.2f}", ha="center", fontsize=11.5, color=style.NIGHT_INK)
    ax.text(3.9, -0.62, f"each distractor {others:.2f}", ha="center", fontsize=11, color=style.NIGHT_MUTED)
    # OV bars
    ax.text(0.14, -1.35, "value/output circuit: how much of each singular mode of M it has learned", fontsize=12.5,
            color=style.NIGHT_INK)
    cols = [P["modes"][0], P["modes"][2], P["modes"][3]]
    for k in range(3):
        y = -2.1 - 0.55 * k
        frac = float(np.clip(interp(MODES[:, k], t) / S[k], 0, 1.35))
        fu = float(np.clip(interp(MODES_U[:, k], t) / S[k], 0, 1.35))
        L = 6.2
        ax.add_patch(FancyBboxPatch((1.3, y - 0.14), L, 0.28, boxstyle="round,pad=0,rounding_size=0.1", fc="#1a1f2b", ec="none"))
        ax.add_patch(FancyBboxPatch((1.3, y - 0.14), L * min(frac, 1.35) / 1.35, 0.28,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc=cols[k], ec="none"))
        ax.plot([1.3 + L * fu / 1.35] * 2, [y - 0.2, y + 0.2], color="#ffffff", lw=1.5, alpha=0.7)
        ax.plot([1.3 + L / 1.35] * 2, [y - 0.22, y + 0.22], color=style.NIGHT_MUTED, lw=1, ls=(0, (1, 1.5)))
        ax.text(1.15, y, f"s={S[k]:g}", ha="right", va="center", fontsize=11.5, color=style.NIGHT_INK)
    ax.text(1.3 + 6.2 / 1.35, -1.83, "target", ha="center", va="bottom", fontsize=10, color=style.NIGHT_MUTED)
    ax.text(1.3, -3.55, "white tick: the same run with attention frozen uniform", ha="left", fontsize=10.5,
            color=style.NIGHT_MUTED)

    # loss
    axl = fig.add_axes([0.63, 0.30, 0.34, 0.46])
    axl.set_facecolor(style.NIGHT)
    ti = np.searchsorted(T_REC, t)
    axl.plot(T_REC, LOSS_U, color="#ffffff", lw=1.2, ls=(0, (3, 2)), alpha=0.35)
    axl.plot(T_REC, LOSS, color="#ffffff", lw=0.8, alpha=0.18)
    axl.plot(T_REC[: ti + 1], LOSS[: ti + 1], color=accent, lw=2.6)
    axl.scatter([t], [float(interp(LOSS, t))], s=40, color=accent, edgecolor=style.NIGHT, lw=1.5, zorder=5)
    axl.set_yscale("log"); axl.set_ylim(2e-3, 12); axl.set_xlim(0, T_END)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axl.spines[sp].set_color(style.NIGHT_RULE)
    axl.tick_params(colors=style.NIGHT_MUTED, labelsize=10)
    axl.set_xlabel("training time  t", color=style.NIGHT_MUTED)
    fig.text(0.63, 0.785, "loss  (dashed: attention frozen uniform)", fontsize=13, color=style.NIGHT_INK)
    phases = [(0.0, "1  value circuit learns through blurry attention"), (T_OV, "2  attention escapes its own saddle"),
              (T_AT, "3  attention locks on: everything finishes")]
    for k, (t0, txt) in enumerate(phases):
        al = smooth((t - t0) / 1.5) if k else 1.0
        active = (t >= t0) and (k == 2 or t < phases[k + 1][0])
        fig.text(0.63, 0.17 - 0.038 * k, txt, fontsize=12.5, color=style.NIGHT_INK if active else style.NIGHT_MUTED,
                 alpha=max(al, 0.0) if k else 1.0, weight="bold" if active else "normal")
    for t0 in (T_OV, T_AT):
        if t >= t0:
            axl.axvline(t0, color=style.NIGHT_MUTED, lw=0.8, ls=(0, (1, 2)))
    fig.savefig(out, dpi=100, facecolor=style.NIGHT)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    if a.still:
        for t in (8.0, 24.5, 40.0):
            draw((t, HERE / f"_preview/attn_anim_{t:g}.png"))
        return
    fr = HERE / "cache/frames_attn"
    if fr.exists():
        shutil.rmtree(fr)
    fr.mkdir(parents=True)
    fps = 30
    ts = list(np.linspace(0, T_END, 13 * fps)) + [T_END] * int(1.5 * fps)
    jobs = [(t, fr / f"f{k:04d}.png") for k, t in enumerate(ts)]
    with mp.Pool(4) as pool:
        pool.map(draw, jobs, chunksize=4)
    out = FIG / "attention_saddles.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fr / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
