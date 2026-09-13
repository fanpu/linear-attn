"""Animation: the network's item -> feature matrix sharpens one singular mode at a time.

Top: target features.  Middle: the network's measured output W(t) x_i for every item.
Bottom: the target split into its SVD pieces s_a u_a v_a^T, each drawn at the strength the network has
learned so far (u_a(t)/s_a), so that the middle row is (up to small cross terms) their sum.

  .venv/bin/python 03-saxe-dynamics/render_matrix.py [--still]
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
from matplotlib.colors import LinearSegmentedColormap

import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
D = np.load(HERE / "cache/semantic.npz")
T_REC, S, MODES, Wt = D["t"], D["s"], D["modes"], D["W"]
U, V, Y = D["U"], D["V"], D["Y"]
ITEMS = [str(x) for x in D["items"]]
FEATS = [str(x) for x in D["features"]]
P = len(ITEMS)
GROUPS = [[0], [1], [2], [3], [4, 5, 6, 7]]
NAMES = ["living thing", "animal vs plant", "bird vs fish", "tree vs flower", "individual items"]
CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#b5452b"])


def interp(arr, t):
    i = int(np.clip(np.searchsorted(T_REC, t) - 1, 0, len(T_REC) - 2))
    w = float(np.clip((t - T_REC[i]) / (T_REC[i + 1] - T_REC[i]), 0, 1))
    return arr[i] * (1 - w) + arr[i + 1] * w


def draw(args):
    t, out = args
    Pl = style.use("light")
    fig = plt.figure(figsize=(16, 9), dpi=100)
    W = interp(Wt, t)
    Yhat = (W * np.sqrt(P)).T  # items x features
    ext = dict(cmap=CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    left, width = 0.1, 0.86
    ax1 = fig.add_axes([left, 0.62, width, 0.18])
    ax1.imshow(Y, **ext)
    ax2 = fig.add_axes([left, 0.37, width, 0.18])
    ax2.imshow(Yhat, **ext)
    fig.text(left, 0.955, "target: which features each item has", fontsize=13.5, weight="bold")
    fig.text(left, 0.565, f"what the network currently says  (training time t = {t:4.1f})", fontsize=13.5, weight="bold")
    for ax in (ax1, ax2):
        ax.set_yticks(range(P)); ax.set_yticklabels(ITEMS, fontsize=10.5)
        ax.set_xticks([]); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax1.set_xticks(range(len(FEATS)))
    ax1.set_xticklabels(FEATS, rotation=55, ha="left", fontsize=8.8, color=Pl["muted"])
    ax1.xaxis.tick_top()
    # mode pieces
    pw = width / 5 - 0.012
    for g, idx in enumerate(GROUPS):
        x0 = left + g * (width / 5)
        ax = fig.add_axes([x0, 0.105, pw, 0.15])
        piece = sum(S[a] * np.outer(U[:, a], V[:, a]) for a in idx) * np.sqrt(P)  # features x items
        frac = float(np.mean([np.clip(interp(MODES[:, a], t) / S[a], 0, 1.2) for a in idx]))
        ax.imshow(piece.T * frac, **ext)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(Pl["modes"][g]); sp.set_linewidth(2.5)
        ax.set_title(NAMES[g], fontsize=11.5, loc="left", pad=5)
        # progress meter
        mx = fig.add_axes([x0, 0.07, pw, 0.012])
        mx.barh([0], [1], color=Pl["rule"], height=1)
        mx.barh([0], [min(frac, 1)], color=Pl["modes"][g], height=1)
        mx.set_xlim(0, 1); mx.axis("off")
        fig.text(x0 + pw, 0.035, f"{min(frac, 1) * 100:3.0f}% learned", ha="right", fontsize=10, color=Pl["muted"])
        if g < 4:
            fig.text(x0 + pw + 0.006, 0.18, "+", fontsize=18, color=Pl["muted"], ha="center", va="center")
    fig.text(left, 0.30, "…which is (almost exactly) a sum of the target's SVD pieces, each switched on at its own time:",
             fontsize=13, color=Pl["ink"])
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    if a.still:
        draw((5.5, HERE / "_preview/matrix_still.png"))
        return
    fr = HERE / "cache/frames_matrix"
    if fr.exists():
        shutil.rmtree(fr)
    fr.mkdir(parents=True)
    fps = 24
    ts = list(np.linspace(0, 16, int(11 * fps))) + [16.0] * int(1.5 * fps)
    jobs = [(t, fr / f"f{k:04d}.png") for k, t in enumerate(ts)]
    with mp.Pool(4) as pool:
        pool.map(draw, jobs, chunksize=4)
    out = FIG / "matrix_sharpening.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fr / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                    str(out)], check=True)
    shutil.copy(fr / f"f{len(ts) - 1:04d}.png", FIG / "matrix_final.png")
    print(out)


if __name__ == "__main__":
    main()
