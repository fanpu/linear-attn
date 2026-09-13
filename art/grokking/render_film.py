"""Crystallization film: the main seed's key-frequency stars {113/k} forming over training.
Each frame = one saved embedding checkpoint, W_E(t) projected onto the FINAL (cos, sin) plane of each key
frequency (a fixed camera), centred and scaled to unit RMS radius (declared).  Bottom strip: measured losses
with a cursor.  Writes gallery/film_<style>.mp4 and .gif"""
import argparse, os, subprocess, shutil
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from styles import STYLES, normalize, star_segments, fig_to_array, riso_composite, hex2rgb, P
from phases import phases

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--styles", type=str, default="nocturne,plotter,plate")
ap.add_argument("--layout", type=str, default="grid", choices=["grid", "overlay", "seeds"])
ap.add_argument("--size", type=int, default=1080)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

A = dict(np.load("cache/analysis.npz"))
s = args.seed
K = int(A["nkeys"][s]); order = np.argsort(A["keys"][s][:K]); keys = A["keys"][s][:K][order]
steps = A["emb_steps"]
ph = phases(A, s)
tag = f"seed{int(A['init_seeds'][s])}"

# frame schedule (declared time warp): every checkpoint early, every 20 steps through the transition,
# every 100 steps once test accuracy has been at 100% for 3000 steps
t_slow = (ph["t_done"] if ph["t_done"] > 0 else steps[-1]) + 3000
sel = [i for i, t in enumerate(steps) if t <= 1000 or t <= t_slow and t % 20 == 0 or t % 100 == 0]
if args.max_frames:
    sel = sel[:: max(1, len(sel) // args.max_frames)]
ev = A["ev_steps"]; te = A["te_loss"][:, s]; tr = np.maximum(A["tr_loss"][:, s], 1e-12)
R = A["R"][:, s][:, order]
rings = A["ring"][:, s][:, order]


def phase_name(t):
    if t < ph["t_circ"]:
        return "memorization"
    if t < ph["t_clean"]:
        return "circuit formation"
    if ph["t_done"] < 0 or t < ph["t_done"]:
        return "cleanup"
    return "grokked"


def layout(K):
    best = None
    for cols in range(1, K + 1):
        rows = int(np.ceil(K / cols))
        size = min(1.0 / cols, 0.70 / rows)
        if best is None or size > best[0]:
            best = (size, rows, cols)
    return best


def build(style):
    st = STYLES[style]
    S_ = args.size
    fig = plt.figure(figsize=(S_ / 100, S_ / 100), dpi=100, facecolor=st["bg"])
    plt.rcParams["font.family"] = st["font"]
    lcs, rtexts = [], []
    if args.layout == "grid":
        size, rows, cols = layout(K)
        for j in range(K):
            r, c = divmod(j, cols)
            n_in_row = min(cols, K - r * cols)
            x0 = (1 - n_in_row * size) / 2 + c * size
            y0 = 0.93 - (r + 1) * size - (0.70 - rows * size) / 2
            ax = fig.add_axes([x0 + 0.01, y0 + 0.01, size - 0.02, size - 0.02]); ax.set_facecolor(st["bg"])
            ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal"); ax.axis("off")
            lw = 0.9 if style != "plotter" else 0.75
            if style == "nocturne":
                lc = LineCollection(np.zeros((P, 2, 2)), colors=st["cyclic"](np.arange(P) / P), linewidths=lw, alpha=0.95)
            else:
                lc = LineCollection(np.zeros((P, 2, 2)), colors=st["ink"], linewidths=lw)
            ax.add_collection(lc); lcs.append(lc)
            ax.text(0, -1.78, f"k = {keys[j]}", ha="center", va="bottom", fontsize=11, color=st["muted"])
            rtexts.append(ax.text(0, 1.78, "", ha="center", va="top", fontsize=9, color=st["muted"]))
    elif args.layout == "seeds":
        cols, rows = 4, 3
        cw = 0.96 / cols; chh = 0.69 / rows
        for i in range(len(A["init_seeds"])):
            r, c = divmod(i, cols)
            ax = fig.add_axes([0.02 + c * cw, 0.935 - (r + 1) * chh + 0.012, cw, chh - 0.012]); ax.set_facecolor(st["bg"])
            ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.axis("off")
            for j in range(int(A["nkeys"][i])):
                if style == "nocturne":
                    lc = LineCollection(np.zeros((P, 2, 2)), colors=st["cyclic"](np.arange(P) / P), linewidths=0.45, alpha=0.6)
                else:
                    lc = LineCollection(np.zeros((P, 2, 2)), colors=st["ink"], linewidths=0.35, alpha=0.85)
                ax.add_collection(lc); lcs.append((i, j, lc))
            rtexts.append(ax.text(0, -1.7, "", ha="center", va="bottom", fontsize=8, color=st["muted"]))
    else:
        ax = fig.add_axes([0.1, 0.24, 0.8, 0.72 * 0.98]); ax.set_facecolor(st["bg"])
        ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.65, 1.65); ax.set_aspect("equal"); ax.axis("off")
        for j in range(K):
            if style == "nocturne":
                lc = LineCollection(np.zeros((P, 2, 2)), colors=st["cyclic"](np.arange(P) / P), linewidths=0.6, alpha=0.6)
            else:
                lc = LineCollection(np.zeros((P, 2, 2)), colors=st["ink"], linewidths=0.5, alpha=0.8)
            ax.add_collection(lc); lcs.append(lc)
    # header
    title = fig.text(0.04, 0.975, "", fontsize=15, color=st["ink"], va="top", style="italic" if style == "plate" else "normal")
    sub = fig.text(0.96, 0.975, "", fontsize=12, color=st["muted"], va="top", ha="right")
    # curve strip
    cax = fig.add_axes([0.08, 0.05, 0.86, 0.15]); cax.set_facecolor(st["bg"])
    if args.layout == "seeds":
        for i in range(len(A["init_seeds"])):
            cax.plot(ev, A["te_loss"][:, i], color=st["test"], lw=0.6, alpha=0.8)
            cax.plot(np.arange(len(tr))[::10], np.maximum(A["tr_loss"][::10, i], 1e-12), color=st["train"], lw=0.6, alpha=0.8,
                     ls="-" if style != "plotter" else (0, (4, 2)))
    else:
        cax.plot(np.arange(len(tr))[::10], tr[::10], color=st["train"], lw=1.0)
        cax.plot(ev, te, color=st["test"], lw=1.0, ls="-" if style != "plotter" else (0, (4, 2)))
    cax.set_yscale("log"); cax.set_xlim(0, steps[-1]); cax.set_ylim(1e-8, 80)
    for sp in ["top", "right"]:
        cax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        cax.spines[sp].set_color(st["muted"]); cax.spines[sp].set_linewidth(0.6)
    cax.tick_params(colors=st["muted"], labelsize=8)
    cax.set_yticks([1e-6, 1e-3, 1])
    cax.text(0.005, 1.02, ("train loss" if args.layout != "seeds" else "12 train losses") + ("  —" if style != "plotter" else " – –" if args.layout == "seeds" else " ——"), transform=cax.transAxes, color=st["train"], fontsize=9)
    cax.text(0.22, 1.02, ("test loss" if args.layout != "seeds" else "12 test losses") + ("  —" if style != "plotter" else " ——" if args.layout == "seeds" else " – –"), transform=cax.transAxes, color=st["test"], fontsize=9)
    cax.text(1.0, -0.33, "training step", transform=cax.transAxes, color=st["muted"], fontsize=8, ha="right")
    cur = cax.axvline(0, color=st["ink"], lw=0.8)
    if style == "plate":
        fr = fig.add_axes([0.012, 0.012, 0.976, 0.976]); fr.set_facecolor("none"); fr.set_xticks([]); fr.set_yticks([])
        for sp in fr.spines.values():
            sp.set_color(st["ink"]); sp.set_linewidth(1.0)
    return fig, lcs, rtexts, title, sub, cur


_state = {}


def render_range(job):
    style, idxs, outdir = job
    from PIL import Image
    fig, lcs, rtexts, title, sub, cur = build(style)
    for fi, ti in idxs:
        t = int(steps[ti])
        if args.layout == "seeds":
            for (i, j, lc) in lcs:
                lc.set_segments(star_segments(normalize(A["ring"][ti, i, j].astype(np.float64))))
            for i, tx in enumerate(rtexts):
                tei_ = min(np.searchsorted(ev, t), len(ev) - 1)
                tx.set_text(f"seed {int(A['init_seeds'][i])}{'' if A['data_seeds'][i] == 598 else '*'}  ·  test {A['te_acc'][tei_, i] * 100:.0f}%")
        else:
          for j in range(K):
            z = normalize(rings[ti, j].astype(np.float64))
            lcs[j].set_segments(star_segments(z))
            if rtexts:
                rtexts[j].set_text(f"R = {R[ti, j]:.2f}")
        title.set_text(f"step {t:>6,d}   ·   {phase_name(t)}" if args.layout != "seeds" else f"step {t:>6,d}")
        tei = min(np.searchsorted(ev, t), len(ev) - 1)
        sub.set_text(f"test acc {A['te_acc'][tei, s] * 100:5.1f}%" if args.layout != "seeds" else "* = different train split")
        cur.set_xdata([t, t])
        arr = fig_to_array(fig)
        Image.fromarray(arr).save(f"{outdir}/{fi:05d}.png", compress_level=1)
    plt.close(fig)
    return len(idxs)


if __name__ == "__main__":
    os.makedirs("gallery", exist_ok=True)
    for style in args.styles.split(","):
        name = f"film_{args.layout}_{tag}_{style}" if args.layout != "seeds" else f"film_seeds_{style}"
        outdir = f"cache/frames_{name}"
        shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
        items = list(enumerate(sel))
        # hold the final frame 4 s
        items += [(len(sel) + h, sel[-1]) for h in range(4 * args.fps)]
        chunks = [(style, items[i::args.workers], outdir) for i in range(args.workers)]
        with Pool(args.workers) as pool:
            pool.map(render_range, chunks)
        mp4 = f"gallery/{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{outdir}/%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", mp4], check=True)
        gif = f"gallery/{name}.gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{outdir}/%05d.png",
                        "-vf", "fps=8,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                        gif], check=True)
        print(name, len(items), "frames", os.path.getsize(mp4) // 1024, "KB mp4", os.path.getsize(gif) // 1024, "KB gif", flush=True)
