"""Task-diversity figures from cache/taskdiv_<tag>_s0.json (taskdiv.py).

  figures/taskdiv.png        three panels vs number of pretraining tasks M
  figures/taskdiv_sweep.mp4  animated sweep over M: error at each context position, PT vs dMMSE vs ridge

  .venv/bin/python 08-icl-linear-attention/render_taskdiv.py --tag main
"""
import argparse, json, os, pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import style

p = argparse.ArgumentParser()
p.add_argument("--tag", default="main")
p.add_argument("--anim", type=int, default=1)
a = p.parse_args()
HERE = pathlib.Path(__file__).parent
h = json.load(open(HERE / "cache" / f"taskdiv_{a.tag}_s0.json"))
logM = [None if s == "inf" else int(s) for s in h["args"]["logM"].split(",")]
ev = h["evals"][-1]
fin = [i for i, m in enumerate(logM) if m is not None]
inf_i = [i for i, m in enumerate(logM) if m is None]
xs = np.array([logM[i] for i in fin])
K, D = len(ev["true"]["pt"][0]), h["args"]["D"]
m = lambda wh, key: np.array(ev[wh][key]).mean(1)       # average over context positions (already /D)

style.use_light()
PT = style.C["softmax"]            # the GPT-2 model is a softmax transformer
DM = "#8f887c"
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.32))
for ax, wh, title in [(axs[0], "true", "new tasks  $w \\sim \\mathcal{N}(0, I)$"), (axs[1], "pretrain", "tasks seen in pretraining")]:
    ax.plot(xs, m(wh, "dmmse")[fin], color=DM, lw=2, marker="o", ms=4)
    ax.axhline(m(wh, "ridge")[fin].mean(), color=style.INK, lw=1.5)
    ax.plot(xs, m(wh, "pt")[fin], color=PT, lw=2.2, marker="o", ms=6, mec=style.PAPER, mew=1.5)
    if inf_i:
        ax.scatter([xs.max() + 2], m(wh, "pt")[inf_i], color=PT, s=40, zorder=5, edgecolor=style.PAPER)
        ax.annotate("M = ∞", (xs.max() + 2, m(wh, "pt")[inf_i][0]), xytext=(0, 12), textcoords="offset points", ha="center", color=style.MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "1246" else ""))
    ax.set_xlabel("number of pretraining tasks  $M$  (log$_2$)")
    ax.set_title(title)
    ax.set_xticks(range(0, xs.max() + 1, 2)); ax.set_xticklabels([f"$2^{{{int(v)}}}$" for v in range(0, xs.max() + 1, 2)])
axs[0].set_ylabel("MSE / D  (avg over 16 positions)")
style.label_end(axs[0], xs[0], m("true", "dmmse")[fin][0], "dMMSE (Bayes for the task pool)", DM, dx=6, dy=10, ha="left")
axs[0].annotate("ridge (Bayes for all tasks)", (xs[-1], m("true", "ridge")[fin].mean()), xytext=(0, -12), textcoords="offset points",
                ha="right", color=style.INK, fontsize=10, fontweight="bold")
axs[0].annotate("transformer", (xs[2], m("true", "pt")[fin][2]), xytext=(8, -14), textcoords="offset points", color=PT, fontsize=10, fontweight="bold")
ax = axs[2]
dr, dd = m("true", "d_pt_ridge")[fin], m("true", "d_pt_dmmse")[fin]
ax.plot(xs, dd, color=DM, lw=2.2, marker="o", ms=5, mec=style.PAPER)
ax.plot(xs, dr, color=style.INK, lw=2.2, marker="o", ms=5, mec=style.PAPER)
ax.set_yscale("log"); ax.set_xlabel("number of pretraining tasks  $M$  (log$_2$)")
ax.set_xticks(range(0, xs.max() + 1, 2)); ax.set_xticklabels([f"$2^{{{int(v)}}}$" for v in range(0, xs.max() + 1, 2)])
ax.set_title("whose predictions does it copy?  (new tasks)")
ax.set_ylabel("mean squared prediction gap / D")
cross = None
s = np.sign(np.log(dr) - np.log(dd))
for i in range(len(xs) - 1):
    if s[i] > 0 and s[i + 1] <= 0:
        f = (np.log(dr[i]) - np.log(dd[i])) / ((np.log(dr[i]) - np.log(dd[i])) - (np.log(dr[i + 1]) - np.log(dd[i + 1])))
        cross = xs[i] + f * (xs[i + 1] - xs[i])
if cross is not None:
    for axx in axs:
        axx.axvline(cross, color="#c9c4ba", lw=1)
    ax.annotate(f"crossover  $M \\approx 2^{{{cross:.1f}}}$", (cross, ax.get_ylim()[1]), xytext=(4, -14), textcoords="offset points", color=style.MUTED, fontsize=10)
style.label_end(ax, xs[-1], dd[-1], "gap to dMMSE", DM, dx=-4, dy=-14, ha="right")
style.label_end(ax, xs[1], dr[1], "gap to ridge", style.INK, dx=8, dy=4, ha="left")
fig.savefig(HERE / "figures" / f"taskdiv{'' if a.tag == 'main' else '_' + a.tag}.png")
json.dump({"crossover_log2M": cross, "final_step": ev["step"]}, open(HERE / "cache" / f"taskdiv_{a.tag}_summary.json", "w"))
print("crossover log2 M =", cross)

# ------------------------------------------------------------------------------------------ animation
if a.anim:
    FR = HERE / "_frames" / "taskdiv"
    FR.mkdir(parents=True, exist_ok=True)
    for f_ in FR.glob("*.png"):
        f_.unlink()
    pos = np.arange(1, K + 1)
    curves = {k: np.array(ev["true"][k]) for k in ["pt", "dmmse", "ridge"]}
    curves_pre = {k: np.array(ev["pretrain"][k]) for k in ["pt", "dmmse", "ridge"]}
    PER, HOLD = 26, 14
    ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * u)
    frames = []
    for j in range(len(fin) - 1):
        for q in range(PER):
            frames.append((j, ease(q / PER)))
    frames += [(len(fin) - 2, 1.0)] * HOLD
    for fi, (j, u) in enumerate(frames):
        i0, i1 = fin[j], fin[j + 1]
        lm = (1 - u) * xs[j] + u * xs[j + 1]
        lerp = lambda arr: np.exp((1 - u) * np.log(arr[i0]) + u * np.log(arr[i1]))
        fig = plt.figure(figsize=(16, 7.2), dpi=100)
        fig.text(0.05, 0.93, "How many tasks does a transformer need to see before it stops memorizing?", fontsize=22, fontweight="bold", color=style.INK)
        fig.text(0.05, 0.885, f"GPT-2-style transformer (4 layers, width 64) pretrained on a pool of $M$ regression tasks, D = {D}, noise variance {h['args']['noise_var']}",
                 fontsize=13, color=style.MUTED)
        for k_, (wh, cv, ttl) in enumerate([("true", curves, "on NEW tasks"), ("pretrain", curves_pre, "on tasks from its own pool")]):
            ax = fig.add_axes([0.05 + 0.33 * k_, 0.12, 0.28, 0.66])
            ax.plot(pos, lerp(cv["ridge"]), color=style.INK, lw=2)
            ax.plot(pos, lerp(cv["dmmse"]), color=DM, lw=2)
            ax.plot(pos, lerp(cv["pt"]), color=PT, lw=3)
            ax.set_yscale("log"); ax.set_ylim(0.02, 2.5); ax.set_xlim(1, K)
            ax.set_xlabel("context position $k$ (examples seen + 1)")
            ax.set_title(ttl)
            if k_ == 0:
                ax.set_ylabel("MSE / D")
                yk = lambda arr: lerp(arr)[-1]
                labs = [("transformer", yk(cv["pt"]), PT), ("dMMSE", yk(cv["dmmse"]), DM), ("ridge", yk(cv["ridge"]), style.INK)]
                labs.sort(key=lambda t: t[1])
                last = None
                for name, yv, col in labs:
                    yy = yv if last is None or np.log(yv / last) > 0.25 else last * np.exp(0.25)
                    ax.annotate(name, (K, yy), xytext=(6, 0), textcoords="offset points", color=col, fontsize=11, fontweight="bold", va="center", annotation_clip=False)
                    last = yy
        bx = fig.add_axes([0.72, 0.12, 0.25, 0.66])
        bx.plot(xs, dd, color=DM, lw=2); bx.plot(xs, dr, color=style.INK, lw=2)
        bx.scatter([lm], [np.exp(np.interp(lm, xs, np.log(dd)))], color=DM, s=70, zorder=5, edgecolor=style.PAPER, lw=1.5)
        bx.scatter([lm], [np.exp(np.interp(lm, xs, np.log(dr)))], color=style.INK, s=70, zorder=5, edgecolor=style.PAPER, lw=1.5)
        bx.axvline(lm, color=PT, lw=1.5, alpha=0.6)
        bx.set_yscale("log"); bx.set_title("gap between transformer and ...")
        bx.set_xlabel("$\\log_2 M$")
        bx.text(0.03, 0.05, "dMMSE", color=DM, transform=bx.transAxes, fontweight="bold")
        bx.text(0.03, 0.12, "ridge", color=style.INK, transform=bx.transAxes, fontweight="bold")
        fig.text(0.72, 0.84, f"M = 2^{lm:4.1f} ≈ {2**lm:,.0f} tasks", fontsize=18, color=PT, fontweight="bold")
        fig.savefig(FR / f"f{fi:04d}.png", facecolor=style.PAPER)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(FR / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", "-vf", "scale=1600:-2", str(HERE / "figures" / f"taskdiv_sweep{'' if a.tag == 'main' else '_' + a.tag}.mp4")], check=True)
    print("wrote animation", len(frames), "frames")
