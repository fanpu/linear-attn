"""Build-on figures from cache/seq_eval.json (analysis_seq.py).

  figures/race.png          risk vs context length, small multiples by depth, four token mixers + classical algorithms
  figures/fingerprint.png   distance of each trained model's predictions to each algorithm's predictions
  figures/shift.png         test-time input rescaling and label-noise shift

  .venv/bin/python 08-icl-linear-attention/render_seq.py
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style

HERE = pathlib.Path(__file__).parent
R = json.load(open(HERE / "cache" / "seq_eval.json"))
t = np.array(R["t"])
D = 10
KINDS = ["softmax", "linear", "delta", "gdelta"]
style.use_light()
M = R["models"]


def get(kind, L, sigma=0.0):
    for v in M.values():
        if v["kind"] == kind and v["L"] == L and abs(v["sigma"] - sigma) < 1e-9:
            return v
    return None


ALG_STYLE = {"ridge": ("ridge = RLS (optimal)", style.INK, 1.6, "-"), "gd1": ("GD, 1 step", "#c9c4ba", 1.6, "-"),
             "gd4": ("GD, 4 tuned steps", "#8f887c", 1.6, "-"), "nlms": ("online SGD (normalized LMS)", "#b9b2a6", 1.4, (0, (3, 2)))}
summary = {}

# ------------------------------------------------------------------------------------------ race
Ls = sorted({v["L"] for v in M.values() if v["sigma"] == 0})
fig, axs = plt.subplots(1, len(Ls), figsize=(5.2 * len(Ls), 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
algs = R["algs"]["0.0"]["risk"]
for j, (ax, L) in enumerate(zip(np.atleast_1d(axs), Ls)):
    for a, (name, col, lw, ls) in ALG_STYLE.items():
        ax.plot(t[1:], np.array(algs[a])[1:] / D, color=col, lw=lw, ls=ls)
    for k in KINDS:
        v = get(k, L)
        if v is None:
            continue
        r = np.array(v["risk"]) / D
        ax.plot(t[1:], r[1:], color=style.C[k], lw=2.6)
        if j == len(Ls) - 1:
            style.label_end(ax, t[-1], r[-1], style.NAMES[k], style.C[k], dx=6)
        summary[f"risk_t40_{k}_L{L}"] = float(r[-1]); summary[f"risk_t20_{k}_L{L}"] = float(r[19])
    ax.axvline(D, color="#c9c4ba", lw=1)
    ax.set_yscale("log"); ax.set_ylim(3e-5, 1.3); ax.set_xlim(1, 40)
    ax.set_title(f"{L} layer{'s' if L > 1 else ''}")
    ax.set_xlabel("context examples seen  $t$")
    if j == 0:
        ax.set_ylabel("excess risk / d   (d = 10, noiseless)")
        for a, (name, col, lw, ls) in ALG_STYLE.items():
            y = np.array(algs[a])[-1] / D
            ax.annotate(name, (t[-1], max(y, 4e-5)), xytext=(-4, 7), textcoords="offset points", ha="right", color=col if col != "#c9c4ba" else "#a39c90",
                        fontsize=9, fontweight="bold")
        ax.text(D + 0.5, 1.0, "t = d", color=style.MUTED, fontsize=9)
for a in ALG_STYLE:
    summary[f"alg_t40_{a}"] = float(np.array(algs[a])[-1] / D); summary[f"alg_t20_{a}"] = float(np.array(algs[a])[19] / D)
fig.savefig(HERE / "figures" / "race.png")
plt.close(fig)

# ------------------------------------------------------------------------------------------ fingerprint
ALGS = ["ridge", "gd1", "gd2", "gd3", "gd4", "nlms", "lms"]
ALGN = ["ridge\n(RLS)", "GD\n1 step", "GD\n2 steps", "GD\n3 steps", "GD\n4 steps", "online SGD\nnormalized", "online SGD\nplain"]
rows = [(k, L) for k in KINDS for L in Ls if get(k, L) is not None]
win = (t >= 11) & (t <= 40)
H = np.zeros((len(rows), len(ALGS)))
for i, (k, L) in enumerate(rows):
    v = get(k, L)
    for jj, a in enumerate(ALGS):
        H[i, jj] = np.mean(np.array(v["dist"][a])[win]) / D
cmap = LinearSegmentedColormap.from_list("seq_blue", ["#184f95", "#3987e5", "#9ec5f4", "#e8f0fb", "#fcfbf8"])
fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(rows) + 1.8))
im = ax.imshow(np.log10(H), cmap=cmap, aspect="auto", vmin=np.log10(H.min()) - 0.1, vmax=0)
for i in range(len(rows)):
    jbest = int(np.argmin(H[i]))
    for jj in range(len(ALGS)):
        ax.text(jj, i, f"{H[i, jj]:.3f}" if H[i, jj] >= 0.001 else f"{H[i, jj]:.0e}", ha="center", va="center", fontsize=9,
                color="#fcfbf8" if np.log10(H[i, jj]) < np.log10(H.min()) * 0.5 else style.INK, fontweight="bold" if jj == jbest else "normal")
    ax.add_patch(plt.Rectangle((jbest - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=style.INK, lw=1.6))
    summary[f"closest_{rows[i][0]}_L{rows[i][1]}"] = ALGS[jbest]
ax.set_xticks(range(len(ALGS))); ax.set_xticklabels(ALGN, fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{style.NAMES[k]}, {L}L" for k, L in rows])
for tl, (k, L) in zip(ax.get_yticklabels(), rows):
    tl.set_color(style.C[k]); tl.set_fontweight("bold")
ax.xaxis.tick_top(); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(1, len(rows)):
    if rows[i][0] != rows[i - 1][0]:
        ax.axhline(i - 0.5, color=style.PAPER, lw=4)
fig.text(0.01, 0.01, "cell = mean squared difference between the model's and the algorithm's predictions, per dimension, averaged over t = 11…40 (boxed: closest)",
         fontsize=9, color=style.MUTED)
fig.savefig(HERE / "figures" / "fingerprint.png")
plt.close(fig)

# ------------------------------------------------------------------------------------------ shift
Lsh = 2 if 2 in Ls else Ls[0]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
cs = np.array(R["scales"])
late = t >= 30
sc_alg = R["algs"]["0.0"]["scale"]
for a, (name, col, lw, ls) in ALG_STYLE.items():
    y = [np.mean(np.array(sc_alg[str(c)][a])[late]) / D for c in cs]
    ax.plot(cs, np.maximum(y, 1e-6), color=col, lw=lw, ls=ls)
    style.label_end(ax, cs[0], max(y[0], 1e-6), name, col if col != "#c9c4ba" else "#a39c90", dx=-6, ha="right", fontsize=9)
for k in KINDS:
    v = get(k, Lsh)
    if v is None:
        continue
    y = [np.mean(np.array(v["scale"][str(c)])[late]) / D for c in cs]
    ax.plot(cs, y, color=style.C[k], lw=2.6, marker="o", ms=5, mec=style.PAPER)
    style.label_end(ax, cs[-1], y[-1], style.NAMES[k], style.C[k], dx=6)
    summary[f"scale_{k}_L{Lsh}"] = dict(zip(map(str, cs), y))
ax.axvline(1, color="#c9c4ba", lw=1)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(1e-5, 30)
ax.set_xticks(cs); ax.set_xticklabels([f"{c:g}" for c in cs])
ax.set_xlabel("test inputs rescaled  $x \\to c\\,x$   (trained at c = 1)")
ax.set_ylabel("excess risk / ($c^2 d$), t = 30…40")
ax.set_title(f"a  change the input scale ({Lsh}-layer models)")
ax = axs[1]
ns = np.array(R["noises"])
nz_alg = R["algs"]["0.5"]["noise"] if "0.5" in R["algs"] else None
for a, (name, col, lw, ls) in ALG_STYLE.items():
    y = [np.mean(np.array(nz_alg[str(s)][a])[late]) / D for s in ns]
    ax.plot(ns, y, color=col, lw=lw, ls=ls)
for k in KINDS:
    v = get(k, 2, 0.5)
    if v is None:
        continue
    y = [np.mean(np.array(v["noise"][str(s)])[late]) / D for s in ns]
    ax.plot(ns, y, color=style.C[k], lw=2.6, marker="o", ms=5, mec=style.PAPER)
    style.label_end(ax, ns[-1], y[-1], style.NAMES[k], style.C[k], dx=6)
    summary[f"noise_{k}_L2"] = dict(zip(map(str, ns), y))
ax.axvline(0.5, color="#c9c4ba", lw=1)
ax.set_yscale("log")
ax.set_xlabel("test label noise  $\\sigma$   (trained at σ = 0.5)")
ax.set_ylabel("excess risk / d, t = 30…40")
ax.set_title("b  change the noise level (2-layer models)")
fig.savefig(HERE / "figures" / "shift.png")
plt.close(fig)
json.dump(summary, open(HERE / "cache" / "seq_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
