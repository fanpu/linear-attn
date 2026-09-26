"""Plates in the surveyor's register."""
import argparse, json, os, textwrap
import numpy as np
import matplotlib.pyplot as plt
from atlas import (Map, PAPER, INK, RED, FAINT, RULE, TASK_TITLE, ARCH_NAME, trig_point, city, road, neat_line, save)

ROOT = os.path.dirname(os.path.abspath(__file__))


def straight(ax, M, lw=0.7):
    G = np.vstack([M.X[M.i0], M.points("geo"), M.X[M.i1]])
    ax.plot(G[:, 0], G[:, 1], color=RED, lw=lw, ls=(0, (6, 4)), zorder=4)


def draw_map(ax, M, show_real=True, show_null=False, show_shuf=False, lw=0.6, sym=1.0, labels=True,
             inits=True, fs=7, real_color=INK):
    straight(ax, M, lw=0.6 * sym)
    if show_null:
        for ri in M.run_ids():
            road(ax, M.traj("perm", ri), color=RED, lw=lw, alpha=0.8, z=5)
    if show_real:
        for ri in M.run_ids():
            road(ax, M.traj("run", ri), color=real_color, lw=lw, alpha=0.85, z=6)
    if show_shuf:
        for ri in M.run_ids("shuf"):
            road(ax, M.traj("run", ri), color=RED, lw=lw * 1.3, alpha=0.9, z=6)
    if inits:
        I = M.points("init")
        ax.scatter(I[:, 0], I[:, 1], marker="+", s=10 * sym, color=INK, linewidths=0.5 * sym, zorder=7)
    trig_point(ax, M.X[M.i0], s=sym); city(ax, M.X[M.i1], s=sym)
    if labels:
        ax.text(M.X[M.i0, 0], M.X[M.i0, 1] - 0.045, "IGNORANCE", ha="center", va="top", fontsize=fs, zorder=12)
        ax.text(M.X[M.i1, 0], M.X[M.i1, 1] - 0.045, "TRUTH", ha="center", va="top", fontsize=fs, zorder=12)


def extent(Ms, pad=0.08, kinds=("run", "perm", "init", "geo")):
    P = np.vstack([np.vstack([M.X[[M.pos[g] for g in M.idx if M.L["kinds"][g] in kinds]], M.X[[M.i0, M.i1]]]) for M in Ms])
    lo, hi = P[:, :2].min(0) - pad, P[:, :2].max(0) + pad
    lo[1] = min(lo[1], -0.11)   # room for the landmark names under the baseline
    return (lo[0], hi[0]), (lo[1], hi[1])


def null_diptych(task, split="tr"):
    M = Map(task, split, "null")
    xl, yl = extent([M])
    w = xl[1] - xl[0]; h = yl[1] - yl[0]
    fig = plt.figure(figsize=(14, 14 * (2 * h + 0.35) / w * 0.5 + 1.2))
    H = 2 * h + 0.5
    for j, (real, null, cap) in enumerate(((True, False, "the roads the networks took"),
                                            (False, True, "the same roads, examples shuffled within each class"))):
        ax = fig.add_axes([0.06, 0.08 + (1 - j) * 0.46, 0.88, 0.40])
        draw_map(ax, M, show_real=real, show_null=null)
        neat_line(ax, xl, yl)
        ax.text(xl[0], yl[1] + 0.03, cap, fontsize=9, style="italic", va="bottom")
    fig.text(0.06, 0.035, f"{TASK_TITLE[task]} — {'train' if split == 'tr' else 'test'} probe, 1,000 examples. "
             f"Both panels are one intensive-PCA embedding; the dashed red line is the straight road.",
             fontsize=8, color=FAINT)
    return save(fig, f"study_null_{task}_{split}.png", dpi=200)




def hero(tasks=("mnist", "fashion", "cifar", "modadd"), split="tr", name="one_road_atlas", width=16.0, metric="B",
         right="null"):
    """Typology: every task a strip map; left the roads the networks took, right a null in the same embedding.
    One shared frame: P0 -> P* has the same length and the same y range on every panel (declared)."""
    Ms = [Map(t, split, "null" if right == "null" else "memo", metric) for t in tasks]
    # one horizontal scale for every row (declared); each row is cropped to its own height
    xl = extent(Ms, pad=0.06)[0]
    yls = [extent([M], pad=0.06)[1] for M in Ms]
    w = xl[1] - xl[0]
    n = len(Ms)
    mx, gap, top, bot, rowgap = 0.045, 0.035, 1.75, 0.8, 0.5
    pw = (width - 2 * mx * width - gap * width) / 2
    phs = [pw * (yl[1] - yl[0]) / w for yl in yls]
    H = top + sum(phs) + (n - 1) * rowgap + bot
    fig = plt.figure(figsize=(width, H))
    suf = "" if metric == "B" else "_" + metric
    for r, M in enumerate(Ms):
        met = json.load(open(os.path.join(ROOT, "cache", f"metrics_{M.task}{suf}.json")))[split]
        yl = yls[r]; ph = phs[r]
        y0 = (bot + sum(phs[r + 1:]) + (n - 1 - r) * rowgap) / H
        for c in range(2):
            ax = fig.add_axes([(mx * width + c * (pw + gap * width)) / width, y0, pw / width, ph / H])
            if right == "null":
                draw_map(ax, M, show_real=(c == 0), show_null=(c == 1), lw=0.75, sym=0.8, fs=6.5)
            else:
                draw_map(ax, M, show_real=(c == 0), show_shuf=(c == 1), lw=0.55, sym=0.8, fs=6.5)
            neat_line(ax, xl, yl, lw=0.7)
            if c == 0:
                ax.text(xl[0], yl[1] + 0.03 * w, f"{['I', 'II', 'III', 'IV', 'V'][r]}.  {TASK_TITLE[M.task]}",
                        fontsize=11, va="bottom")
                nr = len(M.run_ids())
                ax.text(xl[1], yl[1] + 0.03 * w,
                        f"{nr} roads, {len(set(M.runs[i]['arch'] for i in M.run_ids()))} architectures",
                        fontsize=7.5, va="bottom", ha="right", style="italic", color=FAINT)
            elif right == "null":
                dt = met["dtraj_arch"]
                ax.text(xl[1], yl[1] + 0.03 * w,
                        f"between architectures d = {dt['median']:.3f};  against the shuffled road {dt['median_perm']:.3f}",
                        fontsize=7.5, va="bottom", ha="right", style="italic", color=RED)
            else:
                ax.text(xl[1], yl[1] + 0.03 * w, f"{len(M.run_ids('shuf'))} roads to a truth of random labels",
                        fontsize=7.5, va="bottom", ha="right", style="italic", color=RED)
    cy = 1 - (top - 0.28) / H
    lx = mx + pw / width / 2; rx = mx + (pw + gap * width) / width + pw / width / 2
    fig.text(lx, cy, "THE ROADS THE NETWORKS TOOK", ha="center", fontsize=9.5)
    fig.text(rx, cy, ("THE SAME ROADS, EXAMPLES SHUFFLED WITHIN THEIR CLASS" if right == "null"
                      else "THE ROADS TO MEMORISED RANDOM LABELS"), ha="center", fontsize=9.5, color=RED)
    title = "ONE ROAD" if right == "null" else "THE OTHER ROAD"
    sub = ("an atlas of the roads small networks take from ignorance to truth" if right == "null"
           else "networks fitting shuffled labels leave ignorance by a different road")
    if split == "te":
        sub = "the same atlas, surveyed on examples the networks never saw"
    fig.text(0.5, 1 - 0.42 / H, title, ha="center", va="center", fontsize=26)
    fig.text(0.5, 1 - 0.8 / H, sub, ha="center", va="center", fontsize=11, style="italic")
    probe = "training" if split == "tr" else "held-out test"
    dist = ("intensive PCA of Bhattacharyya distances (Mao et al., PNAS 2024)" if metric == "B" else
            "PCA of per-example Hellinger distances, i.e. of sqrt-probabilities (a robustness check on Mao et al.'s InPCA)")
    fig.text(mx, 0.3 / H, f"Each road is one training run: its softmax predictions on 1,000 {probe} examples at ~90 log-spaced "
             f"checkpoints, placed by {dist}. Each row is one embedding. "
             "Dashed: the straight road. +: untrained networks. d: median distance at matched progress.",
             fontsize=7, color=FAINT, va="center")
    return save(fig, f"{name}_{split}{suf}.png", dpi=400)


def memo_plate(tasks=("mnist", "fashion", "cifar", "modadd"), split="tr", width=16.0):
    """True-label roads (ink) and shuffled-label roads (red) in one embedding per task.
    Same orientation rule as every sheet: the true-label roads bow upward."""
    Ms = [Map(t, split, "memo") for t in tasks]
    xl, yl = extent(Ms, pad=0.08, kinds=("run", "init"))
    w = xl[1] - xl[0]; h = yl[1] - yl[0]
    mx, gap, top, bot, rowgap = 0.05, 0.04, 1.5, 0.8, 0.55
    pw = (width - 2 * mx * width - gap * width) / 2; ph = pw * h / w
    H = top + 2 * ph + rowgap + bot
    fig = plt.figure(figsize=(width, H))
    for j, M in enumerate(Ms):
        r, c = divmod(j, 2)
        ax = fig.add_axes([(mx * width + c * (pw + gap * width)) / width, (bot + (1 - r) * (ph + rowgap)) / H, pw / width, ph / H])
        draw_map(ax, M, show_shuf=True, lw=0.6, sym=0.8, fs=6.5)
        neat_line(ax, xl, yl, lw=0.7)
        ax.text(xl[0], yl[1] + 0.02 * w, f"{['I', 'II', 'III', 'IV'][j]}.  {TASK_TITLE[M.task]}", fontsize=11, va="bottom")
        for ri in M.run_ids("shuf"):
            P = M.traj("run", ri); r_ = M.runs[ri]
            ax.text(P[-1, 0] + 0.02, P[-1, 1], ARCH_NAME[r_["arch"]] + ", random labels", fontsize=6.5,
                    style="italic", color=RED, va="center")
    fig.text(0.5, 1 - 0.42 / H, "THE OTHER ROAD", ha="center", va="center", fontsize=26)
    fig.text(0.5, 1 - 0.8 / H, "the same networks, taught shuffled labels, leave ignorance in another direction",
             ha="center", va="center", fontsize=11, style="italic")
    fig.text(mx, 0.3 / H, "Ink: the true-label roads. Red: runs trained to memorise a fixed random relabelling of the "
             "same 10,000 examples, surveyed on the same 1,000 training examples and their true labels. One embedding per panel.",
             fontsize=7, color=FAINT, va="center")
    return save(fig, f"other_road_{split}.png", dpi=250)


def grok_plate(metric="H", width=16.0):
    """a + b mod 97: the road on the examples it was taught, and on the ones it never saw."""
    Ms = [Map("modadd", sp, "main", metric) for sp in ("tr", "te")]
    xl = extent(Ms, pad=0.08, kinds=("run", "init", "geo"))[0]
    yls = [extent([M], pad=0.08, kinds=("run", "init", "geo"))[1] for M in Ms]
    w = xl[1] - xl[0]
    mx, top, bot, rowgap = 0.06, 1.6, 1.3, 0.75
    pw = width * (1 - 2 * mx); phs = [pw * (yl[1] - yl[0]) / w for yl in yls]
    H = top + sum(phs) + rowgap + bot
    fig = plt.figure(figsize=(width, H))
    acc = {}
    for ri, r in enumerate(Ms[0].runs):
        z = np.load(os.path.join(ROOT, "cache", "runs", "modadd", r["name"] + ".npz"))
        acc[ri] = (z["tr_acc"], z["te_acc"], z["steps"])
    for j, M in enumerate(Ms):
        ax = fig.add_axes([mx, (bot + sum(phs[j + 1:]) + (1 - j) * rowgap) / H, pw / width, phs[j] / H])
        draw_map(ax, M, lw=0.8, sym=1.1, fs=8.5)
        placed = []
        for ri in M.run_ids():
            P = M.traj("run", ri); tra, tea, st = acc[ri]
            im = int(np.argmax(tra >= 0.99)) if (tra >= 0.99).any() else None
            ig = int(np.argmax(tea >= 0.99)) if (tea >= 0.99).any() else None
            if im is not None:
                ax.scatter([P[im, 0]], [P[im, 1]], s=26, facecolor=PAPER, edgecolor=INK, linewidths=0.8, zorder=8)
            if ig is not None:
                ax.scatter([P[ig, 0]], [P[ig, 1]], s=22, marker="s", color=INK, zorder=8)
        label_roads(ax, M, fs=8)
        neat_line(ax, xl, yls[j], lw=0.8)
        ax.text(xl[0], yls[j][1] + 0.02 * w, ["I.  On the pairs it was taught", "II.  On the pairs it never saw"][j],
                fontsize=12, va="bottom")
    fig.text(0.5, 1 - 0.45 / H, "THE LONG WAY ROUND", ha="center", va="center", fontsize=24)
    fig.text(0.5, 1 - 0.85 / H, "a + b mod 97: on the pairs it was taught every road arrives; on the pairs it never saw, "
             "the road that generalises first walks away from the truth, then comes back", ha="center", va="center", fontsize=11, style="italic")
    fig.text(mx, 0.75 / H, "○  training accuracy first reaches 99% (memorised)      "
             "■  held-out accuracy first reaches 99% (grokked)      +  untrained networks      dashed: the straight road",
             fontsize=8.5, va="center")
    fig.text(mx, 0.35 / H, textwrap.fill("Each road: softmax predictions on 1,000 pairs at 120 log-spaced steps up to 20,000 "
             "(full-batch). AdamW with weight decay 1 groks; Adam without weight decay does not. Embedding: " + (
             "InPCA of per-example Hellinger distances (declared: the Bhattacharyya distance is dominated by 1e-11 probabilities "
             "once held-out loss reaches 25 nats, and stops behaving like a distance)." if metric == "H" else
             "InPCA of Bhattacharyya distances, as in Mao et al. On the held-out panel this is distorted by 1e-11 probabilities."),
             200), fontsize=7, color=FAINT, va="center")
    return save(fig, f"long_way_round_modadd{'' if metric == 'B' else '_' + metric}.png", dpi=250)


def label_roads(ax, M, fs=7, color=INK, which="true", at="apex", dy=0.018):
    """Place-name labels: one per architecture, at the apex of its first road (declared placement)."""
    done, placed = set(), []
    for ri in M.run_ids(which):
        r = M.runs[ri]
        key = (r["arch"], r["opt"]) if which == "true" else r["name"]
        if key in done:
            continue
        done.add(key)
        P = M.traj("run", ri)
        i = int(np.argmax(P[:, 1])) if at == "apex" else len(P) - 1
        x, y = P[i, 0], P[i, 1] + dy
        while any(abs(x - px) < 0.12 and abs(y - py) < 0.022 for px, py in placed):
            y += 0.024
        placed.append((x, y))
        ax.plot([P[i, 0], x], [P[i, 1], y], color=FAINT, lw=0.4, zorder=14)   # leader line to the apex it names
        if M.task == "modadd":
            txt = ARCH_NAME[r["arch"]] + (", AdamW wd 1" if r["opt"] == "adamw" else ", Adam, no wd")
        else:
            txt = ARCH_NAME[r["arch"]] + ("" if r["opt"] in ("adam",) else f" ({r['opt'].upper() if r['opt'] == 'sgd' else 'AdamW'})")
        ax.text(x, y, txt, fontsize=fs, style="italic", color=color, ha="center", va="bottom", zorder=15)


def task_plate(task, split="tr", metric="B"):
    M = Map(task, split, "main", metric)
    xl, yl = extent([M], pad=0.1)
    w = xl[1] - xl[0]; h = yl[1] - yl[0]
    W = 16; ph = W * 0.9 * h / w
    fig = plt.figure(figsize=(W, ph + 2.2))
    ax = fig.add_axes([0.05, 1.0 / (ph + 2.2), 0.9, ph / (ph + 2.2)])
    draw_map(ax, M, lw=0.7, sym=1.1, fs=9)
    label_roads(ax, M, fs=8.5)
    neat_line(ax, xl, yl)
    ax.text(xl[0], yl[1] + 0.03 * w, f"The roads of {TASK_TITLE[task]}", fontsize=18, va="bottom")
    ax.text(xl[1], yl[1] + 0.03 * w, f"{'training' if split == 'tr' else 'held-out'} probe · top two of intensive PCA, "
            f"{100 * M.stress[1]:.0f}% of stress", fontsize=9, va="bottom", ha="right", style="italic", color=FAINT)
    return save(fig, f"plate_roads_{task}_{split}{'' if metric == 'B' else '_' + metric}.png", dpi=250)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", nargs="+"); a = ap.parse_args()
    for w in a.what:
        if w == "hero3": hero(("mnist", "fashion", "cifar"))
        elif w == "hero": hero()
        elif w == "hero_te": hero(("mnist", "fashion", "cifar"), split="te")
        elif w == "hero_H": hero(metric="H")
        elif w == "memo": memo_plate()
        elif w == "grok": grok_plate(); grok_plate("B")
        elif w.startswith("null_"): null_diptych(w[5:])
        elif w.startswith("plate_"): task_plate(*w[6:].split(":"))
        else: globals()[w]()
