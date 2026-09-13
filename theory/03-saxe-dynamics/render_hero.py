"""Hero animation: a deep linear network 'grows a tree of knowledge'.

Left: a dendrogram whose branch lengths are *measured* distances between the hidden representations
(h_i = W1 x_i, 16-d) of groups of items at time t: trunk = |centroid of all items|, fork = |c_animal - c_all|,
twig = |c_bird - c_animal|, leaf stem = |h_canary - c_bird|. Angles are a fixed layout choice.
Right: mode strengths (measured dots/lines, analytic thin lines) and the loss, with a time cursor.

  .venv/bin/python 03-saxe-dynamics/render_hero.py [--frames N] [--still]
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
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
FR = HERE / "cache" / "frames_hero"

D = np.load(HERE / "cache/semantic.npz")
T_REC = D["t"]
S = D["s"]
ITEMS = [str(x) for x in D["items"]]
W1 = D["W1"]  # (R, H, P)
H_REP = np.einsum("rhi,ij->rhj", W1, np.sqrt(8) * np.eye(8))  # (R, H, items) columns
MODES = D["modes"]
LOSS = D["loss"]
U0 = D["u0_eff"]

GROUPS = dict(all=list(range(8)), animal=[0, 1, 2, 3], plant=[4, 5, 6, 7], bird=[0, 1], fish=[2, 3], tree=[4, 5], flower=[6, 7])
MODE_NAMES = ["living thing", "animal  vs  plant", "bird  vs  fish", "tree  vs  flower", "each individual item"]
# layout: node -> (parent, angle in degrees, mode index for colour)
def _angles():
    tree = [("all", None, 90.0, 0)]
    ang = {"all": 90.0}
    for child, parent, sign, spread, m in [("animal", "all", +1, 30, 1), ("plant", "all", -1, 30, 1),
                                          ("bird", "animal", +1, 24, 2), ("fish", "animal", -1, 24, 2),
                                          ("tree", "plant", +1, 24, 3), ("flower", "plant", -1, 24, 3)]:
        a = ang[parent] + sign * spread - 0.25 * (ang[parent] - 90)
        ang[child] = a
        tree.append((child, parent, a, m))
    leaf = {}
    for g in ("bird", "fish", "tree", "flower"):
        a = ang[g] - 0.3 * (ang[g] - 90)
        leaf[g] = (a + 13, a - 13)
    return tree, leaf


TREE, LEAF_ANGLE = _angles()


def interp(arr, t):
    i = np.clip(np.searchsorted(T_REC, t) - 1, 0, len(T_REC) - 2)
    w = (t - T_REC[i]) / (T_REC[i + 1] - T_REC[i])
    w = np.clip(w, 0, 1)
    return arr[i] * (1 - w) + arr[i + 1] * w


def tree_geometry(t):
    Hc = interp(H_REP, t)  # (H, 8)
    cent = {g: Hc[:, idx].mean(1) for g, idx in GROUPS.items()}
    pos, segs = {}, []
    origin = np.array([0.0, 0.0])
    for node, parent, ang, m in TREE:
        p0 = origin if parent is None else pos[parent][0]
        pdir = 90 if parent is None else pos[parent][1]
        ref = np.zeros_like(cent[node]) if parent is None else cent[parent]
        ln = np.linalg.norm(cent[node] - ref)
        a = np.deg2rad(ang)
        p1 = p0 + ln * np.array([np.cos(a), np.sin(a)])
        pos[node] = (p1, ang)
        segs.append((p0, p1, pdir, ang, m, node))
    leaves = []
    for g, (a1, a2) in LEAF_ANGLE.items():
        p0, pang = pos[g]
        for k, it in enumerate(GROUPS[g]):
            ln = np.linalg.norm(Hc[:, it] - cent[g])
            a = np.deg2rad((a1, a2)[k])
            p1 = p0 + ln * np.array([np.cos(a), np.sin(a)])
            segs.append((p0, p1, pang, (a1, a2)[k], 4, ITEMS[it]))
            leaves.append((p1, ITEMS[it], (a1, a2)[k]))
    return segs, leaves


def branch(ax, p0, p1, start_ang, lw, color, glow=True):
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return
    d0 = np.array([np.cos(np.deg2rad(start_ang)), np.sin(np.deg2rad(start_ang))])
    ctrl = p0 + d0 * L * 0.45
    path = Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    if glow:
        for w, al in ((lw * 5, 0.05), (lw * 2.6, 0.10)):
            ax.add_patch(PathPatch(path, fc="none", ec=color, lw=w, alpha=al, capstyle="round"))
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, capstyle="round", joinstyle="round"))


def _limits():
    segs, leaves = tree_geometry(T_REC[-1])
    pts = np.array([p for sg in segs for p in sg[:2]])
    x0, x1 = pts[:, 0].min() - 0.9, pts[:, 0].max() + 0.9
    y0, y1 = -0.3, pts[:, 1].max() + 0.45
    aspect = (0.56 * 16) / (0.80 * 9)
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        cx = 0.5 * (x0 + x1); w = h * aspect; x0, x1 = cx - w / 2, cx + w / 2
    else:
        h = w / aspect; y1 = y0 + h
    return (x0, x1), (y0, y1)


XLIM, YLIM = _limits()


def draw_frame(args):
    fi, t, out, alpha_fade = args
    P = style.use("dark")
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(style.NIGHT)
    # ---- tree
    ax = fig.add_axes([0.02, 0.03, 0.56, 0.80])
    ax.set_facecolor(style.NIGHT)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal"); ax.axis("off")
    # ghost of the final tree
    segs_f, leaves_f = tree_geometry(T_REC[-1])
    for p0, p1, a0, a1, m, name in segs_f:
        branch(ax, p0, p1, a0, 1.0, style.NIGHT_RULE, glow=False)
    ax.plot([-2.2, 2.2], [0, 0], color=style.NIGHT_RULE, lw=1.2, solid_capstyle="round")
    segs, leaves = tree_geometry(t)
    widths = [7, 4.5, 4.5, 3, 3, 3, 3] + [1.8] * 8
    for k, (p0, p1, a0, a1, m, name) in enumerate(segs):
        branch(ax, p0, p1, a0, widths[k], P["modes"][m])
    ax.scatter([0], [0], s=40, color=P["modes"][0], zorder=5)
    fin = {sg[5]: np.linalg.norm(sg[1] - sg[0]) for sg in segs_f}
    cur = {sg[5]: (np.linalg.norm(sg[1] - sg[0]), sg[1], sg[3]) for sg in segs}

    def vis(name):
        x = np.clip((cur[name][0] / fin[name] - 0.25) / 0.45, 0, 1)
        return x * x * (3 - 2 * x)

    children = dict(all=["animal", "plant"], animal=["bird", "fish"], plant=["tree", "flower"],
                    bird=["canary", "robin"], fish=["salmon", "sunfish"], tree=["oak", "pine"], flower=["rose", "daisy"])
    plural = dict(all="living things", animal="animals", plant="plants", bird="birds", fish="fish", tree="trees",
                  flower="flowers")
    for node, kids in children.items():
        al = vis(node) * (1 - max(vis(k) for k in kids))
        if node == "all":
            al = (1 - max(vis(k) for k in kids)) * np.clip(cur["all"][0] / fin["all"] * 3, 0, 1)
        if al > 0.02:
            _, p, ang = cur[node]
            right = p[0] >= 0 if node != "all" else True
            ax.text(p[0] + (0.2 if right else -0.2), p[1] + 0.05, plural[node], color=style.NIGHT_INK, alpha=al,
                    fontsize=12.5, ha="left" if right else "right", va="center", zorder=8, style="italic")
    for p, name, ang in leaves:
        al = vis(name)
        for sz, a_ in ((520, 0.05), (220, 0.10), (80, 0.25)):
            ax.scatter([p[0]], [p[1]], s=sz, color="#fff4d6", alpha=a_, lw=0, zorder=6)
        ax.scatter([p[0]], [p[1]], s=22, color="#ffffff", lw=0, zorder=7)
        if al < 0.02:
            continue
        a = np.deg2rad(ang)
        off = np.array([np.cos(a), np.sin(a)]) * 0.22
        ha = "center" if abs(np.cos(a)) < 0.35 else ("left" if np.cos(a) > 0 else "right")
        ax.text(p[0] + off[0], p[1] + off[1], name, color=style.NIGHT_INK, alpha=al, fontsize=12.5, ha=ha,
                va="center", zorder=8)
    fig.text(0.035, 0.925, "A linear network learning about living things", fontsize=22, color=style.NIGHT_INK,
             weight="bold")
    fig.text(0.035, 0.868, "branch length = measured distance between the hidden representations of each group\n"
             "branch colour = the distinction (singular mode) that grows it", fontsize=12.5, color=style.NIGHT_MUTED,
             linespacing=1.5)

    # ---- mode rows
    tt = np.linspace(0, 16, 600)
    x0, w = 0.62, 0.34
    row_h, gap = 0.07, 0.042
    top = 0.86
    ti = np.searchsorted(T_REC, t)
    for m in range(5):
        y = top - (m + 1) * row_h - m * gap
        axm = fig.add_axes([x0, y, w, row_h])
        axm.set_facecolor(style.NIGHT)
        idxs = [m] if m < 4 else [4, 5, 6, 7]
        th = np.mean([sc.sigmoid_mode(tt, S[i], U0[i]) / S[i] for i in idxs], axis=0)
        meas = np.mean([MODES[:, i] / S[i] for i in idxs], axis=0)
        c = P["modes"][m]
        axm.plot(tt, th, color="#ffffff", lw=0.8, alpha=0.45)
        axm.fill_between(T_REC[: ti + 1], 0, meas[: ti + 1], color=c, alpha=0.09, lw=0)
        axm.plot(T_REC[: ti + 1], meas[: ti + 1], color=c, lw=2.4)
        cur = float(interp(meas, t))
        axm.scatter([t], [cur], s=38, color=c, edgecolor=style.NIGHT, lw=1.5, zorder=5)
        axm.set_xlim(0, 16); axm.set_ylim(-0.08, 1.12)
        axm.axis("off")
        sval = S[m] if m < 4 else S[4]
        fig.text(x0, y + row_h + 0.004, MODE_NAMES[m], fontsize=13, color=style.NIGHT_INK, va="bottom")
        fig.text(x0 + w, y + row_h + 0.004, f"s = {sval:.2f}", fontsize=11, color=style.NIGHT_MUTED, va="bottom", ha="right")
    # ---- loss
    yl = top - 5 * row_h - 5 * gap - 0.17
    axl = fig.add_axes([x0, yl, w, 0.17])
    axl.set_facecolor(style.NIGHT)
    axl.plot(T_REC, LOSS, color="#ffffff", lw=0.8, alpha=0.3)
    axl.plot(T_REC[: ti + 1], LOSS[: ti + 1], color=style.NIGHT_INK, lw=2.2)
    axl.scatter([t], [float(interp(LOSS, t))], s=38, color=style.NIGHT_INK, edgecolor=style.NIGHT, lw=1.5, zorder=5)
    axl.set_xlim(0, 16); axl.set_ylim(0, LOSS[0] * 1.05)
    for sp in ("left", "top", "right"):
        axl.spines[sp].set_visible(False)
    axl.spines["bottom"].set_color(style.NIGHT_RULE)
    axl.set_yticks([]); axl.tick_params(axis="x", colors=style.NIGHT_MUTED, labelsize=10)
    fig.text(x0, yl + 0.175, "loss", fontsize=13, color=style.NIGHT_INK)
    fig.text(x0 + w, yl - 0.05, f"training time  t = {t:5.2f}", fontsize=12, color=style.NIGHT_MUTED, ha="right")
    if alpha_fade < 1:
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=style.NIGHT,
                                         alpha=1 - alpha_fade, zorder=100))
    fig.savefig(out, dpi=100, facecolor=style.NIGHT)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    if a.still:
        for t in (3.5, 5.0, 16.0):
            draw_frame((0, t, FIG / f"_hero_still_{t:g}.png", 1.0))
        return
    if FR.exists():
        shutil.rmtree(FR)
    FR.mkdir(parents=True)
    t_end = 16.0
    n_run = int(13.0 * a.fps)
    n_hold = int(1.6 * a.fps)
    n_fade = int(0.5 * a.fps)
    jobs = []
    for i in range(n_run):
        s = i / (n_run - 1)
        jobs.append((len(jobs), s * t_end, None, 1.0))
    for i in range(n_hold):
        jobs.append((len(jobs), t_end, None, 1.0))
    for i in range(n_fade):
        jobs.append((len(jobs), t_end, None, 1 - (i + 1) / n_fade))
    for i in range(n_fade // 2):
        jobs.append((len(jobs), 0.0, None, (i + 1) / (n_fade // 2)))
    jobs = [(fi, t, FR / f"f{fi:04d}.png", al) for fi, t, _, al in jobs]
    with mp.Pool(4) as pool:
        for k, _ in enumerate(pool.imap_unordered(draw_frame, jobs, chunksize=4)):
            if k % 60 == 0:
                print("frame", k, "/", len(jobs), flush=True)
    mp4 = FIG / "hero_tree.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(FR / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                    str(mp4)], check=True)
    shutil.copy(FR / f"f{n_run - 1:04d}.png", FIG / "hero_tree_final.png")
    shutil.copy(FR / "f0000.png", FIG / "_hero_first.png")
    print(mp4)


if __name__ == "__main__":
    main()
