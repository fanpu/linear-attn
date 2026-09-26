"""Follow-up hero: ONE ROAD ON WHAT THEY WERE TAUGHT, THREE ON WHAT THEY WERE NOT.

Two InPCA maps of the same 40 CIFAR-10 runs (followup_fork.py): the training probe (left) and the held-out
probe (right). Same declared orientation/scale as the atlas (Ignorance -> Truth left to right, length 1,
roads bow upward, one horizontal scale for both panels). Apparatus stripped (declared): no neat line, no
ticks, no straight road; only the roads, Ignorance (trig pillar) and Truth (town). Valley names are placed at
the mean end-point of each valley's roads.

    python followup_render_fork.py study       # the documented null (the only image made: the fork did not hold as three)
    python followup_render_fork.py hero        # the hero; NOT rendered, because the three valleys did not hold
    python followup_render_fork.py null        # companion: same maps, the within-class shuffled roads in red
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from atlas import PAPER, INK, RED, FAINT, trig_point, city, save
from analyze import inpca
from metrics import orient

ROOT = os.path.dirname(os.path.abspath(__file__))
VALLEY = {"logreg": 0, "mlp_256": 0, "mlp_2048": 0, "mlp_deep": 0, "cnn": 1, "gru": 1, "resnet": 2, "vit": 2}
VLAB = ["MLPs & logistic regression", "CNN & GRU", "ResNet-8 & ViT"]


def load_map(split, with_perm=False):
    L = np.load(os.path.join(ROOT, "cache", f"followup_fork_{split}.npz"))
    kinds, run, k = L["kinds"], L["run"], L["k"]
    if with_perm:
        idx = np.where(np.isin(kinds, ["P0", "Pstar", "geo", "run", "perm"]))[0]
        X, lam, stress = inpca(L["D"][np.ix_(idx, idx)])
        i0 = int(np.where(kinds[idx] == "P0")[0][0]); i1 = int(np.where(kinds[idx] == "Pstar")[0][0])
        X = orient(X[:, :3], i0, i1)
    else:
        z = np.load(os.path.join(ROOT, "cache", f"followup_fork_map_{split}.npz"))
        idx, X = z["idx"], z["X"]
        i0 = int(np.where(kinds[idx] == "P0")[0][0]); i1 = int(np.where(kinds[idx] == "Pstar")[0][0])
    X = X[:, :2] / np.linalg.norm(X[i1, :2] - X[i0, :2])
    kk, rr = kinds[idx], run[idx]
    if X[kk == "run", 1].mean() < 0:
        X[:, 1] *= -1
    trajs = {}
    for kind in ("run", "perm"):
        for ri in np.unique(rr[kk == kind]):
            sel = np.where((kk == kind) & (rr == ri))[0]
            trajs[(kind, int(ri))] = X[sel[np.argsort(k[idx][sel])]]
    return dict(X=X, i0=i0, i1=i1, trajs=trajs)


def main(which="hero"):
    fk = json.load(open(os.path.join(ROOT, "cache", "followup_fork.json")))
    runs = fk["runs"]; archs = [r["arch"] for r in runs]
    Ms = [load_map(s, with_perm=(which == "null")) for s in ("tr", "te")]
    P = np.vstack([np.vstack(list(M["trajs"].values()) + [M["X"][[M["i0"], M["i1"]]]]) for M in Ms])
    lo, hi = np.percentile(P, 0.2, axis=0), np.percentile(P, 99.8, axis=0)   # a few early outlier excursions may crop
    lo = np.minimum(lo, -0.08); hi[0] = max(hi[0], 1.08)
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    width = 20.0; mx = 0.035; gap = 0.03
    pw = width * (1 - 2 * mx - gap) / 2; ph = pw * h / w
    top, bot = 1.9, 0.9
    H = top + ph + bot
    fig = plt.figure(figsize=(width, H))
    for c, M in enumerate(Ms):
        ax = fig.add_axes([mx + c * (pw / width + gap), bot / H, pw / width, ph / H])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.axis("off")
        if which == "null":
            for ri in range(len(runs)):
                T = M["trajs"][("perm", ri)]
                ax.plot(T[:, 0], T[:, 1], color=RED, lw=0.9, alpha=0.45, solid_joinstyle="round", zorder=4)
        for ri in range(len(runs)):
            T = M["trajs"][("run", ri)]
            ax.plot(T[:, 0], T[:, 1], color=INK, lw=1.1 if which == "hero" else 0.7, alpha=0.42 if which == "hero" else 0.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=5)
        trig_point(ax, M["X"][M["i0"]], s=1.3); city(ax, M["X"][M["i1"]], s=1.3)
        ax.text(*(M["X"][M["i0"]] + [0, -0.05]), "IGNORANCE", ha="center", va="top", fontsize=8.5)
        ax.text(*(M["X"][M["i1"]] + [0, -0.05]), "TRUTH", ha="center", va="top", fontsize=8.5)
        if c == 1 and which == "hero":
            for v in range(3):
                ends = np.array([M["trajs"][("run", ri)][-1] for ri in range(len(runs)) if VALLEY[archs[ri]] == v])
                e = np.median(ends, 0)
                ax.text(e[0] + 0.02, e[1], VLAB[v], fontsize=10, style="italic", va="center", ha="left", zorder=9)
        cap = ["on the 1,000 training examples they were taught", "on 1,000 test examples they never saw"][c]
        ax.text(0.0, 1.03, cap, fontsize=12.5, style="italic", va="bottom", transform=ax.transAxes)
    title = "ONE ROAD ON WHAT THEY WERE TAUGHT, THREE ON WHAT THEY WERE NOT" if which == "hero" else \
        "THE SAME ROADS, EXAMPLES SHUFFLED WITHIN THEIR CLASS (RED)"
    fig.text(mx, 1 - 0.55 / H, title, fontsize=19, va="center")
    tr, te = fk["tr"], fk["te"]
    fig.text(mx, 0.45 / H,
             f"CIFAR-10. {len(runs)} training runs: 8 small architectures × 5 seeds, Adam, 30 epochs on 10,000 images. Each road is "
             f"one run's softmax predictions at 74 checkpoints, placed by intensive PCA of Bhattacharyya distances (Mao et al., PNAS 2024), "
             f"one embedding per panel, top two components ({100 * tr['stress'][1]:.0f}% and {100 * te['stress'][1]:.0f}% of stress).",
             fontsize=8, color=FAINT, va="center")
    return save(fig, f"followup_fork_{which}.png", dpi=300)


def study():
    """The documented null: the held-out map with every architecture named at its road's end, beside the 8x8
    matrix of median trajectory distances (project metric d_traj, held-out probe). Red boxes: the three
    hypothesised valleys. If they were valleys, the boxes would be the pale blocks."""
    from atlas import ARCH_NAME
    fk = json.load(open(os.path.join(ROOT, "cache", "followup_fork.json")))
    runs = fk["runs"]; archs = [r["arch"] for r in runs]
    A = ["mlp_256", "mlp_2048", "mlp_deep", "logreg", "cnn", "gru", "resnet", "vit"]
    M = load_map("te")
    z = np.load(os.path.join(ROOT, "cache", "followup_fork_mats_te.npz")); ar = z["archs"]
    fig = plt.figure(figsize=(20, 9.6))
    ax = fig.add_axes([0.03, 0.1, 0.52, 0.78])
    for ri in range(len(runs)):
        T = M["trajs"][("run", ri)]
        ax.plot(T[:, 0], T[:, 1], color=INK, lw=0.8, alpha=0.45, solid_joinstyle="round", zorder=5)
    trig_point(ax, M["X"][M["i0"]], s=1.2); city(ax, M["X"][M["i1"]], s=1.2)
    ax.text(*(M["X"][M["i0"]] + [0, -0.05]), "IGNORANCE", ha="center", va="top", fontsize=8.5)
    ax.text(*(M["X"][M["i1"]] + [0, -0.05]), "TRUTH", ha="center", va="top", fontsize=8.5)
    placed = []
    for a in A:
        e = np.mean([M["trajs"][("run", i)][-1] for i in range(len(runs)) if archs[i] == a], 0)
        y = e[1]
        while any(abs(e[0] - px) < 0.2 and abs(y - py) < 0.035 for px, py in placed):
            y -= 0.04
        placed.append((e[0], y))
        ax.plot([e[0], e[0] + 0.02], [e[1], y], color=FAINT, lw=0.5)
        ax.text(e[0] + 0.025, y, ARCH_NAME[a], fontsize=9.5, style="italic", va="center")
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 1.02, "the held-out map: 8 architectures x 5 seeds (top two components, 72% of stress)", fontsize=11,
            style="italic", transform=ax.transAxes)
    # matrix
    R = z["R"]
    Mm = np.array([[np.nanmedian(R[np.ix_(ar == a, ar == b)][~np.eye(5, dtype=bool)]) if a == b else np.nanmedian(R[np.ix_(ar == a, ar == b)])
                    for b in A] for a in A])
    ax2 = fig.add_axes([0.6, 0.14, 0.36, 0.7])
    vmax = Mm.max()
    ax2.imshow(1 - Mm / vmax, cmap="gray", vmin=0, vmax=1, alpha=0.9)   # dark = far, declared: ink for distance
    for i in range(8):
        for j in range(8):
            ax2.text(j, i, f"{Mm[i, j]:.2f}", ha="center", va="center", fontsize=9, color=PAPER if Mm[i, j] / vmax > 0.55 else INK)
    for lo_, hi_ in ((0, 3), (4, 5), (6, 7)):   # hypothesised valleys: MLPs+logreg, CNN+GRU, ResNet+ViT
        ax2.add_patch(plt.Rectangle((lo_ - 0.5, lo_ - 0.5), hi_ - lo_ + 1, hi_ - lo_ + 1, fill=False, ec=RED, lw=2.2))
    ax2.set_xticks(range(8)); ax2.set_yticks(range(8))
    ax2.set_xticklabels([ARCH_NAME[a] for a in A], rotation=40, ha="right", fontsize=9, style="italic")
    ax2.set_yticklabels([ARCH_NAME[a] for a in A], fontsize=9, style="italic")
    for sp_ in ax2.spines.values(): sp_.set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title("median distance between roads at matched progress (d_traj, nats); diagonal = seed to seed.\n"
                  "Red: the three valleys proposed from two seeds.", fontsize=10, style="italic", loc="left")
    te = fk["te"]["dtraj"]
    fig.text(0.03, 0.955, "DO THE HELD-OUT ROADS FORK INTO THREE?  NOT AS THREE.", fontsize=19, va="center")
    fig.text(0.03, 0.035,
             f"CIFAR-10 held-out probe, 40 runs. Between proposed valleys d = {te['cats']['between']['median']:.3f}; within a valley, "
             f"across architectures {te['cats']['valley']['median']:.3f}; seed to seed {te['cats']['seed']['median']:.3f}. "
             f"Silhouette of the three-valley grouping {te['silhouette_valleys']:.2f}, rank {te['silhouette_rank']} of {te['n_groupings']} "
             "groupings into 4+2+2.\nWhat survives: the three MLPs split from the rest (0.33 vs 0.18); CNN and GRU pair up; ResNet-8 and ViT do not; "
             "MLP 1x2048's own seeds are as far apart as anything.",
             fontsize=8.5, color=FAINT, va="center")
    return save(fig, "followup_fork_study.png", dpi=250)


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["hero"]):
        study() if w == "study" else main(w)
