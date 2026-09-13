"""Is the basin-tree structure self-similar?  Two scale-free diagnostics, real graph vs null model.

1. Subtree-size distribution P(S >= s) over all non-cyclic nodes.  Critical random trees (the random-
   mapping null) give P(S >= s) ~ s^(-1/2).
2. Horton-Strahler branching: number of streams N_k of Strahler order k; a constant bifurcation ratio
   R_b = N_k / N_(k+1) across orders is topological self-similarity (critical binary Galton-Watson: R_b = 4).

Null model: succ[perm] (same in-degree sequence, random wiring), 20 draws.

    python analyze_trees.py      # -> cache/tree_stats.json, gallery/tree_selfsimilarity.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from graph import analyze  # noqa: E402

HERE = Path(__file__).parent
CACHE, GALLERY = HERE / "cache", HERE / "gallery"


def subtree_and_strahler(succ, depth):
    n = len(succ)
    size = np.ones(n, np.int64)
    order_ = np.ones(n, np.int64)
    maxch = np.zeros(n, np.int64)
    cntmax = np.zeros(n, np.int64)
    for c in np.argsort(-depth, kind="stable"):
        if depth[c] == 0:
            continue
        # finalize c's Strahler order from its children stats
        if maxch[c] > 0:
            order_[c] = maxch[c] + (1 if cntmax[c] >= 2 else 0)
        p = succ[c]
        size[p] += size[c]
        if order_[c] > maxch[p]:
            maxch[p], cntmax[p] = order_[c], 1
        elif order_[c] == maxch[p]:
            cntmax[p] += 1
    for c in np.flatnonzero(depth == 0):
        if maxch[c] > 0:
            order_[c] = maxch[c] + (1 if cntmax[c] >= 2 else 0)
    noncyc = depth > 0
    # stream counts: a stream of order k starts at a node of order k whose parent has higher order (or is cyclic)
    par_order = order_[succ]
    starts = noncyc & ((par_order > order_) | (depth[succ] == 0))
    Nk = np.bincount(order_[starts])
    return size[noncyc], Nk


def ccdf(x):
    xs = np.sort(x)
    s = np.unique(xs)
    p = 1 - np.searchsorted(xs, s, side="left") / len(xs)
    return s, p


def slope(s, p, lo=3, hi=None):
    hi = hi or s.max() / 10
    m = (s >= lo) & (s <= hi) & (p > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(s[m]), np.log(p[m]), 1)[0])


def main():
    G = np.load(CACHE / "graphs.npz")
    rng = np.random.default_rng(5)
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=200, facecolor="#f3eee2")
    for col, fmt in enumerate(["float16", "bfloat16"]):
        succ, depth = G[f"{fmt}_succ"], G[f"{fmt}_depth"]
        sz, Nk = subtree_and_strahler(succ, depth)
        s, p = ccdf(sz)
        res = dict(subtree_ccdf_slope=slope(s, p), strahler_counts=Nk.tolist(),
                   bifurcation_ratios=(Nk[1:-1] / np.maximum(Nk[2:], 1)).round(2).tolist())
        ax, ax2 = axes[0, col], axes[1, col]
        ax.loglog(s, p, color="#b8322a", lw=2, label=f"{fmt} (slope {res['subtree_ccdf_slope']:.2f})")
        null_sl, null_R = [], []
        for r in range(20):
            sn = succ[rng.permutation(len(succ))]
            a = analyze(sn, want_node_arrays=True)
            szn, Nkn = subtree_and_strahler(sn, a["depth"])
            sN, pN = ccdf(szn)
            null_sl.append(slope(sN, pN))
            null_R.append((Nkn[1:-1] / np.maximum(Nkn[2:], 1)).tolist())
            ax.loglog(sN, pN, color="#2f5d8a", lw=0.6, alpha=0.35, label="in-degree-preserving null" if r == 0 else None)
        ss = np.logspace(0, np.log10(s.max()), 20)
        ax.loglog(ss, ss ** -0.5, "k--", lw=1, label="s^-1/2 (critical random tree)")
        ax.set_title(f"{fmt}: subtree sizes", family="P052", fontsize=18, loc="left")
        ax.set_xlabel("subtree size s", family="Nimbus Mono PS")
        ax.set_ylabel("P(S >= s)", family="Nimbus Mono PS")
        ax.legend(frameon=False, prop=dict(family="Nimbus Mono PS"))
        res["null_subtree_slope_mean"] = float(np.nanmean(null_sl))
        res["null_subtree_slope_std"] = float(np.nanstd(null_sl))
        res["null_bifurcation_ratios_example"] = null_R[:3]
        k = np.arange(1, len(Nk))
        ax2.semilogy(k, Nk[1:], "o-", color="#b8322a", lw=2, label=fmt)
        for rr in null_R[:20]:
            ax2.semilogy(np.arange(1, len(rr) + 1), np.maximum(1, np.cumprod([1] + rr)[:len(rr)]) * 0 + np.nan)
        for r in range(5):
            sn = succ[rng.permutation(len(succ))]
            a = analyze(sn, want_node_arrays=True)
            _, Nkn = subtree_and_strahler(sn, a["depth"])
            ax2.semilogy(np.arange(1, len(Nkn)), Nkn[1:], "o-", color="#2f5d8a", lw=0.8, alpha=0.5,
                         label="null" if r == 0 else None)
        ax2.set_title(f"{fmt}: Horton-Strahler stream counts N_k", family="P052", fontsize=18, loc="left")
        ax2.set_xlabel("Strahler order k", family="Nimbus Mono PS")
        ax2.set_ylabel("N_k", family="Nimbus Mono PS")
        ax2.legend(frameon=False, prop=dict(family="Nimbus Mono PS"))
        for a_ in (ax, ax2):
            a_.set_facecolor("#f3eee2")
        out[fmt] = res
        print(fmt, res)
    fig.suptitle("Are the basin trees self-similar?  exact graph (red) vs in-degree-preserving random rewiring (blue)",
                 family="P052", fontsize=20, x=0.02, ha="left")
    fig.savefig(GALLERY / "tree_selfsimilarity.png", facecolor="#f3eee2")
    (CACHE / "tree_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
