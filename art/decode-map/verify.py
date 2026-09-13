"""§11 verification for decode maps: placement reproducibility, resolution check, box counting
with null models, and the zoom-refinement measurement. Writes cache/verify.json and
gallery/verify_*.png plots."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as A

OUT = {}
C = "cache/"


def region_split(d):
    """coherent / degenerate by mean model entropy along the text (threshold 3 nats)."""
    return A.mean_entropy(d) > 3.0


def placement():
    pairs = [("toy_story_tp64.npz", "toy_story_tp64_trie.npz", "brute force B=256 vs trie Fb=128"),
             ("toy_story_tp64_trie.npz", "toy_story_tp64_fast.npz", "old engine (eager, V-wide sampler) vs CUDA graph + certified sampler"),
             ("place_story64_eager.npz", "place_story64_slow_eager.npz", "certified sampler vs V-wide sampler, same eager forward"),
             ("place_story64_eager.npz", "toy_story_tp64_fast.npz", "eager forward vs CUDA-graph forward (masked attention kernel)"),
             ("toy_story_tp64_fast.npz", "place_story64_shuffle.npz", "trie vs trie, shuffled pixel order, Ncap=32, Fb=16"),
             ("story_tp128.npz", "place_story128_shuffle.npz", "trie vs trie, shuffled, Ncap=96, Fb=32 (128², L=48)"),
             ("toy_story_tp64_fast.npz", "place_story64_fp32.npz", "bf16 body vs fp32 body (a different model precision)")]
    res = []
    for a, b, desc in pairs:
        if not (os.path.exists(C + a) and os.path.exists(C + b)):
            continue
        da, db = A.load(C + a), A.load(C + b)
        ta, tb = da["tokens"], db["tokens"]
        diff = (ta != tb).any(-1)
        deg = region_split(da)
        fda = np.where((ta != tb).any(-1), (ta != tb).argmax(-1), ta.shape[2])
        r = dict(desc=desc, a=a, b=b, pixels=int(diff.size), mismatch_rate=float(diff.mean()),
                 mismatch_coherent=float(diff[~deg].mean()) if (~deg).any() else None,
                 mismatch_degenerate=float(diff[deg].mean()) if deg.any() else None,
                 frac_degenerate=float(deg.mean()),
                 median_first_diff_token=float(np.median(fda[diff])) if diff.any() else None)
        res.append(r)
        print(r)
    OUT["placement"] = res


def boxcount_global():
    """Box counting of the cell-boundary set in parameter units, at two resolutions, several L."""
    eps_px = np.array([1, 2, 4, 8, 16, 32, 64])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    res = {}
    for f, ls in [("story_tp128.npz", "--"), ("story_tp256.npz", "-")]:
        if not os.path.exists(C + f):
            continue
        d = A.load(C + f)
        tk = d["tokens"]
        R, L = tk.shape[0], tk.shape[2]
        hs = A.prefix_hashes(tk)
        for l, col in zip([4, 8, 16, 32, 64], plt.cm.viridis(np.linspace(0, 0.9, 5))):
            if l > L:
                continue
            b = A.boundary(A.labels_from_hash(hs[l - 1]))
            e = eps_px[eps_px <= R // 4]
            N = A.box_count(b, e)
            eps_u = e / R                                    # fraction of the full window
            ax[0].loglog(eps_u, N, ls, color=col, marker="o", ms=3, label=f"{R}², L={l}")
            loc = -np.diff(np.log(N)) / np.diff(np.log(e))
            res[f"{R}_L{l}"] = dict(eps=eps_u.tolist(), N=N.tolist(), local_slopes=loc.tolist(),
                                    boundary_frac=float(b.mean()))
            ax[1].semilogx(np.sqrt(eps_u[1:] * eps_u[:-1]), loc, ls, color=col, marker="o", ms=3)
    # null models on the 256 grid
    R = 256
    yy, xx = (np.mgrid[0:R, 0:R] + 0.5) / R
    smooth = (yy > 0.8 / (1 + 3 * xx ** 2)).astype(int)
    rng = np.random.default_rng(0)
    speck = rng.integers(0, 50, (R, R))
    for lab, name, col in [(smooth, "null: smooth curve", "k"), (speck, "null: iid speckle", "grey")]:
        b = A.boundary(lab)
        e = eps_px[eps_px <= R // 4]
        N = A.box_count(b, e)
        loc = -np.diff(np.log(N)) / np.diff(np.log(e))
        ax[0].loglog(e / R, N, ":", color=col, label=name)
        ax[1].semilogx(np.sqrt(e[1:] * e[:-1]) / R, loc, ":", color=col)
        res[name] = dict(eps=(e / R).tolist(), N=N.tolist(), local_slopes=loc.tolist())
    ax[0].set_xlabel("box side ε (fraction of the map width)"); ax[0].set_ylabel("occupied boxes N(ε)")
    ax[1].set_xlabel("ε"); ax[1].set_ylabel("local slope  −dlogN/dlogε")
    ax[1].axhline(1, color="k", lw=0.5); ax[1].axhline(2, color="k", lw=0.5)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Box counting of the cell-boundary set, story prompt, (T, top-p) map")
    fig.tight_layout()
    fig.savefig("gallery/verify_boxcount.png", dpi=150)
    OUT["boxcount_global"] = res


def cells_vs_resolution():
    res = {}
    for f in ["story_tp128.npz", "story_tp256.npz", "list_tp256.npz", "fact_tp256.npz", "story_tr192.npz",
              "story_tp128_gumbel.npz"]:
        if os.path.exists(C + f):
            d = A.load(C + f)
            nc = A.n_cells_per_length(d["tokens"])
            res[f] = dict(R=int(d["tokens"].shape[0]), n_cells=nc.tolist(),
                          singleton_frac=float(np.mean(np.bincount(A.cell_labels(d["tokens"]).ravel()) == 1)))
            print(f, nc[[0, 3, 7, 15, 31, 63]] if len(nc) >= 64 else nc[-1], res[f]["singleton_frac"])
    OUT["cells"] = res


def zoom(name):
    files = sorted(glob.glob(f"{C}zoom_{name}_*.npz"))
    if not files:
        return
    rows = []
    for f in files:
        d = A.load(f)
        tk = d["tokens"]
        R = tk.shape[0]
        hs = A.prefix_hashes(tk)
        k = int(f[-6:-4])
        row = dict(level=k, width_T=float(d["xs"][-1] - d["xs"][0] + d["xs"][1] - d["xs"][0]))
        for l in (8, 16, 32, 64):
            lab = A.labels_from_hash(hs[l - 1])
            b = A.boundary(lab)
            e = np.array([2, 4, 8, 16, 32])
            N = A.box_count(b, e)
            slope, r2 = A.fit_slope(e.astype(float), N, 2, 32) if (N > 0).sum() > 2 else (np.nan, np.nan)
            row[f"cells_L{l}"] = int(lab.max() + 1)
            row[f"bfrac_L{l}"] = float(b.mean())
            row[f"D_L{l}"] = float(slope)
        rows.append(row)
    OUT[f"zoom_{name}"] = rows
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    w = [r["width_T"] for r in rows]
    for l, col in zip((8, 16, 32, 64), plt.cm.viridis(np.linspace(0, 0.9, 4))):
        ax[0].loglog(w, [max(r[f"cells_L{l}"], 0.8) for r in rows], "o-", color=col, label=f"L={l}")
        ax[1].semilogx(w, [r[f"D_L{l}"] for r in rows], "o-", color=col, label=f"L={l}")
    ax[0].invert_xaxis(); ax[1].invert_xaxis()
    ax[0].set_xlabel("window width in T (zooming in →)"); ax[0].set_ylabel("distinct outputs in the 128² window")
    ax[1].set_xlabel("window width in T (zooming in →)"); ax[1].set_ylabel("box-count slope of boundaries (ε = 2–32 px)")
    ax[1].axhline(1, color="k", lw=0.5)
    ax[0].legend(); fig.suptitle(f"Zoom sequence {name}: where refinement stops")
    fig.tight_layout(); fig.savefig(f"gallery/verify_zoom_{name}.png", dpi=150)
    for r in rows:
        print(r)


if __name__ == "__main__":
    what = sys.argv[1:] or ["placement", "cells", "box", "zoom"]
    if os.path.exists(C + "verify.json"):
        OUT.update(json.load(open(C + "verify.json")))
    if "placement" in what:
        placement()
    if "cells" in what:
        cells_vs_resolution()
    if "box" in what:
        boxcount_global()
    if "zoom" in what:
        for n in ("zA", "zB"):
            zoom(n)
    json.dump(OUT, open(C + "verify.json", "w"), indent=1)
