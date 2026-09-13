"""NC1-NC4 plate: small multiples on one log-epoch axis, train (solid) vs test (dashed).

  python render_curves.py c10 [c4] --style paper|night
"""
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import nclib as N
import artlib as A

ST = {
    "paper": dict(bg=A.PAPER, ink=A.INK, muted="#8a857a", tr="#1b1b1f", te="#c0392b", grid="#dcd5c4"),
    "night": dict(bg="#0b0c10", ink="#e8e4d8", muted="#7d7a70", tr="#e8e4d8", te="#f0a35e", grid="#23252c"),
}


def series(mets, key):
    x = np.array([m["epoch"] for m in mets if key in m], float)
    y = np.array([m[key] for m in mets if key in m], float)
    return x, y


def xmap(x):
    return np.where(x <= 0, 0.004, x)  # initialisation drawn at 0.004 epochs (declared)


def plate(tags, style, fname):
    s = ST[style]
    rows = [
        ("NC1  within-class variability  tr(Σ_W Σ_B⁺)/C", [("nc1", "train"), ("nc1_test", "test")], "log"),
        ("NC2  std of pairwise cosines of centred means (ideal 0)", [("M_cos_std", "train"), ("Mtest_cos_std", "test")], "log"),
        ("NC2  norm spread  std/mean of ‖μ_c − μ_G‖ (ideal 0)", [("M_equinorm", "train"), ("Mtest_equinorm", "test")], "log"),
        ("NC3  ‖W/‖W‖ − M/‖M‖‖  classifier vs means", [("nc3", "train")], "log"),
        ("NC4  classifier ≠ nearest class mean (fraction)", [("nc4_train", "train"), ("nc4_test", "test")], "log"),
        ("error rate (0 drawn at 1e-5)", [("acc_train", "train"), ("acc_test", "test")], "log"),
    ]
    nc = len(tags)
    fig, axs = plt.subplots(len(rows), nc, figsize=(7.2 * nc, 2.1 * len(rows)), dpi=220, facecolor=s["bg"],
                            sharex="col", squeeze=False)
    plt.rcParams.update({"font.family": "serif"})
    for c, tag in enumerate(tags):
        meta, mets, ck = N.load_run(tag)
        C = len(meta["classes"]); E = meta["epochs"]
        for r, (title, keys, yscale) in enumerate(rows):
            ax = axs[r, c]
            ax.set_facecolor(s["bg"])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(colors=s["muted"], labelsize=7, length=2)
            ax.grid(True, color=s["grid"], lw=0.4, which="major")
            for dl in (E // 3, 2 * E // 3):
                ax.axvline(dl, color=s["muted"], lw=0.5, ls=(0, (2, 3)))
            for key, lab in keys:
                x, y = series(mets, key)
                if key.startswith("acc"):
                    y = np.maximum(1 - y, 1e-5)
                col, ls = (s["tr"], "-") if lab == "train" else (s["te"], (0, (4, 2)))
                ax.plot(xmap(x), y, color=col, lw=1.2, ls=ls)
                ax.text(xmap(x)[-1] * 1.08, y[-1], f"{lab} {y[-1]:.2g}", color=s["ink"], fontsize=6.5, va="center")
            ax.set_xscale("log"); ax.set_yscale(yscale)
            ax.yaxis.set_minor_formatter(NullFormatter())
            nk = {"M_cos_std": "null256_cos_std", "M_equinorm": "null256_equinorm"}.get(keys[0][0])
            mf = os.path.join(N.HERE, "cache", f"misfit_{tag}.npz")
            if nk and os.path.exists(mf):
                lo, md, hi = np.load(mf)[nk]
                ax.axhspan(lo, hi, color=s["grid"], alpha=0.9, lw=0, zorder=0)
                ax.text(0.0035, hi * 1.05, "random Gaussian means, d=256 (5–95 %)", color=s["muted"], fontsize=6, va="bottom")
            ax.set_xlim(0.003, E * 3.2)
            ax.set_title(title, loc="left", color=s["ink"], fontsize=8.5)
            if r == 0:
                ax.text(0, 1.42, f"C = {C}   ({', '.join(N.CIFAR[k] for k in meta['classes'])})" if C < 10 else
                        "C = 10   (CIFAR-10)", transform=ax.transAxes, color=s["ink"], fontsize=11)
        axs[-1, c].set_xlabel("epoch (log; init at left edge; dotted = lr ÷10)", color=s["muted"], fontsize=7.5)
    fig.tight_layout(h_pad=1.2)
    return A.save(fig, fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+"); ap.add_argument("--style", default="paper")
    a = ap.parse_args()
    print(plate(a.tags, a.style, f"curves_{'_'.join(a.tags)}_{a.style}.png"))
