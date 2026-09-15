"""Slice atlases for the depth towers (2-D plates, matplotlib): every layer's plane on its own, in the same
normalised, Procrustes-aligned coordinates the towers use. This is the companion that shows the inside of
each tower (spec §0.4). Reads cache/geom_M2.npz and cache/geom_M2.json; writes gallery/atlas_*.png.

  python render_atlas.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import colorcet  # noqa: F401
import common as C

NIGHT = (0.035, 0.035, 0.045)
CMAP = "cet_cyclic_rygcbmr_50_90_c64_s25"


def atlas(name, words, G, meta, out):
    cm = matplotlib.colormaps[CMAP]
    for kind in ("measured", "null"):
        tw = G[f"tower_{name}_{kind}"]
        L = tw.shape[0]
        if name == "numbers":
            val, K = np.asarray(G["a"]), 100
        else:
            val, K = G[f"tower_{name}_labels"], len(words)
        cols = cm((val % K) / K)
        ncol = 6 if L > 17 else 6
        nrow = int(np.ceil(L / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 3.5), facecolor=NIGHT)
        lim = np.quantile(np.abs(tw), 0.995) * 1.1
        for l, ax in enumerate(axs.flat):
            ax.set_facecolor(NIGHT); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color((0.25, 0.25, 0.3))
            if l >= L:
                ax.axis("off"); continue
            s = 2 if name == "numbers" else 9
            ax.scatter(tw[l, :, 0], tw[l, :, 1], c=cols, s=s, lw=0)
            if name != "numbers":
                Mn = G[f"tower_{name}_{kind}_means"][l]
                ax.plot(np.r_[Mn[:, 0], Mn[0, 0]], np.r_[Mn[:, 1], Mn[0, 1]], color=(0.8, 0.8, 0.8), lw=0.6)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            info = meta[f"tower_{name}_{kind}"][l]
            txt = (f"L{l}  R²ho {info['r2_ho']:.2f}{'  cyclic' if info['cyclic'] else ''}" if name != "numbers"
                   else f"L{l}  ΔR² {info['dR2']:.3f}")
            ax.set_title(txt, color=(0.85, 0.85, 0.85), fontsize=9)
        title = {"days": "Qwen3-0.6B day tokens", "months": "Qwen3-0.6B month tokens",
                 "numbers": "OLMo-2-0425-1B numbers 0-999, T=100 plane (template 'The number {a}')"}[name]
        sub = ("mean-difference plane per layer, rotated to the calendar angles, scaled to unit RMS radius"
               if name != "numbers" else "fitted cos/sin frame per layer, scaled to unit RMS radius")
        fig.suptitle(f"{title}: {kind}{' (shuffled labels, identical pipeline)' if kind == 'null' else ''}\n{sub}",
                     color=(0.92, 0.92, 0.92), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / f"atlas_{name}_{kind}.png", dpi=130, facecolor=NIGHT)
        plt.close(fig)


def main():
    G = dict(np.load(C.CACHE / "geom_M2.npz"))
    meta = json.loads((C.CACHE / "geom_M2.json").read_text())
    out = C.ROOT / "gallery"; out.mkdir(exist_ok=True)
    atlas("days", C.DAYS, G, meta, out)
    atlas("months", C.MONTHS, G, meta, out)
    atlas("numbers", None, G, meta, out)


if __name__ == "__main__":
    main()
