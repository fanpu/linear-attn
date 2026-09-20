"""Sheets for the twitter_pipeline experiment: the damage, and the law that predicts it.

    python render_pipeline_sheet.py      # reads twitter_pipeline.json, writes two PNGs
"""
import json
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P
import sources as S
import render_sheets as R
import twitter_pipeline as T

GAL = T.GAL
SHOW = ["sd_spectral", "aurora_ember", "klimt_lapis", "riso_pink_blue", "riso_halftone"]


def damage_sheet(n=2048, crop_px=440):
    """Native-resolution crops of the 1200 px serve: what you meant, what X serves, the difference.

    The damage is a high-frequency chroma effect, so it must be shown at 1:1 pixels. Downscaling the
    comparison for display destroys the very thing being measured.
    """
    d = S.basin_signed(tag="z1T_4096", stride=2)
    x = d["x"]
    c, h = x.shape[0] // 2, int(x.shape[0] * 0.31)
    x = x[c - h:c + h, c - h:c + h]
    x = np.asarray(Image.fromarray(x.astype(np.float32), "F").resize((n, n), Image.BILINEAR))
    imgs = T.render_all(x, np.abs(x))
    rows = {r["name"]: r for r in json.load(open(GAL + "../twitter_pipeline.json"))}

    def cut(a):
        """Declared crop: the busiest region of the 1200 px serve (upper-left fractal band)."""
        y0, x0 = int(a.shape[0] * 0.10), int(a.shape[1] * 0.06)
        return a[y0:y0 + crop_px, x0:x0 + crop_px]

    tiles = []
    for name in SHOW:
        if name not in imgs:
            continue
        src = T.to_u8(imgs[name][0])
        want, got = cut(T.ideal(src, "medium")), cut(T.pipeline(src, "medium"))
        m = rows.get(name, {})
        bar = P.split_cmap(P.PAIRINGS[name]["neg"], P.PAIRINGS[name]["pos"]) if name in P.PAIRINGS else None
        sub = f"chroma-carried contrast {m.get('chroma_fraction', float('nan')):.2f}"
        tiles.append(dict(img=Image.fromarray(want), label=f"{name} — as rendered", sub=sub, bar=bar))
        tiles.append(dict(img=Image.fromarray(got),
                          label=f"{name} — after X  (mean dE00 {m.get('dE_medium', float('nan')):.1f})",
                          sub=sub, bar=bar))
        tiles.append(dict(img=Image.fromarray(T.to_u8(T.diff_map(got, want))),
                          label=f"{name} — where it broke", sub=sub, bar=bar))

    R.sheet(tiles, "pipeline_damage.png",
            "What a social-media pipeline does to a colour scheme",
            "Real data: the same GD basin crop as split_basins.png, rendered at 2048^2, resized 2048 -> 1200 with a "
            f"JPEG q75 4:2:0 re-encode at each serve, against the identical resize done losslessly. {crop_px} px crops, 1:1 pixels.",
            ncols=3, footer="Measured: CIEDE2000 per pixel. Third column is that dE00 on a black-to-ember ramp, "
                            "full scale = dE00 10. Declared: the crop, the ramp, and which five schemes are shown.")


def law_plot():
    """The finding: codec damage is predicted by how much contrast the scheme puts in chroma."""
    rows = json.load(open(GAL + "../twitter_pipeline.json"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor="#eeeae3")
    fam_c = {"split": "#c1436d", "sequential": "#2f6aa3", "riso": "#c8801f"}

    ax = axes[0]
    for r in rows:
        ax.scatter(r["chroma_fraction"], r["dE_medium"], s=90, zorder=3,
                   c=fam_c[r["family"]], edgecolor="#1c1c20", linewidth=0.7)
        ax.annotate(r["name"], (r["chroma_fraction"], r["dE_medium"]), fontsize=7.5,
                    xytext=(6, -3), textcoords="offset points", color="#3a3a40")
    xs = np.array([r["chroma_fraction"] for r in rows])
    ys = np.array([r["dE_medium"] for r in rows])
    b, a = np.polyfit(xs, ys, 1)
    xg = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xg, a + b * xg, "--", color="#6a655f", lw=1.2, zorder=2)
    rho = np.corrcoef(xs, ys)[0, 1]
    ax.set_title(f"Codec damage is predicted by chroma-carried contrast   (r = {rho:.2f})",
                 fontsize=11.5, weight="bold", loc="left")
    ax.set_xlabel("share of local contrast carried in chroma rather than lightness")
    ax.set_ylabel("mean CIEDE2000 vs lossless resize (1200 px serve)")

    ax = axes[1]
    order = sorted(rows, key=lambda r: -r["detail_retained"])
    ax.barh([r["name"] for r in order], [r["detail_retained"] for r in order],
            color=[fam_c[r["family"]] for r in order], edgecolor="#1c1c20", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_title("Spatial detail surviving the 440 px timeline serve", fontsize=11.5, weight="bold", loc="left")
    ax.set_xlabel("L* gradient energy retained vs the 2048 px render")
    ax.tick_params(labelsize=8)

    for a_ in axes:
        a_.set_facecolor("#eeeae3")
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(GAL + "pipeline_law.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote pipeline_law.png   r =", round(rho, 3), " slope =", round(b, 2))


if __name__ == "__main__":
    damage_sheet()
    law_plot()
