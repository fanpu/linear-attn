"""The fragility map of Qwen3-0.6B layers.2.mlp.down_proj (1024 x 3072 = 3,145,728 weights).

Measured:
  - first-order fragility of every weight: |dNLL| predicted for setting it to zero, |w * dNLL/dw|,
    on 8 fineweb-edu documents (4,055 tokens), fp32;
  - exact ablation (zero one weight, re-run the model) for the 50 weights with the largest
    first-order score and for 50 uniformly random weights, same documents.

Declared choices:
  - one disc per weight at its (row, column); disc AREA is proportional to the first-order score,
    scaled so that the largest disc has diameter DMAX px. Anything under ~0.1 px^2 is dust by
    construction, because that is its share of the whole;
  - where the exact ablation was measured, a thin madder ring is drawn with area proportional to the
    exact |dNLL| on the same scale (ring inside the disc = first order over-estimates; ring outside =
    under-estimates). Rings under 3 px radius are omitted;
  - ink on cream; one madder accent for row 35 (the massive-activation dimension) in the margin only.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
CREAM = (242, 237, 227)
INK = (27, 26, 31)
MADDER = (170, 52, 44)
SERIF = "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"


def font(s):
    return ImageFont.truetype(SERIF, s)


def field(tag="qwen3-0.6b", cell=2, DMAX=48, ss=2, name="fragility_field", win=None, ref=None, label="Qwen3-0.6B", note=None):
    z = np.load(os.path.join(CACHE, f"superweight_{tag}.npz"))
    r = json.load(open(os.path.join(CACHE, f"superweight_{tag}.json")))
    pred = np.abs(z["pred"]).astype(np.float64)  # (1024, 3072)
    smax = pred.max() if ref is None else ref  # the value drawn at diameter DMAX
    r0, r1, c0, c1 = win if win else (0, pred.shape[0], 0, pred.shape[1])
    pred = pred[r0:r1, c0:c1]
    R, C = pred.shape
    # stats for the README
    srt = np.sort(pred.ravel())[::-1]
    print("exact max", max(abs(x) for x in r["cand_exact"] + r["rand_exact"]), "first-order max", pred.max())
    print(f"first-order: max {smax:.4g}; top8 {np.round(srt[:8] / smax, 3)}; "
          f"#>10% of max {int((pred > 0.1 * smax).sum())}, #>1% {int((pred > 0.01 * smax).sum())}, "
          f"#>0.1% {int((pred > 0.001 * smax).sum())}; median/max {np.median(pred) / smax:.2e}; "
          f"top-6 share of total {srt[:6].sum() / srt.sum():.3f}")
    m = 160
    W, H = C * cell + 2 * m, R * cell + 2 * m + 170
    # supersampled canvas
    img = Image.new("L", (W * ss, H * ss), 0)  # coverage
    dr = ImageDraw.Draw(img)
    area_max = np.pi * (DMAX / 2) ** 2
    rows, cols = np.nonzero(pred > 0)
    rad = np.sqrt(pred / smax * area_max / np.pi)  # px
    # dust: sub-pixel discs rendered as fractional coverage into the per-cell grid
    cov = np.minimum(np.pi * rad ** 2 / (cell * cell), 1.0)  # coverage of its own cell
    small = rad * 2 < cell
    base = np.zeros((R * cell, C * cell), np.float32)
    base_cells = np.where(small, cov, 0.0).astype(np.float32)
    base[:] = np.kron(base_cells, np.ones((cell, cell), np.float32))
    big = np.argwhere(~small)
    order = np.argsort(pred[~small])  # draw small first
    big = big[order]
    for (i, j) in big:
        cx, cy = (m + (j + 0.5) * cell) * ss, (m + (i + 0.5) * cell) * ss
        rr = rad[i, j] * ss
        dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=255)
    cov_img = np.asarray(img, np.float32) / 255.0
    cov_img = cov_img.reshape(H, ss, W, ss).mean(axis=(1, 3))
    cov_img[m:m + R * cell, m:m + C * cell] = np.maximum(cov_img[m:m + R * cell, m:m + C * cell], base)
    rgb = (np.array(CREAM)[None, None] * (1 - cov_img[..., None]) + np.array(INK)[None, None] * cov_img[..., None]).astype(np.uint8)
    im = Image.fromarray(rgb)
    d2 = ImageDraw.Draw(im)
    # rings: exact ablation, same area scale
    for key_rc, key_ex in (("cand", "cand_exact"), ("rand", "rand_exact")):
        for (i, j), ex in zip(r[key_rc], r[key_ex]):
            i, j = i - r0, j - c0
            if not (0 <= i < R and 0 <= j < C):
                continue
            rr = np.sqrt(abs(ex) / smax * area_max / np.pi)
            if rr < 3:
                continue
            cx, cy = m + (j + 0.5) * cell, m + (i + 0.5) * cell
            d2.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=MADDER, width=3 if rr > 40 else 1)
    # frame + margin marks
    d2.rectangle([m - 1, m - 1, m + C * cell, m + R * cell], outline=INK, width=1)
    row = r["row"]
    y = m + (row - r0 + 0.5) * cell
    d2.line([(m - 40, y), (m - 12, y)], fill=MADDER, width=3)
    d2.text((m - 150, y - 16), f"row {row}", font=font(26), fill=MADDER)
    fs = font(28)
    head = (f"{label}  layers.{r['layer']}.mlp.down_proj  {R:,} × {C:,} = {R * C:,} weights" if not win else
            f"{label} layers.{r['layer']}.mlp.down_proj, rows {r0}-{r1 - 1}, columns {c0}-{c1 - 1}")
    d2.text((m, m + R * cell + 30), head, font=font(34), fill=INK)
    d2.text((m, m + R * cell + 80),
            "disc area ∝ first-order fragility |w · ∂L/∂w|; madder ring = exact ablation, same scale. " if not win else
            "disc: first-order estimate; ring: measured by zeroing the weight. Same area scale. "
            + (note if note is not None else f"Six weights in row {row} write the massive activation."), font=fs, fill=INK)
    out = os.path.join(GAL, f"{name}.png")
    im.save(out, optimize=True)
    print(out, im.size)
    return im


def calibration(tags=(("qwen3-0.6b", "Qwen3-0.6B", INK), ("qwen3-1.7b", "Qwen3-1.7B", MADDER))):
    """first-order vs exact, log-log, ink on cream (a specimen plot, not a hero)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
    fig.patch.set_facecolor(np.array(CREAM) / 255)
    ax.set_facecolor(np.array(CREAM) / 255)
    for tag, name, col in tags:
        r = json.load(open(os.path.join(CACHE, f"superweight_{tag}.json")))
        for key, lab, mk in (("cand", "top by first order", "o"), ("rand", "random", "x")):
            p = np.abs(np.array(r[f"{key}_pred"]))
            e = np.abs(np.array(r[f"{key}_exact"]))
            kw = dict(facecolors="none", edgecolors=np.array(col) / 255) if mk == "o" else dict(color=np.array(col) / 255)
            ax.scatter(np.maximum(p, 1e-9), np.maximum(e, 1e-9), s=16, marker=mk, label=f"{name}: {lab}", linewidths=0.8, **kw)
        i = [tuple(c) for c in r["cand"]].index((r["row"], r["col"]))
        ax.annotate(f"{name} super weight [{r['row']}, {r['col']}]", (abs(r["cand_pred"][i]), abs(r["cand_exact"][i])),
                    xytext=(-215, -2), textcoords="offset points", fontsize=8, color=np.array(col) / 255,
                    arrowprops=dict(arrowstyle="-", color=np.array(col) / 255, lw=0.6, shrinkB=4))
    lo, hi = 1e-9, 10
    ax.plot([lo, hi], [lo, hi], color=(0.5, 0.5, 0.5), lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("first-order |w \u00b7 \u2202L/\u2202w|  (nats)")
    ax.set_ylabel("exact |\u0394L| from zeroing the weight  (nats)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.set_title("what the fragility maps can and cannot see", fontsize=11)
    out = os.path.join(GAL, "fragility_calibration.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    os.makedirs(GAL, exist_ok=True)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "1.7b":
        r17 = json.load(open(os.path.join(CACHE, "superweight_qwen3-1.7b.json")))
        ex = max(abs(x) for x in r17["cand_exact"])
        note = (f"The super weight [{r17['row']}, {r17['col']}]: first order says {max(r17['cand_pred']):.3f} nats, "
                f"zeroing it costs {ex:.2f}.")
        field(tag="qwen3-1.7b", cell=1, DMAX=190, ss=2, name="superweight_field_1.7b", ref=ex, label="Qwen3-1.7B", note=note)
        field(tag="qwen3-1.7b", cell=2, DMAX=380, ss=1, name="superweight_detail_1.7b", ref=ex, label="Qwen3-1.7B",
              note="", win=(1400, 2048, 1500, 2150))
        calibration()
        raise SystemExit
    field()
    field(cell=8, DMAX=48 * 4, ss=1, name="fragility_detail", win=(0, 200, 0, 1600))
    calibration()
