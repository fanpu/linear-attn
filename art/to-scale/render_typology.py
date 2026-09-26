"""To Scale, nine models: the loudest token of one page, every model at the same scale.

For each model: the residual stream after its peak layer (the layer where max|h| / median|h| is
largest over 16 fineweb-edu pages), document 0, first 256 tokens; we take the token whose largest
|h_d| is biggest and draw all of its dims as bars.

Measured: bar height = |h_d| / (that layer's median |h| over all tokens and dims), in mm at 1 mm = median.
Every model uses the same px-per-mm, so heights are comparable across the sheet.

Declared:
  - widths are NOT to scale: each model gets the same column width and each pixel column shows the
    tallest of the dims it covers (max-pooling), so a single massive dim stays visible;
  - models grouped by family; ink for softmax attention, madder for models with no softmax attention,
    grey for the untrained control; cream ground.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
CREAM = np.array([242, 237, 227], np.float32)
INK = np.array([27, 26, 31], np.float32)
MADDER = np.array([170, 52, 44], np.float32)
GREY = np.array([140, 136, 130], np.float32)
SERIF = "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"

MODELS = [("qwen3-0.6b", "Qwen3\n0.6B", INK), ("qwen3-1.7b", "Qwen3\n1.7B", INK), ("qwen3-4b", "Qwen3\n4B", INK),
          ("olmo2-1b", "OLMo-2\n1B", INK),
          ("gdn-1.3b", "Gated DeltaNet\n1.3B", MADDER), ("gla-1.3b", "GLA\n1.3B", MADDER),
          ("deltanet-1.3b", "DeltaNet\n1.3B", MADDER), ("rwkv7-1.5b", "RWKV-7\n1.5B", MADDER),
          ("qwen3-0.6b-randinit", "Qwen3-0.6B\nuntrained", GREY)]


def font(s):
    return ImageFont.truetype(SERIF, s)


def pick(name):
    r = json.load(open(os.path.join(CACHE, f"acts_{name}.json")))
    z = np.load(os.path.join(CACHE, f"acts_{name}.npz"))
    L = r["layers"]
    ratios = np.array([l["top1"] / l["median"] for l in L])
    li = int(ratios.argmax())
    med = L[li]["median"]
    H = np.abs(z["doc0"][li])  # (T, D)
    t = int(H.max(1).argmax())
    v = H[t] / med
    return {"v": v, "layer": li, "tok": str(z["tokens0"][t]), "pos": t, "ratio_all": float(ratios[li]),
            "ratio_here": float(v.max()), "D": len(v)}


def main(colw=150, gut=115, px_per_m=58.0):
    data = [(n, lab, col, pick(n)) for n, lab, col in MODELS]
    tallest = max(d["v"].max() for *_, d in data) / 1000.0  # m
    Hb = int(np.ceil(tallest * px_per_m)) + 40
    mL, mT, foot = 170, 330, 330
    W = mL * 2 + len(data) * colw + (len(data) - 1) * gut
    Ht = mT + Hb + foot
    arr = np.empty((Ht, W, 3), np.float32)
    arr[:] = CREAM
    floor = mT + Hb
    for k, (n, lab, col, d) in enumerate(data):
        x0 = mL + k * (colw + gut)
        v_px = d["v"] / 1000.0 * px_per_m  # heights in px
        # max-pool dims into colw pixel columns
        edges = np.linspace(0, d["D"], colw + 1).astype(int)
        for c in range(colw):
            h = v_px[edges[c]:max(edges[c + 1], edges[c] + 1)].max()
            full = int(np.floor(h))
            frac = h - full
            if full > 0:
                arr[floor - full:floor, x0 + c] = col
            if frac > 0:
                y = floor - full - 1
                arr[y, x0 + c] = CREAM * (1 - frac) + col * frac
    im = Image.fromarray(arr.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    ink = tuple(INK.astype(int))
    dr.line([(mL - 30, floor), (W - mL + 30, floor)], fill=ink, width=1)
    # metre ticks on the left
    for m in range(0, int(tallest) + 1, 5):
        y = floor - m * px_per_m
        dr.line([(mL - 60, y), (mL - 40, y)], fill=ink, width=2)
        dr.text((mL - 140, y - 14), f"{m} m", font=font(24), fill=ink)
    for k, (n, lab, col, d) in enumerate(data):
        x0 = mL + k * (colw + gut)
        c = tuple(col.astype(int))
        y = floor + 24
        dr.text((x0, y), lab, font=font(24), fill=c)
        tk = d["tok"].replace("Ġ", "␣").replace("▁", "␣").replace("Ċ", "\\n")
        lines = [f"{d['ratio_here'] / 1000:.2f} m" if d["ratio_here"] >= 1000 else f"{d['ratio_here']:.0f} mm",
                 f"token [{tk}]", f"position {d['pos']}", f"after layer {d['layer']}"]
        for j, s in enumerate(lines):
            dr.text((x0, y + 70 + j * 30), s, font=font(22), fill=c)
    dr.text((mL, 60), "To Scale: nine models, one page of text", font=font(64), fill=ink)
    dr.text((mL, 160), "Each column is the loudest token on the page: its residual stream at 1 mm = the median |activation| of that layer.", font=font(30), fill=ink)
    dr.text((mL, 205), "Same scale for every model. Ink: softmax attention. Madder: no softmax attention anywhere. Grey: untrained.", font=font(30), fill=ink)
    dr.text((mL, 250), "Column widths are not to scale (each pixel column shows the tallest of the dimensions it covers).", font=font(26), fill=ink)
    out = os.path.join(GAL, "typology.png")
    im.save(out, optimize=True)
    print(out, im.size)
    for n, lab, col, d in data:
        print(f"  {lab.replace(chr(10), ' '):22s} layer {d['layer']:2d} token {d['tok']!r} pos {d['pos']:3d}: {d['ratio_here']:9.0f} x median "
              f"(16-page max {d['ratio_all']:.0f})")


if __name__ == "__main__":
    main()
