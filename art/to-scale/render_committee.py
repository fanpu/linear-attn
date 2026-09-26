"""Committee: all 64 ways of removing the six weights that write Qwen3-0.6B's massive activation.

Measured (superweight.py --quick, held-out fineweb-edu, 7,993 tokens, fp32):
  for every subset S of the six weights W[35, c] (c = 55, 128, 1489, 321, 46, 646) of
  layers.2.mlp.down_proj, the held-out NLL with the weights in S set to zero.

Layout:
  - x = how much of the massive activation S removes: sum over c in S of W[35,c] * x_c at position 0
    of the calibration document (measured, same units as the activation; the full value is 6,541);
  - y = |S|, the number of weights removed (structure, declared);
  - hairlines join subsets that differ by one weight (the 6-cube; declared);
  - disc AREA proportional to the rise in held-out NLL (nats) over the intact model (measured).
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
HAIR = (150, 145, 140)
MADDER = (170, 52, 44)
SERIF = "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"


def font(s):
    return ImageFont.truetype(SERIF, s)


def main(tag="q-qwen3-0.6b", W=3000, H=2400, ss=3):
    r = json.load(open(os.path.join(CACHE, f"superweight_{tag}.json")))
    cols = r["subset_cols"]
    con = np.array([r["contribs"][c] for c in cols])
    total = float(r["out_at_pos"])
    base = r["base_held_nll"]
    S = r["subsets"]
    K = len(cols)
    nodes = {}
    for s, v in S.items():
        idx = [int(ch) for ch in s]
        nodes[s] = (con[idx].sum() if idx else 0.0, len(idx), v - base)
    dmax = max(v[2] for v in nodes.values())
    # print the cliff for the README
    xs = sorted(nodes.values(), key=lambda t: t[0])
    print("removed activation vs dNLL:")
    for x, k, d in xs[::4]:
        print(f"   removed {x:7.0f} of {total:.0f} ({x / total:.2f})  k={k}  dNLL {d:.3f}  ppl x{np.exp(d):.2f}")
    mL, mR, mT, mB = 260, 200, 330, 330
    X = lambda x: mL + x / total * (W - mL - mR)
    Y = lambda k: mT + k / K * (H - mT - mB)
    RMAX = 125.0
    area_max = np.pi * RMAX ** 2
    img = Image.new("RGB", (W * ss, H * ss), CREAM)
    dr = ImageDraw.Draw(img)
    # edges
    for s in S:
        for j in range(K):
            if str(j) in s:
                continue
            t = "".join(sorted(s + str(j)))
            x0, k0, _ = nodes[s]
            x1, k1, _ = nodes[t]
            dr.line([(X(x0) * ss, Y(k0) * ss), (X(x1) * ss, Y(k1) * ss)], fill=HAIR, width=max(1, ss))
    # discs, smallest last so none hides
    for s, (x, k, d) in sorted(nodes.items(), key=lambda kv: -kv[1][2]):
        rr = np.sqrt(max(d, 0) / dmax * area_max / np.pi)
        cx, cy = X(x) * ss, Y(k) * ss
        if rr * ss < 2.5 * ss:
            dr.ellipse([cx - 5 * ss, cy - 5 * ss, cx + 5 * ss, cy + 5 * ss], fill=CREAM, outline=INK, width=ss)
        else:
            dr.ellipse([cx - rr * ss, cy - rr * ss, cx + rr * ss, cy + rr * ss], fill=INK, outline=CREAM, width=3 * ss)
    img = img.resize((W, H), Image.LANCZOS)
    dr = ImageDraw.Draw(img)
    f = font(34)
    dr.text((mL - 200, mT - 20), "none\nremoved", font=font(26), fill=INK)
    dr.text((mL - 200, H - mB - 20), "all six\nremoved", font=font(26), fill=INK)
    # axis: removed activation
    yb = H - mB + 120
    dr.line([(X(0), yb), (X(total), yb)], fill=INK, width=2)
    for v in range(0, int(total) + 1, 1000):
        dr.line([(X(v), yb), (X(v), yb + 14)], fill=INK, width=2)
        dr.text((X(v) - 30, yb + 22), f"{v:,}", font=font(26), fill=INK)
    dr.text((X(0), yb + 70), f"massive activation removed (of {total:,.0f}, dimension 35, first token)", font=f, fill=INK)
    dr.text((mL, 70), "Committee", font=font(72), fill=INK)
    dr.text((mL, 170), "Qwen3-0.6B has no single super weight. Six weights in one row of layers.2.mlp.down_proj write its massive activation.",
            font=f, fill=INK)
    dr.text((mL, 220), f"All 64 ways to remove them. Disc area ∝ rise in held-out loss; the largest is +{dmax:.2f} nats "
            f"(perplexity {np.exp(base):.1f} → {np.exp(base + dmax):,.0f}).", font=f, fill=INK)
    out = os.path.join(GAL, "committee.png")
    img.save(out, optimize=True)
    print(out)


if __name__ == "__main__":
    main()
