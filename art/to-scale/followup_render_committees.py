"""Follow-up plate: committee or single weight, across Qwen3 sizes and Base vs post-trained.

One row per model. The bar is the massive activation written by the spiking mlp.down_proj
(|out[row]| at the spike position), on one common scale for every row (measured). It is cut into the
per-weight contributions W[row, c] * x[c] that write it, largest first (measured); a single super
weight is one slab, a committee is several. Contributions after the 12th, and any of opposite sign,
are left out of the slabs (declared; they are < 1% in every model measured).
Right: three discs, area ∝ rise in held-out loss (nats) when the top weight alone is zeroed (open) and
when the k95 weights are zeroed together (filled), and a grey disc (floored at 4 px) for 100 random same-|w| weights
zeroed together (the null). Common area scale on every row (measured). Ink = post-trained checkpoint,
madder = -Base checkpoint (declared). Cream ground, C059 type (declared).
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CREAM = (242, 237, 227)
INK = (27, 26, 31)
MADDER = (170, 52, 44)
GREY = (140, 136, 130)
SERIF = "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"
ITAL = "/usr/share/fonts/opentype/urw-base35/C059-Italic.otf"
ROWS = [("qwen3-0.6b-base", "Qwen3-0.6B-Base", MADDER), ("qwen3-0.6b", "Qwen3-0.6B", INK),
        ("qwen3-1.7b-base", "Qwen3-1.7B-Base", MADDER), ("qwen3-1.7b", "Qwen3-1.7B", INK),
        ("qwen3-4b", "Qwen3-4B", INK), ("qwen3-8b", "Qwen3-8B", INK), ("qwen3-14b", "Qwen3-14B", INK)]


def f(s, it=False):
    return ImageFont.truetype(ITAL if it else SERIF, s)


def main(out):
    rows = []
    for tag, name, col in ROWS:
        p = os.path.join(CACHE, f"followup_sw_{tag}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            if "base_held_nll" in r and "null_all_held_nll" in r:
                rows.append((tag, name, col, r))
    W = 3000
    mL, barL, barR = 150, 760, 2150
    discX = [2380, 2640, 2860]
    rowH = 330
    top = 420
    H = top + rowH * len(rows) + 330
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    d.text((mL, 70), "Committee or single", font=f(84), fill=INK)
    d.text((mL, 190), "The massive activation of each model, cut into the weights that write it (one row of the first spiking "
           "mlp.down_proj; same scale on every row).", font=f(34), fill=INK)
    d.text((mL, 240), "Discs: area ∝ rise in held-out loss when the largest weight alone is zeroed (open), when the fewest weights "
           "writing 95% of it are zeroed (filled),", font=f(34), fill=INK)
    d.text((mL, 290), "and when 100 random weights of the same magnitude are zeroed together (grey; floored at a 4 px dot). Ink: post-trained checkpoint. "
           "Madder: -Base checkpoint.", font=f(34), fill=INK)
    vmax = max(abs(r["out_at_pos"]) for *_, r in rows)
    sc = (barR - barL) / vmax
    dmax = max(max(r["cum_held_nll"][0], r["k95_held_nll"]) - r["base_held_nll"] for *_, r in rows)
    rmax = 120  # px radius for the largest loss rise
    d.text((discX[0], top - 70), "top one", font=f(28, True), fill=GREY, anchor="ms")
    d.text((discX[1], top - 70), "all k95", font=f(28, True), fill=GREY, anchor="ms")
    d.text((discX[2], top - 70), "null", font=f(28, True), fill=GREY, anchor="ms")
    for k, (tag, name, col, r) in enumerate(rows):
        y = top + k * rowH + 110
        d.text((mL, y - 20), name, font=f(42), fill=col)
        d.text((mL, y + 36), f"layers.{r['layer']}.mlp.down_proj", font=f(26), fill=GREY)
        d.text((mL, y + 70), f"row {r['row']}", font=f(26), fill=GREY)
        tot = abs(r["out_at_pos"])
        sgn = 1 if r["out_at_pos"] >= 0 else -1
        x = barL
        th = 64
        n_seg = 0
        for c, v in zip(r["contrib_order"][:12], r["contrib_top"][:12]):
            if v * sgn <= 0:
                continue
            w = v * sgn * sc
            if w < 1:
                continue
            d.rectangle([x, y - th / 2, x + w - 1, y + th / 2], fill=col)
            if w > 70:
                d.text((x + 10, y + th / 2 + 34), f"[{r['row']},{c}]" if n_seg == 0 else f"{c}", font=f(22), fill=GREY)
            x += w
            n_seg += 1
            if x - barL > 0.999 * tot * sc:
                break
            d.line([(x, y - th / 2 - 6), (x, y + th / 2 + 6)], fill=CREAM, width=5)
        d.text((barL + tot * sc + 18, y - 4), f"{tot:,.0f}", font=f(30), fill=col, anchor="lm")
        share1 = abs(r["contrib_top"][0]) / tot
        d.text((barL, y - th / 2 - 44), f"k95 = {r['k95']}   ·   the largest single weight writes {share1:.1%}   ·   "
               f"massive / median {r['ratio']:,.0f}", font=f(28, True), fill=col)
        b = r["base_held_nll"]
        rises = [r["cum_held_nll"][0] - b, r["k95_held_nll"] - b, r["null_all_held_nll"] - b]
        for j, (dx, dv) in enumerate(zip(discX, rises)):
            rad = rmax * math.sqrt(max(dv, 0) / dmax)
            if j == 2:
                rr = max(rad, 4)
                d.ellipse([dx - rr, y - rr, dx + rr, y + rr], fill=GREY)
            elif j == 0:
                rr = max(rad, 3)
                d.ellipse([dx - rr, y - rr, dx + rr, y + rr], outline=col, width=4)
            else:
                rr = max(rad, 3)
                d.ellipse([dx - rr, y - rr, dx + rr, y + rr], fill=col)
            ppl = math.exp(b + dv)
            d.text((dx, y + rmax + 20), f"{ppl:,.1f}" if ppl < 1000 else f"{ppl:,.0f}", font=f(26), fill=GREY if j == 2 else col,
                   anchor="ms")
        d.text((discX[0] - 150, y + rmax + 20), f"ppl {math.exp(b):.1f} →", font=f(26), fill=GREY, anchor="rs")
    d.text((mL, H - 150), "Measured on fineweb-edu: location and contributions from 16 pages; perplexity on 8 held-out pages; "
           "fp32 (Qwen3-14B: bf16). Bar length and disc area are linear in the measured values.", font=f(28, True), fill=GREY)
    im.save(out, optimize=True)
    print("wrote", out, [t for t, *_ in rows])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gallery", "followup_committee_or_single.png"))
