"""Plate series 'Instead': a model's single most frequent reply to "Please repeat the string '<X>'."
over its verified-unsayable tokens, set large in madder, above every token that drew exactly that
reply, set as one justified block in ink.

Measured: the reply string (verbatim, chat template, greedy) and the set of tokens that produced it.
Declared: token order (indicator, most under-trained first), the block's type size (the largest that
fills the text area), justification, transcription marks (grey)."""
import argparse, json, os
from collections import Counter
import look as L
import typeset as ts
from render_register import NICE, CACHE


def layout(items, size, width, gap):
    lines, cur, w = [], [], 0.0
    for s, iw in items:
        iw = iw * size
        if cur and w + gap * size + iw > width:
            lines.append(cur); cur, w = [], 0.0
        w += (gap * size if cur else 0) + iw
        cur.append((s, iw))
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--field", default="chat")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = [r for r in json.load(open(os.path.join(CACHE, f"{a.model}_table.json")))
         if r["set"] == "cand" and r["verified"]]
    c = Counter(r[a.field] for r in R)
    reply, n = c.most_common()[a.rank]
    toks = sorted([r for r in R if r[a.field] == reply], key=lambda r: r["ind"])

    im, d = L.sheet()
    M = 250
    right = L.W - M
    width = right - M
    L.smallcaps(d, (M, 300), f"{NICE[a.model].upper()}, ASKED TO REPEAT EACH OF THESE {n} TOKENS, SAID", 34, L.GREY, track=0.12)
    # the reply, large, in madder; placed by its measured ink box (declared: size = up to 1300 px,
    # shrunk to fit the measure)
    from PIL import Image, ImageDraw
    rs = 1300
    wv = L.vis_width(reply, rs)
    if wv > width:
        rs *= width / wv
    lines = [reply]
    top, y = 420, 420
    if rs < 260 and "\n" in reply:
        # declared: a long looping reply is set as a justified run, one item per model-emitted line
        # (the line breaks themselves are not drawn); height budget 1500 px
        lines = [x for x in reply.split("\n") if x.strip()]
        unit_r = [(x, L.vis_width(x, 100) / 100) for x in lines]
        lo, hi = 20.0, 1300.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if len(layout(unit_r, mid, width, 0.35)) * mid * 1.25 <= 1500:
                lo = mid
            else:
                hi = mid
        rs = lo
        for li, line in enumerate(layout(unit_r, rs, width, 0.35)):
            tw = sum(w for _, w in line)
            g = (width - tw) / (len(line) - 1) if len(line) > 1 else 0
            x = M
            for t, w in line:
                L.draw_visible(d, (x, y + rs), t, rs, L.MADDER, marker_fill=(200, 150, 140))
                x += w + g
            y += rs * 1.25
    else:
        scratch = Image.new("L", (L.W + int(rs), int(rs * 2.2)), 0)
        L.draw_visible(ImageDraw.Draw(scratch), (0, rs * 1.3), reply, rs, 255, marker_fill=150)
        glyph = scratch.crop(scratch.getbbox())
        im.paste(Image.new("RGB", glyph.size, L.MADDER), (M, top), glyph)
        y = top + glyph.size[1]
    y0 = y + 200
    d.line([(M, y0 - 90), (right, y0 - 90)], fill=L.INK, width=3)
    y1 = L.H - 420
    # block of tokens: largest size that fits
    unit = [(r["str"], L.vis_width(r["str"], 100) / 100) for r in toks]
    gap, lead = 0.62, 1.42
    lo, hi = 8.0, 400.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ls = layout(unit, mid, width, gap)
        if len(ls) * mid * lead <= (y1 - y0):
            lo = mid
        else:
            hi = mid
    size = lo
    lines = layout(unit, size, width, gap)
    y = y0 + size
    for li, line in enumerate(lines):
        tw = sum(w for _, w in line)
        g = gap * size if (li == len(lines) - 1 or len(line) == 1) else (width - tw) / (len(line) - 1)
        x = M
        for s, w in line:
            L.draw_visible(d, (x, y), s, size, L.INK)
            x += w + g
        y += size * lead
    d.line([(M, L.H - 360), (right, L.H - 360)], fill=L.INK, width=3)
    tot = len(R)
    notes = [f"{n} of the {tot} tokens {NICE[a.model]} cannot say (p < 0.01) drew exactly this reply to “Please repeat the string '‹token›'.”",
             "Chat template, thinking off, greedy decoding; the reply is verbatim up to end-of-turn" + (" or the 40-token cap" if len(lines) > 1 else "") + ". A boxed word is one of the model's own control tokens.",
             "Tokens fed as ids, set in ink in indicator order (most under-trained first) at the largest size that fills the block. Grey: ␣ space, ↵ newline."]
    for k, t in enumerate(notes):
        ts.draw(d, (M, L.H - 290 + 50 * k), t, 34, [(ts.F["text_i"], 0)] + ts.SERIF_CHAIN, L.GREY)
    im.save(a.out, optimize=True)
    print("wrote", a.out, repr(reply), n, "tokens at", round(size, 1), "px")


if __name__ == "__main__":
    main()
