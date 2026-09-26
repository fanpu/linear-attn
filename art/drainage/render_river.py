"""One sea and its rivers, set as text.

Flow runs left to right, which is also reading order and generation order: a starting token
(a spring, at the left) is followed by the text the model generates, until that text joins other
texts and finally enters the loop (the sea, at the right, set as the loop text repeated).

Measured: the tree (which starting tokens share which text, token for token, before the loop),
river width ∝ sqrt(number of starting tokens drained), river length = the typeset length of its
own text (x = typeset width of the text between a point and the sea, at the reference size).
Declared: vertical placement (every starting token gets an equal slot, largest tributary in the
middle), the right-angle-with-rounded-bend joint geometry, typefaces, and colours.
"""
import argparse
import json
import pickle

import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

import typeset as ts
from rivers import SeaTree

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="cache/q06b")
ap.add_argument("--sea", type=int, default=5)
ap.add_argument("--out", default="scratch/river.png")
ap.add_argument("--W", type=int, default=7200)
ap.add_argument("--H", type=int, default=4800)
ap.add_argument("--ref", type=float, default=30, help="reference type size (px) for river text")
ap.add_argument("--ref_max", type=float, default=64)
ap.add_argument("--fixed_ref", type=int, default=0, help="1 = same type size on every plate (typology)")
ap.add_argument("--wk", type=float, default=1.6, help="stroke px per sqrt(count)")
ap.add_argument("--label_min", type=int, default=2, help="label rivers draining >= this many starts")
ap.add_argument("--xscale", type=float, default=0, help="0 = fit")
ap.add_argument("--ss", type=int, default=2, help="supersampling")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
B = json.load(open(args.run + "/basins.json"))
nodes = pickle.load(open(args.run + "/tree.pkl", "rb"))["nodes"]
t = SeaTree(nodes, args.sea)
b = B[args.sea]

S = args.ss
W, H = args.W * S, args.H * S
PAPER = (243, 238, 226); INK = (27, 26, 23); RUB = (168, 48, 36); GREY = (130, 124, 114)
img = Image.new("RGB", (W, H), PAPER)
LEAF = (170, 163, 150)
d = ImageDraw.Draw(img)

M = 260 * S
sea_w = int(W * 0.24)
x_sea = W - M - sea_w
top, bot = M + 330 * S, H - M - 40 * S

# --- typeset distance upstream, per node
tw = {}
def tokw(tk):
    if tk not in tw:
        tw[tk] = ts.length(ts.visible(tok.decode([int(tk)])), args.ref * S, "italic")
    return tw[tk]
dist = np.zeros(t.n)
order = np.argsort(t.depth, kind="stable")
for i in order:
    p = t.parent[i]
    dist[i] = (dist[p] if p >= 0 else 0.0) + tokw(t.token[i])
# fit the shared rivers (count >= 2) fully; single private streams longer than that are cut at
# the margin and marked with an arrowhead (declared)
span = max(dist[t.count >= 2].max() * 1.25 if (t.count >= 2).any() else dist.max(), 1.0)
span = min(span, dist.max())
fit = (x_sea - M) / span
if args.fixed_ref:
    pass
elif abs(fit - 1.0) > 0.01:                     # scale the type so the rivers fill the width
    args.ref = min(args.ref * fit * 0.995, args.ref_max)
    tw.clear()
    for i in order:
        p = t.parent[i]
        dist[i] = (dist[p] if p >= 0 else 0.0) + tokw(t.token[i])
    span = min(max(dist[t.count >= 2].max() * 1.25 if (t.count >= 2).any() else dist.max(), 1.0), dist.max())
xs = 1.0
X = x_sea - dist * xs

# --- vertical slots
t.layout(x0=0, slot=1.0)
dense = (bot - top) / t.width < args.ref * S * 0.75
Y = top + (t.x / t.width) * (bot - top)

nk = np.array([len(t.kids.get(i, [])) for i in range(t.n)])
drawn = (nk != 1) | (t.parent == -1)
low = {}
for i in np.nonzero(drawn)[0]:
    p = t.parent[i]
    chain = [i]
    while p != -1 and not drawn[p]:
        chain.append(p); p = t.parent[p]
    low[i] = (p, chain)

def rub_draw(d, xy, text, size, col, max_x, halo=None):
    x, y = xy; buf = ""
    for ch in text + "\0":
        if ch in "\n\t\0":
            if buf:
                x = ts.draw(d, (x, y), buf, size, "italic", col, max_x=max_x, halo=halo); buf = ""
            if ch != "\0" and x < max_x:
                x = ts.draw(d, (x, y), "¶" if ch == "\n" else "⇥", size, "italic", RUB, max_x=max_x, halo=halo)
        else:
            buf += ch
    return x


# edges: horizontal run at the child's y from its upstream end to the junction x, then a
# rounded bend into the parent's y.  The label is the chain's own text.
order_draw = sorted(low.keys(), key=lambda i: t.count[i])        # thin first, thick on top
labels = []
for i in order_draw:
    p, chain = low[i]
    w = max(1, args.wk * S * np.sqrt(t.count[i]))
    xa = X[i]                                    # upstream end of this chain's text
    cut = xa < M
    xa = max(xa, M)
    xb = X[p] if p >= 0 else x_sea               # junction = where the parent's text begins
    y0 = Y[i]; y1 = Y[p] if p >= 0 else Y[i]
    r = min(abs(y1 - y0), 14 * S, max(1.0, (xb - xa) * 0.5))
    pts = [(xa, y0), (xb - r, y0)]
    if abs(y1 - y0) > 0.5:
        th = np.linspace(0, np.pi / 2, 12)
        sgn = 1 if y1 > y0 else -1
        pts += [(xb - r + r * np.sin(a), y0 + sgn * r * (1 - np.cos(a))) for a in th]
        pts += [(xb, y1)]
    else:
        pts += [(xb, y0)]
    col = INK if t.count[i] >= 2 else GREY
    if t.count[i] == 1 and dense:
        col = LEAF
    d.line(pts, fill=col, width=int(round(w)), joint="curve")
    # label: the chain's own text; rivers draining >= label_min starts in ink, single springs grey
    txt = tok.decode([int(t.token[c]) for c in chain])
    big = t.count[i] >= args.label_min
    size = args.ref * S * (1.0 if big else 0.8)
    room = t.count[i] / t.width * (bot - top)          # vertical room this river owns
    if cut:
        d.polygon([(M - 14 * S, y0), (M, y0 - 5 * S), (M, y0 + 5 * S)], fill=col)
        txt = "…" + txt[-max(4, int((xb - xa) / (0.45 * size))):] if len(txt) > 4 else txt
    if room >= 0.75 * size or (big and room >= 0.45 * size):
        labels.append(((xa, y0 - w / 2 - 5 * S), txt, size, INK if big else GREY, xb + 2 * S))

# labels after all strokes, each with a paper halo so crossing hairlines stop at the letters
for xy, txt, size, col, mx in labels:
    x_end = min(mx, xy[0] + ts.length(ts.visible(txt), size, "italic"))
    d.rectangle([xy[0] - 2 * S, xy[1] - size * 0.78, x_end + 2 * S, xy[1] + size * 0.2], fill=PAPER)
    rub_draw(d, xy, txt, size, col, mx)

# springs that are also starting tokens (own > 0 on an interior node): a small ring
for i in np.nonzero((t.own > 0) & (nk > 0))[0]:
    rr = 5 * S
    d.ellipse([X[i] - rr, Y[i] - rr, X[i] + rr, Y[i] + rr],
              outline=RUB, width=2 * S)

# --- the sea: the loop text, repeated as one continuous stream wrapped into the basin
sz = args.ref * S * 1.25
lead = sz * 1.3
per = tok.decode(b["rep"])
stream = per * max(4000, 60000 // max(1, len(per)))
yy = top
pos = 0
x_end = W - M
while yy <= bot + lead * 0.2:
    # take as many characters as fit on this line
    lo, hi = 1, 400
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ts.length(ts.visible(stream[pos:pos + mid]), sz, "italic") <= x_end - x_sea - 20 * S:
            lo = mid
        else:
            hi = mid - 1
    rub_draw(d, (x_sea + 20 * S, yy), stream[pos:pos + lo], sz, INK, x_end)
    pos += lo
    yy += lead
d.line([(x_sea + 6 * S, top - lead), (x_sea + 6 * S, bot + lead * 0.3)], fill=INK, width=2 * S)

# --- heading
title = "“" + tok.decode(b["rep"]).strip() + "”"
rub_draw(d, (M, M + 100 * S), title.replace("\n", " \n "), 96 * S, INK, x_sea)
ts.draw(d, (M, M + 178 * S), f"Sea no. {args.sea + 1} of {len(B):,}.  {b['count']:,} starting tokens drain into this loop of {b['period']} token{'s' if b['period'] > 1 else ''}"
        f" (median {b['entry_median']:.0f} tokens to arrive).  Each river is set in its own text and runs, as it was written, left to right.", 34 * S, "roman", GREY)
img = img.resize((args.W, args.H), Image.LANCZOS)
img.save(args.out)
print("saved", args.out, "nodes", t.n, "span px", span / S)
