"""The watershed: the whole vocabulary, the handful of first words it collapses into, and the
thousands of loops it spreads out into again.

Top band: every starting token, set in micro type, in one column per first generated word;
column width ∝ number of starting tokens (measured). Waist: the first words. Bottom band: one
column per terminal state (the largest seas individually, then all smaller seas together, then
the runs still talking at the 256-token cap), width ∝ number of starting tokens (measured),
filled with the loop text repeated (or, for the unresolved column, the last words each run was
saying at the cap). Ribbons carry the measured flows; ink is laid down at constant opacity so
overlaps darken (declared). Typefaces, band heights, ordering and colours are declared.
"""
import argparse
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

import typeset as ts
from analyze import load_run

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="cache/q06b")
ap.add_argument("--out", default="scratch/hourglass.png")
ap.add_argument("--W", type=int, default=7200)
ap.add_argument("--H", type=int, default=10200)
ap.add_argument("--nseas", type=int, default=24)
ap.add_argument("--micro", type=float, default=0, help="micro type size; 0 = fit all text")
ap.add_argument("--model", default="Qwen3-0.6B")
ap.add_argument("--dark", type=int, default=0)
ap.add_argument("--lw", type=float, default=0.9)
ap.add_argument("--alpha", type=float, default=0.18)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
starts, L, R, seqs = load_run(args.run)
z = np.load(args.run + "/analysis.npz")
B = json.load(open(args.run + "/basins.json"))
N = len(starts)
first = np.array([int(s[1]) for s in seqs])
sea = z["sea"]
cls = z["cls"]

if args.dark:
    PAPER, INK, GREY, RUB = (16, 15, 14), (236, 230, 216), (150, 144, 134), (222, 96, 72)
else:
    PAPER, INK, GREY, RUB = (243, 238, 226), (27, 26, 23), (118, 112, 102), (168, 48, 36)
W, H = args.W, args.H
M = int(W * 0.045)
gap = 8

# ---------------------------------------------------------------- groups
fw = collections.Counter(first.tolist())
words = [w for w, _ in fw.most_common()]
min_w_px = 10
usable = W - 2 * M
big_words = [w for w in words if fw[w] / N * usable >= min_w_px]
OTHER_W = -99
wkey = np.array([w if w in set(big_words) else OTHER_W for w in first])
wgroups = big_words + ([OTHER_W] if (wkey == OTHER_W).any() else [])
wcount = {g: int((wkey == g).sum()) for g in wgroups}

# terminal columns
named = [b["id"] for b in B if b["key"] != "EOS"][:args.nseas]
SMALL, UNRES, EOSC = -2, -3, -4
eos_ids = [b["id"] for b in B if b["key"] == "EOS"]
tkey = np.where(cls == 0, UNRES, np.where(np.isin(sea, eos_ids), EOSC,
               np.where(np.isin(sea, named), sea, SMALL)))
# order named seas by their dominant first-word group, then size
dom = {}
for s_ in named:
    c = collections.Counter(wkey[tkey == s_].tolist())
    dom[s_] = wgroups.index(c.most_common(1)[0][0])
named_sorted = sorted(named, key=lambda s_: (dom[s_], -int((tkey == s_).sum())))
tgroups = named_sorted + [SMALL] + ([EOSC] if (tkey == EOSC).any() else []) + [UNRES]
tcount = {g: int((tkey == g).sum()) for g in tgroups}


def spans(groups, counts, x0, x1, g=gap):
    tot = sum(counts[k] for k in groups)
    avail = (x1 - x0) - g * (len(groups) - 1)
    out, x = {}, x0
    for k in groups:
        w = counts[k] / tot * avail
        out[k] = (x, x + w)
        x += w + g
    return out


topA, botA = int(H * 0.085), int(H * 0.34)
waist_y0, waist_y1 = int(H * 0.435), int(H * 0.54)
topB, botB = int(H * 0.64), int(H * 0.925)
colA = spans(wgroups, wcount, M, W - M)
waist_x0, waist_x1 = W * 0.40, W * 0.60
waist = spans(wgroups, wcount, waist_x0, waist_x1, g=3)
colB = spans(tgroups, tcount, M, W - M)

# ---------------------------------------------------------------- streamlines (matplotlib layer)
# one hairline per starting token: from its place in the vocabulary column, through its first
# word at the neck, to its place in the terminal column.
from matplotlib.collections import LineCollection
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
fig.patch.set_alpha(0)
inkf = tuple(c / 255 for c in INK)
wi = np.array([wgroups.index(g) for g in wkey])
ti = np.array([tgroups.index(g) for g in tkey])
xt = np.zeros(N); xw = np.zeros(N); xb = np.zeros(N)
for gi, g in enumerate(wgroups):
    idx = np.nonzero(wi == gi)[0]
    idx = idx[np.lexsort((idx, ti[idx]))]
    a0, a1 = colA[g]; w0, w1 = waist[g]
    r = (np.arange(len(idx)) + 0.5) / len(idx)
    xt[idx] = a0 + r * (a1 - a0); xw[idx] = w0 + r * (w1 - w0)
for tj, t_ in enumerate(tgroups):
    idx = np.nonzero(ti == tj)[0]
    idx = idx[np.lexsort((idx, wi[idx]))]
    b0, b1 = colB[t_]
    xb[idx] = b0 + (np.arange(len(idx)) + 0.5) / len(idx) * (b1 - b0)
tt = np.linspace(0, 1, 28)[:, None]
def cub(x0, y0, x1, y1):
    h = y1 - y0
    X = (1 - tt) ** 3 * x0 + 3 * (1 - tt) ** 2 * tt * x0 + 3 * (1 - tt) * tt ** 2 * x1 + tt ** 3 * x1
    Y = (1 - tt) ** 3 * y0 + 3 * (1 - tt) ** 2 * tt * (y0 + 0.5 * h) + 3 * (1 - tt) * tt ** 2 * (y1 - 0.5 * h) + tt ** 3 * y1
    return X, Y
X1, Y1 = cub(xt[None], botA + 10, xw[None], waist_y0)
X2, Y2 = cub(xw[None], waist_y1, xb[None], topB - 10)
Xs = np.concatenate([X1, X2], 0).T; Ys = np.concatenate([np.broadcast_to(Y1, X1.shape), np.broadcast_to(Y2, X2.shape)], 0).T
segs = np.stack([Xs, Ys], -1)
lw = args.lw
alpha = args.alpha * min(1.0, 18956 / N) ** 0.5          # constant-ish ink per starting token (declared)
ax.add_collection(LineCollection(segs, colors=[inkf], linewidths=lw * 72 / 100, alpha=alpha))
# the neck: every line passes straight through it
ax.add_collection(LineCollection(np.stack([np.stack([xw, np.full(N, waist_y0)], -1), np.stack([xw, np.full(N, waist_y1)], -1)], 1),
                                 colors=[inkf], linewidths=lw * 72 / 100, alpha=alpha))
fig.canvas.draw()
layer = np.asarray(fig.canvas.buffer_rgba()).copy()
plt.close(fig)

img = Image.new("RGB", (W, H), PAPER)
img.paste(Image.fromarray(layer[..., :3]), (0, 0), Image.fromarray(layer[..., 3]))
d = ImageDraw.Draw(img)


# ---------------------------------------------------------------- micro text fill
def fill(rect, stream_iter, size, col):
    x0, y0, x1, y1 = rect
    lead = size * 1.18
    y = y0 + size
    x = x0
    guard = 0
    for piece in stream_iter:
        guard += 1
        if guard > 400000:
            return False
        vis = ts.visible(piece)
        if not vis.strip():
            vis = "␣" if vis else "∅"
        w = ts.length(vis, size, "roman")
        if x + w > x1 and x > x0:
            x = x0; y += lead
        if y > y1:
            return False
        x = ts.draw(d, (x, y), vis, size, "roman", col, max_x=x1)
    return True


vocab_text = {g: [tok.decode([int(s_)]) for s_ in starts[wkey == g]] for g in wgroups}
tot_chars = sum(sum(len(t) + 1 for t in v) for v in vocab_text.values())
areaA = (botA - topA) * (W - 2 * M)
micro = args.micro or float(np.sqrt(areaA / (tot_chars * 0.55 * 1.18)) * 0.97)
print("micro size", micro, "chars", tot_chars, flush=True)
def fit_size(rect, pieces, lead=1.18):
    """Largest type size (to 0.1 px) at which the pieces, set as running text with the same
    wrapping rule as fill(), fit inside the rectangle (declared: each column is set to fill)."""
    x0, y0, x1, y1 = rect
    vis = [(ts.visible(p) if ts.visible(p).strip() else ("␣" if p else "∅")) for p in pieces]
    w100 = np.array([ts.length(v, 100.0, "roman") for v in vis]) / 100.0

    def fits(sz):
        # glyph widths scale ~linearly with size (fonts are rounded to integer px when drawn)
        w = w100 * sz
        x, y = x0, y0 + sz
        for wi in w:
            if x + wi > x1 and x > x0:
                x = x0; y += sz * lead
                if y > y1:
                    return False
            x += wi
        return True
    lo, hi = 2.0, 80.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


for g in wgroups:
    a0, a1 = colA[g]
    pieces = [t + " " for t in vocab_text[g]]
    sz = min(fit_size((a0, topA, a1, botA), pieces), 60)
    print("column", g, len(pieces), "size", round(sz, 2), flush=True)
    fill((a0, topA, a1, botA), iter(pieces), sz, GREY)

# terminal columns
def loop_stream(sid):
    per = tok.decode(B[sid]["rep"]) or "∅"
    while True:
        yield per
unres_idx = np.nonzero(tkey == UNRES)[0]
for t_ in tgroups:
    b0, b1 = colB[t_]
    if t_ >= 0:
        fill((b0, topB, b1, botB), loop_stream(t_), micro * 1.15, INK)
    elif t_ == SMALL:
        ids = sorted(set(sea[tkey == SMALL].tolist()))
        pieces = [tok.decode(B[i]["rep"]) + "  " for i in ids]
        fill((b0, topB, b1, botB), iter(pieces), fit_size((b0, topB, b1, botB), pieces), INK)
    elif t_ == UNRES:
        pieces = ["…" + tok.decode(seqs[i][-8:].tolist()) + "  " for i in unres_idx]
        fill((b0, topB, b1, botB), iter(pieces), fit_size((b0, topB, b1, botB), pieces), GREY)

# ---------------------------------------------------------------- words in the neck, knocked out
for g in wgroups:
    b0, b1 = waist[g]
    wd = "others" if g == OTHER_W else ts.visible(tok.decode([g])).strip()
    hmax = (waist_y1 - waist_y0) * 0.86
    size = min((b1 - b0) * 0.78, 150)
    if size < 9:
        continue
    lw_ = ts.length(wd, size, "italic")
    if lw_ > hmax:
        size *= hmax / lw_; lw_ = hmax
    tmp = Image.new("RGBA", (int(lw_) + 8, int(size * 1.45)), (0, 0, 0, 0))
    ts.draw(ImageDraw.Draw(tmp), (4, size * 1.08), wd, size, "italic", PAPER)
    tmp = tmp.rotate(90, expand=True)
    cy = (waist_y0 + waist_y1) / 2
    img.paste(tmp, (int((b0 + b1) / 2 - tmp.width / 2 + size * 0.12), int(cy - tmp.height / 2)), tmp)

# column captions (counts) under the bottom band
for t_ in tgroups:
    b0, b1 = colB[t_]
    if b1 - b0 < 60:
        continue
    lab = f"{tcount[t_]:,}"
    if t_ == SMALL:
        lab += f"  ·  {len(set(sea[tkey == SMALL].tolist())):,} smaller loops"
    elif t_ == UNRES:
        lab += "  ·  still talking at token 256"
    ts.draw(d, (b0, botB + 46), lab, 30, "roman", GREY, max_x=b1)

# ---------------------------------------------------------------- titles
ts.draw(d, (M, topA - 150), "The Watershed", 120, "italic", INK)
ts.draw(d, (M, topA - 60), f"{N:,} starting tokens of {args.model}, each followed by the word the model greedily writes next, "
        f"and where the text ends up.  Column widths ∝ number of starting tokens.", 34, "roman", GREY)
ts.draw(d, (waist_x1 + 60, (waist_y0 + waist_y1) / 2), f"{len(fw):,} distinct first words", 40, "italic", GREY)
ts.draw(d, (waist_x1 + 60, (waist_y0 + waist_y1) / 2 + 52), ", ".join(ts.visible(tok.decode([w])).strip() for w in words[:8]) + ", …", 30, "italic", GREY)
ts.draw(d, (M, topB - 40), f"{len(B):,} loops, and the runs that had not looped by token 256", 34, "italic", GREY)
img.save(args.out)
print("saved", args.out)
