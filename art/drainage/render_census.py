"""The seas, by size: a typeset census of every loop the model falls into.

One line per sea (loop), in order of basin size. The line is the loop text repeated to the end of
the measure, starting at the phase the model most often enters it. Type size ∝ sqrt(count), so
the area of ink per character is proportional to the number of starting tokens (declared). Below
a floor size the remaining seas are set once each, in running text, at the floor size.
Invisible characters are transcribed (newline ¶, tab ⇥) and printed in the second ink (declared).
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoTokenizer

import typeset as ts

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="cache/q06b")
ap.add_argument("--out", default="gallery/census.png")
ap.add_argument("--W", type=int, default=6000)
ap.add_argument("--H", type=int, default=8400)
ap.add_argument("--k", type=float, default=2.2, help="px of type size per sqrt(count)")
ap.add_argument("--floor", type=float, default=14)
ap.add_argument("--tail_size", type=float, default=11)
ap.add_argument("--title", default="Drainage")
ap.add_argument("--model", default="Qwen3-0.6B")
ap.add_argument("--dark", type=int, default=0)
ap.add_argument("--lines", default="", help="caption lines separated by |, overriding the default")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
B = json.load(open(args.run + "/basins.json"))
z = np.load(args.run + "/analysis.npz")
N = len(z["cls"])
n_cyc = int((z["cls"] == 1).sum()); n_eos = int((z["cls"] == 2).sum()); n_unres = int((z["cls"] == 0).sum())

PAPER = (243, 238, 226) if not args.dark else (17, 16, 15)
INK = (27, 26, 23) if not args.dark else (236, 230, 216)
RUB = (168, 48, 36) if not args.dark else (222, 96, 72)     # rubrication for invisible chars
GREY = (120, 114, 104)

AUTO = args.H == 0
if AUTO:
    args.H = 60000
img = Image.new("RGB", (args.W, args.H), PAPER)
d = ImageDraw.Draw(img)
M = int(args.W * 0.06)
colnum = M + int(args.W * 0.07)       # right edge of numerals
x0 = colnum + int(args.W * 0.02)
x1 = args.W - M


def loop_text(b, reps):
    if b["key"] == "EOS":
        return None
    return tok.decode(b["rep"] * reps)


def draw_rubricated(xy, text, size, style, max_x):
    """Draw with ¶ / ⇥ in the second ink."""
    x, y = xy
    buf = ""
    for ch in text:
        if ch in "\n\t":
            if buf:
                x = ts.draw(d, (x, y), buf, size, style, INK, max_x=max_x); buf = ""
            if x >= max_x:
                return x
            x = ts.draw(d, (x, y), "¶" if ch == "\n" else "⇥", size, style, RUB, max_x=max_x)
        else:
            buf += ch
    if buf:
        x = ts.draw(d, (x, y), buf, size, style, INK, max_x=max_x)
    return x


# --- title block
y = M + 150
ts.draw(d, (M, y), args.title, 150, "italic", INK)
y += 80
lines = [
    f"Every loop that {args.model} falls into when it is started, with no prompt, from a single token of its vocabulary and left to",
    f"decode greedily.  {N:,} starting tokens: {n_cyc:,} fell into a loop within 256 tokens, {n_eos:,} stopped, {n_unres:,} were still talking at 256.",
    f"{len(B):,} distinct loops.  One line per loop, largest basin first; type size ∝ √(starting tokens drained).  ¶ marks a newline.",
]
if args.lines:
    lines = args.lines.split("|")
for ln in lines:
    y += 62
    draw_rubricated((M, y), ln, 44, "roman", x1)
y += 70
d.line([(M, y), (x1, y)], fill=INK, width=2)
y += 40

# --- lines
i = 0
while i < len(B):
    b = B[i]
    size = args.k * np.sqrt(b["count"])
    if size < args.floor:
        break
    lead = size * 1.12
    if not AUTO and y + lead > args.H - M - 400:
        break
    y += lead
    num = f"{b['count']:,}"
    ts.draw(d, (colnum - ts.length(num, min(size, 60) * 0.7, "roman"), y), num, min(size, 60) * 0.7, "roman", GREY)
    if b["key"] == "EOS":
        ts.draw(d, (x0, y), "(end of text)", size, "italic", GREY)
    else:
        per = len(tok.decode(b["rep"])) or 1
        reps = int((x1 - x0) / max(1.0, 0.35 * size * per)) + 2
        draw_rubricated((x0, y), loop_text(b, reps), size, "italic", x1)
    i += 1

# --- tail: remaining seas set once each in running text
y += 40
d.line([(x0, y), (x1, y)], fill=INK, width=1)
y += 20
rest = B[i:]
ts.draw(d, (x0, y + 36), f"and {len(rest):,} smaller loops, each set once, largest first:", 34, "italic", GREY)
y += 60
size = args.tail_size
lead = size * 1.25
x = x0
y += lead
for k, b in enumerate(rest):
    if b["key"] == "EOS":
        continue
    t = tok.decode(b["rep"])
    w = ts.length(ts.visible(t), size, "italic") + size * 0.9
    if x + w > x1:
        x = x0; y += lead
        if y > args.H - M - 120:
            left = len(rest) - k
            y += lead * 2.2
            ts.draw(d, (x0, y), f"… and {left:,} more loops that did not fit on this sheet (all {len(B):,} are listed in census.tsv).", 34, "italic", GREY)
            print("tail truncated, left", left); break
    x = draw_rubricated((x, y), t, size, "italic", x1)
    x = ts.draw(d, (x, y), " · ", size, "roman", GREY)
if AUTO:
    img = img.crop((0, 0, args.W, int(y + M)))
img.save(args.out)
print("saved", args.out, "lines", i, "tail", len(rest), "last y", y, "size", img.size)
