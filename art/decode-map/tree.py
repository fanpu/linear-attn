"""Rose window: the prefix trie of a decode map drawn as a sunburst.

Ring l (from the centre outward) holds every distinct prefix of length l that occurs
anywhere on the (T, p) grid. The angular width of a wedge is the fraction of grid samples
that produce that prefix, and its colour is the mean temperature of those samples
(declared map: cmc.lipari, low T dark blue, high T pale gold). Children are ordered
by mean temperature inside their parent. Everything is measured. The one choice is to
order siblings by mean T, which only permutes them within their parent.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm  # noqa

import analysis as A
import render as Rr


def build_nodes(d, Lmax=None):
    tk = d["tokens"]
    H, W, L = tk.shape
    L = Lmax or L
    hs = A.prefix_hashes(tk)
    Tpix = np.broadcast_to(d["xs"][None, :], (H, W)).ravel()
    levels = []
    prev = np.zeros(H * W, np.int64)
    for l in range(L):
        lab = A.labels_from_hash(hs[l]).ravel()
        n = lab.max() + 1
        area = np.bincount(lab, minlength=n)
        meanT = np.bincount(lab, Tpix, minlength=n) / area
        parent = np.zeros(n, np.int64)
        parent[lab] = prev
        levels.append(dict(area=area, meanT=meanT, parent=parent))
        prev = lab
    return levels


def layout(levels, total):
    """Angles: children partition their parent's arc in order of mean T."""
    start = [np.zeros(1)]
    span = [np.array([2 * np.pi])]
    for l, lv in enumerate(levels):
        order = np.lexsort((lv["meanT"], lv["parent"]))
        st = np.zeros(len(lv["area"]))
        sp = lv["area"] / total * 2 * np.pi
        pst = start[-1]
        acc = {}
        for c in order:
            p = lv["parent"][c] if l > 0 else 0
            base = pst[p] if l > 0 else 0.0
            st[c] = base + acc.get(p, 0.0)
            acc[p] = acc.get(p, 0.0) + sp[c]
        start.append(st)
        span.append(sp)
    return start[1:], span[1:]


def draw(d, out, size=2400, r0_frac=0.07, cmap="cmc.lipari", ground=(8, 8, 10), line=(8, 8, 10),
         Lmax=None, gamma=0.75, ss=2, line_w=1.0):
    levels = build_nodes(d, Lmax)
    H, W = d["tokens"].shape[:2]
    st, sp = layout(levels, H * W)
    L = len(levels)
    S = size * ss
    img = Image.new("RGB", (S, S), ground)
    dr = ImageDraw.Draw(img)
    c = S / 2
    R = S / 2 * 0.96
    r0 = R * r0_frac
    # ring radii: declared sqrt-like spacing so outer rings (many thin wedges) get more room
    radii = r0 + (R - r0) * (np.arange(L + 1) / L) ** gamma
    cm = plt.get_cmap(cmap)
    x0, x1 = d["xs"][0], d["xs"][-1]
    for l in range(L - 1, -1, -1):                        # fills, outer first
        ro, ri = radii[l + 1], radii[l]
        for k in range(len(sp[l])):
            a0 = np.degrees(st[l][k]) - 90
            a1 = a0 + np.degrees(sp[l][k])
            col = tuple(int(255 * v) for v in cm((levels[l]["meanT"][k] - x0) / (x1 - x0))[:3])
            dr.pieslice([c - ro, c - ro, c + ro, c + ro], a0, a1, fill=col)
        dr.ellipse([c - ri, c - ri, c + ri, c + ri], fill=ground)
    lw = max(1, int(line_w * ss))
    for l in range(L):                                    # leading: radial edges + arcs where a prefix splits
        ro, ri = radii[l + 1], radii[l]
        par = levels[l]["parent"]
        nsib = np.bincount(par) if l > 0 else np.array([len(sp[0])])
        for k in range(len(sp[l])):
            if sp[l][k] * ro < 6 * ss:
                continue                                  # sub-pixel wedge: leave as texture
            th = st[l][k] - np.pi / 2
            if len(sp[l]) > 1:                            # every wedge start is a boundary between distinct prefixes
                dr.line([(c + ri * np.cos(th), c + ri * np.sin(th)), (c + ro * np.cos(th), c + ro * np.sin(th))],
                        fill=line, width=lw)
            if nsib[par[k] if l > 0 else 0] > 1:          # arc only where this prefix was born
                a0 = np.degrees(th)
                dr.arc([c - ri, c - ri, c + ri, c + ri], a0, a0 + np.degrees(sp[l][k]), fill=line, width=lw)
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out, optimize=True)
    print(out, [len(lv["area"]) for lv in levels][-1], "leaf wedges")


if __name__ == "__main__":
    d = A.load(sys.argv[1])
    draw(d, sys.argv[2], Lmax=int(sys.argv[3]) if len(sys.argv) > 3 else None)
