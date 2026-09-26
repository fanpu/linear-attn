"""Plan-view river layout: edge length = tokens (measured), junction angles declared."""
import json, pickle, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
from rivers import SeaTree
d, out = sys.argv[1], sys.argv[2]; seas = [int(a) for a in sys.argv[3].split(",")]
SPREAD = math.radians(float(sys.argv[4]) if len(sys.argv) > 4 else 150); GAM = 0.5; s = 9.0
B = json.load(open(d + "/basins.json")); nodes = pickle.load(open(d + "/tree.pkl", "rb"))["nodes"]
W, H = 2400 * len(seas), 2400
fig = plt.figure(figsize=(W/72, H/72), dpi=72); ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
fig.patch.set_facecolor("#f3eee2")
for j, sea in enumerate(seas):
    t = SeaTree(nodes, sea)
    nk = np.array([len(t.kids.get(i, [])) for i in range(t.n)])
    drawn = (nk != 1) | (t.parent == -1)
    # compressed children: for drawn node, its drawn descendants
    down = {}
    for i in np.nonzero(drawn)[0]:
        p = t.parent[i]
        while p != -1 and not drawn[p]: p = t.parent[p]
        down.setdefault(p, []).append(i)
    segs, lws = [], []
    def place(i_list, base, heading, sector, base_depth):
        ch = sorted(i_list, key=lambda c: -t.count[c])
        w = np.array([t.count[c] ** GAM for c in ch], float)
        tot = min(sector, SPREAD)
        # centre-out ordering: largest in the middle
        order = []
        for k, c in enumerate(ch): (order.append if k % 2 == 0 else (lambda v: order.insert(0, v)))(c)
        ww = np.array([t.count[c] ** GAM for c in order], float); ww = ww / ww.sum() * tot
        a0 = heading - tot / 2
        for c, wc in zip(order, ww):
            h = a0 + wc / 2; a0 += wc
            L = (t.depth[c] - base_depth) * s
            q = base + L * np.array([math.cos(h), math.sin(h)])
            segs.append(np.array([base, q])); lws.append(0.5 * math.sqrt(t.count[c]))
            if c in down: place(down[c], q, h, max(wc, math.radians(40)), t.depth[c])
    import sys as _s; _s.setrecursionlimit(100000)
    place(down[-1], np.array([1200.0 + 2400 * j, 150.0]), math.pi / 2, math.radians(60), -1)
    ax.add_collection(LineCollection(segs, linewidths=lws, colors="#1b1a17", capstyle="round"))
fig.savefig(out, dpi=72, facecolor=fig.get_facecolor())
