"""Quick prototype: river trees for the top seas of a run (no typography yet)."""
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from rivers import SeaTree, bezier

d = sys.argv[1]
K = int(sys.argv[2])
out = sys.argv[3]
B = json.load(open(d + "/basins.json"))
T = pickle.load(open(d + "/tree.pkl", "rb"))
nodes = T["nodes"]

W, H = 6000, 2400
sy = 7.0                       # px per token
gap = 40
tot = sum(B[k]["count"] for k in range(K))
sx = (W - 200 - gap * (K - 1)) / tot
fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("#f3eee2")
base = 300
x0 = 100
for k in range(K):
    t = SeaTree(nodes, k)
    t.layout(x0=x0, slot=sx)
    E = t.edges()
    segs, lws = [], []
    for xu, du, xl, dl, c in E:
        pts = bezier(xu, base + sy * (du + 1), xl, base + sy * (dl + 1), n=20)
        segs.append(pts); lws.append(0.35 * np.sqrt(c))
    ax.add_collection(LineCollection(segs, linewidths=lws, colors="#1b1a17", capstyle="round", joinstyle="round"))
    ax.plot([x0, x0 + t.width], [base, base], color="#1b1a17", lw=1.5)
    x0 += t.width + gap
fig.savefig(out, dpi=72, facecolor=fig.get_facecolor())
print("saved", out)
