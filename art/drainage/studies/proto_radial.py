import json, pickle, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.collections import LineCollection
from rivers import SeaTree
d, sea, out, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
B = json.load(open(d + "/basins.json")); nodes = pickle.load(open(d + "/tree.pkl", "rb"))["nodes"]
S = 2400; R0 = 140; dr = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
t = SeaTree(nodes, sea); t.layout(0, 2 * np.pi / B[sea]["count"])
def rad(dep):
    dep = np.asarray(dep, float) + 1
    return R0 + (dr * dep if mode == "lin" else dr * 12 * np.sqrt(dep))
def P(th, r): return np.stack([S/2 + r*np.cos(th), S/2 + r*np.sin(th)], -1)
tt = np.linspace(0, 1, 24)[:, None]
segs, lws = [], []
for xu, du, xl, dl, c in t.edges():
    ru, rl = rad(du), rad(dl)
    tq = np.linspace(0, 1, 40)
    sm = tq * tq * (3 - 2 * tq)           # angle eases from child to parent
    r = ru + (rl - ru) * tq
    th = xu + (xl - xu) * sm
    pts = P(th, r)
    segs.append(pts); lws.append(0.45*np.sqrt(c))
fig = plt.figure(figsize=(S/72, S/72), dpi=72); ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,S); ax.set_ylim(0,S); ax.axis("off")
fig.patch.set_facecolor("#f3eee2")
ax.add_collection(LineCollection(segs, linewidths=lws, colors="#1b1a17", capstyle="round"))
ax.add_patch(plt.Circle((S/2, S/2), R0, fill=False, lw=2, color="#1b1a17"))
fig.savefig(out, dpi=72, facecolor=fig.get_facecolor())
