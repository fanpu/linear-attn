import sys, json, pickle, collections, numpy as np
d = sys.argv[1]
z = np.load(d + "/analysis.npz"); B = json.load(open(d + "/basins.json"))
C, E, S, SEA = z["cls"], z["entry"], z["shared"], z["sea"]
m = C != 0
print("entry e: pct 10/50/90", np.percentile(E[m], [10, 50, 90]))
print("shared depth (tokens of path shared with >=1 other start): pct 10/50/90", np.percentile(S[m], [10, 50, 90]))
print("fraction of path tokens shared:", S[m].sum() / (E[m] + 1).sum())
print("starts whose path shares >=1 token beyond entry:", np.mean(S[m] >= 2), " >=5:", np.mean(S[m] >= 5), " >=20:", np.mean(S[m] >= 20))
print("private length (e+1-shared): median", np.median((E - S + 1)[m]))
cnt = np.array([b["count"] for b in B])
print("basin sizes: top", cnt[:10], " singletons", np.mean(cnt == 1), " n", len(cnt))
t = pickle.load(open(d + "/tree.pkl", "rb"))["nodes"]
kids = collections.Counter(v[2] for v in t.values())
conf = sum(1 for k, n in kids.items() if n >= 2 and not (isinstance(k, tuple)))
print("confluence nodes (>=2 children):", conf)
