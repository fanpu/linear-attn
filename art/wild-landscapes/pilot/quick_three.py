import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files = sys.argv[2:]; out = sys.argv[1]
fig, ax = plt.subplots(3, len(files), figsize=(3.6*len(files), 10.5))
for i, f in enumerate(files):
    d = np.load(f); L = np.log10(d["loss"]); ax[0, i].imshow(L, origin="lower", cmap="magma"); ax[0,i].set_title(f"{L.min():.2f}..{L.max():.2f}")
    for r, pre in ((1, "ag"), (2, "dg")):
        G = np.stack([d[pre+str(k)] for k in range(3)], -1)
        G = (G - G.min()) / (G.max() - G.min()); ax[r, i].imshow(G, origin="lower")
    for a in ax[:, i]: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=70)
