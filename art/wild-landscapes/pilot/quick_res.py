import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files = sys.argv[2:]; out = sys.argv[1]
fig, ax = plt.subplots(2, len(files), figsize=(3.4*len(files), 6.8))
ax = np.array(ax).reshape(2, -1)
for i, f in enumerate(files):
    d = np.load(f); L = np.log10(d["loss"]); ax[0, i].imshow(L, origin="lower", cmap="magma"); ax[0, i].set_title(f.split("/")[-1][:-4] + f"\n{L.min():.2f}..{L.max():.1f}", fontsize=7)
    k = "agree" if "agree" in d else "acc"
    ax[1, i].imshow(d[k], origin="lower", cmap="viridis")
    for a in ax[:, i]: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=80)
