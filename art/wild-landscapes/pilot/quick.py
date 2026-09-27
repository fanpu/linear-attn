import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files = sys.argv[2:]; out = sys.argv[1]
fig, ax = plt.subplots(2, len(files), figsize=(3.2*len(files), 6.4))
ax = np.array(ax).reshape(2, -1)
for i, f in enumerate(files):
    d = np.load(f); L = np.log10(d["loss"]); ax[0, i].imshow(L, origin="lower", cmap="magma"); ax[0, i].set_title(f.split("/")[-1][:-4], fontsize=7)
    if "lyap" in d: ax[1, i].imshow(d["lyap"], origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    for a in ax[:, i]: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=80)
