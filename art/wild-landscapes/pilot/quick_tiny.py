import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files = sys.argv[2:]; out = sys.argv[1]
fig, ax = plt.subplots(2, len(files), figsize=(3.2*len(files), 6.4))
for i, f in enumerate(files):
    d = np.load(f); ax[0, i].imshow(d["loss"], origin="lower", cmap="magma"); ax[0, i].set_title(f.split("/")[-1][:-4], fontsize=7)
    u, inv = np.unique(d["fn"], return_inverse=True); ax[1, i].imshow((inv.reshape(d["fn"].shape) * 0.618) % 1, origin="lower", cmap="hsv", interpolation="nearest")
    for a in ax[:, i]: a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=80)
