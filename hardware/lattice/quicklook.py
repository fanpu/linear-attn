"""quicklook.py TAG - print dispatch vocabulary, fingerprint/kernel agreement, noise; save small preview."""
import sys, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from common import *
tag = sys.argv[1]
T = load("time", tag); K = load("kernels", tag); G = T["G"]
labs = [kernel_label(s) for s in K["names"]]
kid, kv = categorize(labs, G)
print(len(kv), "kernel labels:"); [print(f"  {i:2d} {c:6d} {l}") for i, (l, c) in enumerate(kv[:30])]
fl = fp_labels(T["fp"]); fid, fv = categorize(fl, G); print(len(fv), "fingerprint classes; top counts", [c for _, c in fv[:15]])
pair = collections.Counter(zip(labs, fl))
# purity: fraction of shapes whose fingerprint class is the majority one for its kernel label, and vice versa
byk = collections.defaultdict(collections.Counter); byf = collections.defaultdict(collections.Counter)
for (l, f), c in pair.items(): byk[l][f] += c; byf[f][l] += c
S = len(labs)
print("fp classes per kernel (weighted purity)", sum(max(v.values()) for v in byk.values()) / S,
      " kernels per fp class purity", sum(max(v.values()) for v in byf.values()) / S)
for l, c in kv[:12]: print(f"  {l[:60]:60s} nfp={len(byk[l])} top={[x for _, x in byk[l].most_common(3)]}")
t = T["t_med"].reshape(G, G); lt = np.log(t); res = lt - median_filter(lt, 15, mode="nearest")
print("t us min/med/max", t.min()*1e6, np.median(t)*1e6, t.max()*1e6, "IQR/med median", np.median(T["iqr"]), "p99", np.percentile(T["iqr"], 99))
r = T["ref"]; print("ref drift %", 100*(r[:,2].max()-r[:,2].min())/np.median(r[:,2]), "std%", 100*r[:,2].std()/np.median(r[:,2]), "one", r[:,3].min()*1e6, r[:,3].max()*1e6)
print(T["info"]["smi_before"], T["info"]["smi_after"])
fig, ax = plt.subplots(1, 4, figsize=(20, 5.3))
ax[0].imshow(np.log10(gflops(T)), origin="lower", cmap="magma"); ax[0].set_title("log10 GFLOPS")
ax[1].imshow(res, origin="lower", cmap="RdBu_r", vmin=-.3, vmax=.3); ax[1].set_title("log t - median15")
ax[2].imshow(kid % 20, origin="lower", cmap="tab20", interpolation="nearest"); ax[2].set_title("kernel")
ax[3].imshow(fid % 20, origin="lower", cmap="tab20", interpolation="nearest"); ax[3].set_title("fingerprint")
fig.tight_layout(); fig.savefig(f"cache/quick_{tag}.png", dpi=70)
