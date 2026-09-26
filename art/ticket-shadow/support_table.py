"""How many connections does a pixel keep, as a function of how many training images ever light it?
Prints a table (seed-mean shadow per activity bin) and corr with the binary support map (pixel ever > 0)."""
import json, numpy as np
from imp import load_dataset
xtr, _, _, _ = load_dataset("mnist"); cnt = (xtr[:55000] > 0).sum(0)
z = np.load("cache/analysis.npz")
bins = [0, 1, 10, 100, 1000, 5000, 20000, 60000]
out = {}
print("condition            round  " + "  ".join(f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)) + "   r(support) r(std)")
for c in ["mnist_adam_norm_rw0", "mnist_adam_raw_rw0", "mnist_sgd_norm_rw0", "mnist_sgd_raw_rw0"]:
    md = z[c + "__modes"]; idx = [m for m in range(len(md)) if md[m] == "imp"]
    for r in [5, 10, 15, 19]:
        s = z[c + "__shadow"][r, idx].mean(0)
        row = [float(s[(cnt >= bins[i]) & (cnt < bins[i + 1])].mean()) for i in range(len(bins) - 1)]
        rs = float(np.corrcoef(s, cnt > 0)[0, 1]); rstd = float(np.corrcoef(s, z["data_mnist_std"])[0, 1])
        out[f"{c}/r{r}"] = dict(bins=row, r_support=rs, r_std=rstd)
        print(f"{c:22s} r{r:02d}   " + "  ".join(f"{v:8.1f}" for v in row) + f"   {rs:6.2f}  {rstd:6.2f}")
print("pixels per bin:", [int(((cnt >= bins[i]) & (cnt < bins[i + 1])).sum()) for i in range(len(bins) - 1)])
json.dump(out, open("cache/support_table.json", "w"), indent=1)
