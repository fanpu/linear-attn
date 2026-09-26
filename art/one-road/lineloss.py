"""Goodfellow, Vinyals & Saxe (2015): loss along the straight line from init to the trained solution.

theta(alpha) = (1 - alpha) theta_init + alpha theta_final, alpha in [-0.25, 1.25], evaluated on the
train and test probe sets (CPU). BatchNorm running statistics are interpolated like every other
tensor (declared: the line is taken over the whole state_dict)."""
import glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from models import build

ROOT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(16)
ALPHA = np.linspace(-0.25, 1.25, 61)


@torch.no_grad()
def main():
    out = {}
    for task in ("mnist", "fashion", "cifar", "modadd"):
        dp = os.path.join(ROOT, "cache", f"data_{task}.npz")
        if not os.path.exists(dp):
            continue
        d = np.load(dp); C = int(d["C"]); img = task != "modadd"
        X = {}
        for sp, key, pk, yk in (("tr", "xtr", "ptr", "ytr"), ("te", "xte", "pte", "yte")):
            x = d[key][d[pk]]
            X[sp] = (torch.tensor(x, dtype=torch.float32) if img else torch.tensor(x), torch.tensor(d[yk][d[pk]]))
        for f in sorted(glob.glob(os.path.join(ROOT, "cache", "weights", task, "*.pt"))):
            name = os.path.basename(f)[:-3]
            if "_shuf_" in name:
                continue
            w = torch.load(f)
            arch = name.split("_adam")[0].split("_sgd")[0]
            m = build(task, arch, C, tuple(X["tr"][0].shape[1:]) if img else None).eval()
            res = {"tr": [], "te": []}
            for al in ALPHA:
                sd = {k: ((1 - al) * w["init"][k].double() + al * w["final"][k].double()).to(w["final"][k].dtype)
                      if w["final"][k].is_floating_point() else w["final"][k] for k in w["final"]}
                m.load_state_dict(sd)
                for sp in ("tr", "te"):
                    x, y = X[sp]
                    res[sp].append(float(F.cross_entropy(m(x).double(), y)))
            out[f"{task}/{name}"] = res
            print(task, name, round(res["tr"][20], 3), round(res["tr"][-21], 4))
    json.dump(dict(alpha=ALPHA.tolist(), runs=out), open(os.path.join(ROOT, "cache", "lineloss.json"), "w"))


if __name__ == "__main__":
    main()
