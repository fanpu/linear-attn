"""Shadows, class path counts, and null-control statistics from the IMP runs (CPU).

For every condition / round / model:
  shadow[i]        = number of surviving first-layer weights leaving input pixel i (0..300)
  paths[c, i]      = number of surviving input->class paths = (M1 M2 M3)[i, c]   (exact, float64)
  signed[c, i]     = sum over surviving paths of the product of the *trained* weights
                   = (W1*M1)(W2*M2)(W3*M3)[i, c]  -- the network with every ReLU switched on
  signed0[c, i]    = the same with the *rewound* (init) weights: the ticket as it is before training
Permuted runs are mapped back to image coordinates.

Writes cache/analysis.npz and cache/metrics.json.
"""
import os, json, glob
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from imp import load_dataset, SIZES

ROOT = "/home/fzeng/ml/research/art/ticket-shadow"
CACHE = f"{ROOT}/cache"


def unpack(z, l):
    return np.unpackbits(z[f"mask{l}"], axis=-1)[..., :SIZES[l][1]].astype(bool)


def load_cond(cond):
    files = sorted(glob.glob(f"{CACHE}/{cond}/round_*.npz"))
    if not files:
        return None
    z0 = np.load(files[0])
    modes = z0["modes"]; seeds = z0["seeds"]; perms = z0["perms"]
    init = [z0[f"init_W{l}"] for l in range(3)]
    rew = [z0[f"rewind_W{l}"] for l in range(3)] if "rewind_W0" in z0 else init
    R, M = len(files), len(modes)
    out = dict(modes=modes, seeds=seeds, R=R,
               shadow=np.zeros((R, M, 784)), paths=np.zeros((R, M, 10, 784)),
               signed=np.zeros((R, M, 10, 784)), signed0=np.zeros((R, M, 10, 784)),
               counts=np.zeros((R, 3), int), es_test=np.zeros((R, M)), final_test=np.zeros((R, M)),
               es_iter=np.zeros((R, M)), w1_final_abs=np.zeros((R, M, 784)), w1_init_abs=np.zeros((M, 784)), w1_disp0=np.zeros((M, 784)))
    for r, f in enumerate(files):
        z = np.load(f)
        mk = [unpack(z, l) for l in range(3)]
        W = [z[f"W{l}"].astype(np.float64) for l in range(3)]
        out["counts"][r] = z["counts"]
        out["es_test"][r] = z["es_test_acc"]; out["final_test"][r] = z["final_test_acc"]; out["es_iter"][r] = z["es_iter"]
        for m in range(M):
            inv = np.argsort(perms[m])  # input i of the permuted net is image pixel perms[m][i]
            P = mk[0][m].astype(np.float64) @ mk[1][m].astype(np.float64) @ mk[2][m].astype(np.float64)
            S = (W[0][m] * mk[0][m]) @ (W[1][m] * mk[1][m]) @ (W[2][m] * mk[2][m])
            base = rew if modes[m] != "reinit" else None
            if base is not None:
                S0 = (base[0][m] * mk[0][m]) @ (base[1][m] * mk[1][m]) @ (base[2][m] * mk[2][m])
            else:
                S0 = np.full_like(S, np.nan)
            sh = mk[0][m].sum(1)
            wabs = (np.abs(W[0][m]) * mk[0][m]).sum(1) / np.maximum(sh, 1)
            # map back to image coordinates: img[perm[i]] = net[i]
            def unperm(v):
                o = np.empty_like(v); o[..., perms[m]] = v; return o
            out["shadow"][r, m] = unperm(sh)
            out["paths"][r, m] = unperm(P.T)
            out["signed"][r, m] = unperm(S.T)
            out["signed0"][r, m] = unperm(S0.T)
            out["w1_final_abs"][r, m] = unperm(wabs)
            if r == 0:
                out["w1_init_abs"][m] = unperm(np.abs(init[0][m]).mean(1))
                out["w1_disp0"][m] = unperm(np.abs(W[0][m] - init[0][m]).mean(1))
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def main():
    stats = {}
    for ds in ["mnist", "fashion"]:
        xtr, ytr, _, _ = load_dataset(ds)
        xtr, ytr = xtr[:55000], ytr[:55000]
        stats[ds] = dict(mean=xtr.mean(0), std=xtr.std(0), nz=(xtr > 0).mean(0),
                         cmean=np.stack([xtr[ytr == c].mean(0) for c in range(10)]))
    conds = [os.path.basename(d) for d in sorted(glob.glob(f"{CACHE}/*")) if os.path.isdir(d) and not os.path.basename(d).startswith("_")]
    res, metrics = {}, {}
    for cond in conds:
        o = load_cond(cond)
        if o is None:
            continue
        ds = "fashion" if cond.startswith("fashion") else "mnist"
        st = stats[ds]
        res[cond] = o
        mt = dict(modes=o["modes"].tolist(), seeds=o["seeds"].tolist(), rounds=o["R"],
                  remaining_frac=[float(c[0]) / 235200 for c in o["counts"]],
                  total_remaining_frac=[float(sum(c)) / 266200 for c in o["counts"]],
                  es_test=o["es_test"].tolist(), final_test=o["final_test"].tolist(), es_iter=o["es_iter"].tolist())
        per = []
        for r in range(o["R"]):
            row = []
            for m in range(len(o["modes"])):
                s = o["shadow"][r, m]
                d = dict(r_std=corr(s, st["std"]), r_mean=corr(s, st["mean"]), r_nz=corr(s, st["nz"]),
                         rho_std=float(spearmanr(s, st["std"])[0]) if s.std() > 0 else float("nan"))
                if s.std() > 0:
                    iso = IsotonicRegression(increasing="auto").fit(st["std"], s)
                    fit = iso.predict(st["std"])
                    d["R2_iso_std"] = 1 - float(((s - fit) ** 2).sum() / ((s - s.mean()) ** 2).sum())
                    iso2 = IsotonicRegression(increasing="auto").fit(st["nz"], s)
                    d["R2_iso_nz"] = 1 - float(((s - iso2.predict(st["nz"])) ** 2).sum() / ((s - s.mean()) ** 2).sum())
                row.append(d)
            per.append(row)
        mt["shadow_vs_data"] = per
        metrics[cond] = mt
        print(cond, "rounds", o["R"], "r_std(last)", np.round([d.get("r_std") for d in per[-1]], 3))
    np.savez_compressed(f"{CACHE}/analysis.npz",
                        **{f"{c}__{k}": v for c, o in res.items() for k, v in o.items() if isinstance(v, np.ndarray)},
                        **{f"data_{ds}_{k}": v for ds, s in stats.items() for k, v in s.items()})
    json.dump(metrics, open(f"{CACHE}/metrics.json", "w"))


if __name__ == "__main__":
    main()
