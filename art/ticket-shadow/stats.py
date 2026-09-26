"""Null-control statistics on top of cache/analysis.npz -> cache/stats.json + printed tables.

1. Shadow vs data statistics: Pearson r with the pixel std map, mean image, P(pixel>0);
   isotonic R^2 of the shadow on std (the best any monotone function of std can do).
2. Seed reliability: mean pairwise correlation of shadows across seeds, and of the
   residuals (shadow - isotonic fit on std). A residual that repeats across seeds is structure;
   one that doesn't is noise.
3. Cross-condition similarity: shadow of each condition vs primary (seed-averaged).
4. Class maps: is the class-c map a digit? Contrast K_c = map_c/sum(map_c) - mean over classes,
   target T_c = class-c mean image - mean of class means. Report mean_c corr(K_c, T_c) and the
   number of classes whose best-matching template is their own (0..10, chance 1).
"""
import json, itertools
import numpy as np
from sklearn.isotonic import IsotonicRegression

CACHE = "/home/fzeng/ml/research/art/ticket-shadow/cache"


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def iso_resid(s, x):
    if s.std() == 0:
        return np.zeros_like(s), float("nan")
    fit = IsotonicRegression(increasing=True).fit(x, s).predict(x)
    return s - fit, 1 - ((s - fit) ** 2).sum() / ((s - s.mean()) ** 2).sum()


def class_score(maps, T, signed=False):
    """maps (10,784). Returns mean diagonal corr and #classes matched to own template."""
    if signed:
        K = maps / (np.abs(maps).sum(1, keepdims=True) + 1e-30)
    else:
        K = maps / (maps.sum(1, keepdims=True) + 1e-30)
    K = K - K.mean(0, keepdims=True)
    C = np.array([[corr(K[c], T[d]) for d in range(10)] for c in range(10)])
    if np.isnan(C).all():
        return float("nan"), 0, C
    return float(np.nanmean(np.diag(C))), int((np.nanargmax(C, 1) == np.arange(10)).sum()), C


def main():
    z = np.load(f"{CACHE}/analysis.npz")
    conds = sorted({k.split("__")[0] for k in z if "__" in k})
    out = {}
    for cond in conds:
        ds = "fashion" if cond.startswith("fashion") else "mnist"
        std, mean, nz, cm = (z[f"data_{ds}_{k}"] for k in ["std", "mean", "nz", "cmean"])
        T = cm - cm.mean(0, keepdims=True)
        sh = z[f"{cond}__shadow"]; modes = z[f"{cond}__modes"]
        R, M = sh.shape[:2]
        rows = []
        for r in range(R):
            row = dict(round=r)
            for md in sorted(set(modes)):
                idx = [m for m in range(M) if modes[m] == md]
                S = sh[r, idx]
                rs = [corr(s, std) for s in S]; rm = [corr(s, mean) for s in S]; rn = [corr(s, nz) for s in S]
                res, r2 = zip(*[iso_resid(s, std) for s in S])
                D0 = z[f"{cond}__w1_disp0"][idx]
                rdisp = [corr(s, dd) for s, dd in zip(S, D0)]
                r2disp = [iso_resid(s, dd)[1] for s, dd in zip(S, D0)]
                pair = [corr(S[a], S[b]) for a, b in itertools.combinations(range(len(S)), 2)]
                pres = [corr(res[a], res[b]) for a, b in itertools.combinations(range(len(S)), 2)]
                d = dict(r_std=np.mean(rs), r_mean=np.mean(rm), r_nz=np.mean(rn), R2_iso_std=float(np.nanmean(r2)),
                         seed_r=float(np.mean(pair)) if pair else float("nan"),
                         resid_seed_r=float(np.nanmean(pres)) if pres else float("nan"),
                         r_disp0=float(np.mean(rdisp)), R2_iso_disp0=float(np.nanmean(r2disp)))
                # class maps
                for key in ["paths", "signed", "signed0"]:
                    A = z[f"{cond}__{key}"][r, idx]
                    sc = [class_score(a, T, signed=key != "paths") for a in A]
                    d[f"{key}_diag"] = float(np.nanmean([s[0] for s in sc]))
                    d[f"{key}_own"] = float(np.mean([s[1] for s in sc]))
                d["es_test"] = float(z[f"{cond}__es_test"][r, idx].mean())
                d["es_test_sd"] = float(z[f"{cond}__es_test"][r, idx].std())
                d["final_test"] = float(z[f"{cond}__final_test"][r, idx].mean())
                d["frac_w1"] = float(z[f"{cond}__counts"][r, 0] / 235200)
                row[md] = d
            rows.append(row)
        out[cond] = rows
    # cross-condition: seed-averaged imp shadows vs primary at the same round
    prim = "mnist_adam_norm_rw0"
    if prim in conds:
        pm = z[f"{prim}__modes"]
        P = z[f"{prim}__shadow"][:, [m for m in range(len(pm)) if pm[m] == "imp"]].mean(1)
        cross = {}
        for cond in conds:
            md = z[f"{cond}__modes"]
            for mode in sorted(set(md)):
                S = z[f"{cond}__shadow"][:, [m for m in range(len(md)) if md[m] == mode]].mean(1)
                R = min(len(S), len(P))
                cross[f"{cond}/{mode}"] = [corr(S[r], P[r]) for r in range(R)]
        out["_cross_vs_primary"] = cross
    json.dump(out, open(f"{CACHE}/stats.json", "w"), indent=1, default=float)
    # print a compact table
    for cond in conds:
        rows = out[cond]
        for md in rows[0]:
            if md == "round":
                continue
            print(f"\n== {cond} / {md}")
            print(" r  w1%    es_test   r_std r_mean  r_nz  R2iso seed_r res_seed r_disp R2disp paths(d/own) signed(d/own) signed0(d/own)")
            for row in rows:
                d = row[md]
                print(f"{row['round']:2d} {100*d['frac_w1']:5.1f}  {d['es_test']:.4f}±{d['es_test_sd']:.4f} {d['r_std']:6.3f} {d['r_mean']:6.3f} {d['r_nz']:6.3f} {d['R2_iso_std']:5.2f} {d['seed_r']:6.3f} {d['resid_seed_r']:6.3f} {d['r_disp0']:6.3f} {d['R2_iso_disp0']:5.2f}"
                      f"  {d['paths_diag']:5.2f}/{d['paths_own']:4.1f}  {d['signed_diag']:5.2f}/{d['signed_own']:4.1f}  {d['signed0_diag']:5.2f}/{d['signed0_own']:4.1f}")
    if "_cross_vs_primary" in out:
        print("\ncross-condition corr with primary (seed-avg), rounds 0..:")
        for k, v in out["_cross_vs_primary"].items():
            print(f"{k:34s}", " ".join(f"{x:5.2f}" for x in v))


if __name__ == "__main__":
    main()
