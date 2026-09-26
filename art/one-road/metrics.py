"""From the distance matrices written by analyze.py: InPCA maps and the quantitative 'one road' tests."""
import argparse, json, os
from itertools import combinations
import numpy as np
from analyze import inpca

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(task, split, metric="B"):
    z = np.load(os.path.join(ROOT, "cache", f"D_{task}_{split}.npz"))
    runs = json.load(open(os.path.join(ROOT, "cache", f"runs_{task}.json")))["runs"]
    info = json.loads(str(z["info"]))
    return dict(D=z["D"] if metric == "B" else z["DH"], s=z["s"], kinds=z["kinds"], run=z["run"], k=z["k"], info=info, runs=runs)


def traj_index(L):
    """dict: ('run', ri) -> point indices in checkpoint order; ('perm', ri); ('geo',); ('mix',)."""
    T = {}
    for kind in ("run", "perm"):
        for ri in np.unique(L["run"][L["kinds"] == kind]):
            idx = np.where((L["kinds"] == kind) & (L["run"] == ri))[0]
            T[(kind, int(ri))] = idx[np.argsort(L["k"][idx])]
    P0 = np.where(L["kinds"] == "P0")[0]; Ps = np.where(L["kinds"] == "Pstar")[0]
    for kind in ("geo", "mix"):
        T[(kind,)] = np.concatenate([P0, np.where(L["kinds"] == kind)[0], Ps])
    return T


def dtraj(L, iu, iv, ngrid=60, min_range=0.05):
    """Mao et al. eq. 5: mean over a progress grid of d_B between the two trajectories' checkpoints
    nearest in progress. Progress is made monotone (running max) along each trajectory."""
    su = np.maximum.accumulate(L["s"][iu]); sv = np.maximum.accumulate(L["s"][iv])
    lo, hi = max(su[0], sv[0]), min(su[-1], sv[-1])
    if hi - lo < min_range:
        return np.nan, lo, hi
    g = np.linspace(lo, hi, ngrid)
    a = iu[np.clip(np.searchsorted(su, g), 0, len(iu) - 1)]
    b = iv[np.clip(np.searchsorted(sv, g), 0, len(iv) - 1)]
    return float(L["D"][a, b].mean()), float(lo), float(hi)


def orient(X, i0, i1):
    """Rotate the (PC1, PC2) plane so P0 -> P* points along +x; flip PC3 so P* has x3 >= 0 (declared)."""
    v = X[i1, :2] - X[i0, :2]
    th = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Y = X.copy(); Y[:, :2] = (X[:, :2] - X[i0, :2]) @ R.T
    if Y.shape[1] > 2:
        Y[:, 2] -= Y[i0, 2]
        if Y[i1, 2] < 0: Y[:, 2] *= -1
    return Y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--metric", default="B")
    args = ap.parse_args(); task, metric = args.task, args.metric
    suf = "" if metric == "B" else "_" + metric
    out = {}
    for split in ("tr", "te"):
        L = load(task, split, metric); runs = L["runs"]; kinds = L["kinds"]
        T = traj_index(L)
        i0 = int(np.where(kinds == "P0")[0][0]); i1 = int(np.where(kinds == "Pstar")[0][0])
        true_runs = [i for i, r in enumerate(runs) if r["labels"] == "true"]
        shuf_runs = [i for i, r in enumerate(runs) if r["labels"] == "shuf"]
        in_true = np.isin(L["run"], true_runs) & (kinds == "run")
        in_shuf = np.isin(L["run"], shuf_runs) & (kinds == "run")
        base = np.isin(kinds, ["P0", "Pstar", "geo", "init"]) | in_true
        maps = {"main": base, "null": base | (kinds == "perm") | (kinds == "mix"), "memo": base | in_shuf}
        res = dict(d_P0_Pstar=float(L["D"][i0, i1]))
        for mname, mask in maps.items():
            idx = np.where(mask)[0]
            X, lam, stress = inpca(L["D"][np.ix_(idx, idx)])
            X = orient(X[:, :3], int(np.searchsorted(idx, i0)), int(np.searchsorted(idx, i1)))
            np.savez(os.path.join(ROOT, "cache", f"map_{task}_{split}_{mname}{suf}.npz"), idx=idx, X=X, lam=lam[:20])
            neg = lam < 0
            res[f"stress_{mname}"] = [round(float(x), 4) for x in stress]
            res[f"lam_{mname}"] = [round(float(x), 5) for x in lam[:6]]
            res[f"neg_share_{mname}"] = float((lam[neg] ** 2).sum() / (lam ** 2).sum())

        # trajectory distances at matched progress
        pairs = {"seed": [], "opt": [], "arch": []}
        rows = []
        for a, b in combinations(true_runs, 2):
            ra, rb = runs[a], runs[b]
            cat = ("seed" if ra["opt"] == rb["opt"] else "opt") if ra["arch"] == rb["arch"] else "arch"
            dab, lo, hi = dtraj(L, T[("run", a)], T[("run", b)])
            dapb = dtraj(L, T[("run", a)], T[("perm", b)])[0]
            dbpa = dtraj(L, T[("run", b)], T[("perm", a)])[0]
            if np.isnan(dab):
                continue
            rows.append(dict(a=ra["name"], b=rb["name"], cat=cat, d=dab, d_perm=0.5 * (dapb + dbpa), lo=lo, hi=hi))
        for c in pairs:
            sel = [r for r in rows if r["cat"] == c]
            if sel:
                d = np.array([r["d"] for r in sel]); dp = np.array([r["d_perm"] for r in sel])
                res[f"dtraj_{c}"] = dict(n=len(sel), median=float(np.median(d)), mean=float(d.mean()),
                                         median_perm=float(np.median(dp)), median_ratio=float(np.median(d / dp)),
                                         frac_closer_than_perm=float((d < dp).mean()))
        res["dtraj_rows"] = rows
        # each run vs the straight roads and vs its own permutation
        per = []
        for a in true_runs:
            per.append(dict(name=runs[a]["name"], arch=runs[a]["arch"], opt=runs[a]["opt"],
                            s_end=float(np.maximum.accumulate(L["s"][T[("run", a)]])[-1]),
                            d_geo=dtraj(L, T[("run", a)], T[("geo",)])[0],
                            d_mix=dtraj(L, T[("run", a)], T[("mix",)])[0],
                            d_selfperm=dtraj(L, T[("run", a)], T[("perm", a)])[0]))
        res["per_run"] = per
        # tube: distance of each run's checkpoints to the pooled checkpoints of all *other* architectures
        tube = []
        for a in true_runs:
            others = np.where(in_true & ~np.isin(L["run"], [i for i in true_runs if runs[i]["arch"] == runs[a]["arch"]]))[0]
            ia = T[("run", a)]; ip = T[("perm", a)]
            keep = L["s"][ia] > 0.05
            if keep.sum() < 3 or len(others) == 0:
                continue
            tr = L["D"][np.ix_(ia[keep], others)].min(1).mean()
            tp = L["D"][np.ix_(ip[keep], others)].min(1).mean()
            tube.append(dict(name=runs[a]["name"], tube=float(tr), tube_perm=float(tp)))
        if tube:
            res["tube_median"] = float(np.median([t["tube"] for t in tube]))
            res["tube_perm_median"] = float(np.median([t["tube_perm"] for t in tube]))
            res["tube_ratio_median"] = float(np.median([t["tube"] / t["tube_perm"] for t in tube]))
        res["tube_rows"] = tube
        # untrained inits
        ii = np.where(kinds == "init")[0]
        arch_of = np.array([L["info"][i]["arch"] for i in ii])
        res["init_d_P0"] = {a: float(L["D"][ii[arch_of == a], i0].mean()) for a in np.unique(arch_of)}
        res["init_s"] = {a: float(L["s"][ii[arch_of == a]].mean()) for a in np.unique(arch_of)}
        Dii = L["D"][np.ix_(ii, ii)]; same = arch_of[:, None] == arch_of[None]; off = ~np.eye(len(ii), dtype=bool)
        res["init_pair_same_arch"] = float(Dii[same & off].mean()); res["init_pair_diff_arch"] = float(Dii[~same].mean())
        # memorisation: shuffled runs' distance to the true-label roads, and to the true truth
        memo = []
        for a in shuf_runs:
            ia = T[("run", a)]
            others = np.where(in_true)[0]
            memo.append(dict(name=runs[a]["name"], d_end_Pstar=float(L["D"][ia[-1], i1]),
                             d_end_P0=float(L["D"][ia[-1], i0]), s_end=float(L["s"][ia[-1]]),
                             tube_to_true=float(L["D"][np.ix_(ia, others)].min(1).mean()),
                             final_train_loss_shuf=runs[a]["final_probe_train_loss"]))
        res["memo"] = memo
        out[split] = res
        print(f"== {task} {split}: d(P0,P*)={res['d_P0_Pstar']:.3f}  stress main top1-3 {res['stress_main'][:3]}  "
              f"neg share {res['neg_share_main']:.3f}")
        for c in pairs:
            if f"dtraj_{c}" in res: print("  ", c, {k: round(v, 4) for k, v in res[f'dtraj_{c}'].items()})
        if tube: print("   tube", round(res["tube_median"], 4), "perm", round(res["tube_perm_median"], 4), "ratio", round(res["tube_ratio_median"], 3))
        print("   init d(.,P0)", {k: round(v, 3) for k, v in res["init_d_P0"].items()})
    json.dump(out, open(os.path.join(ROOT, "cache", f"metrics_{task}{suf}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
