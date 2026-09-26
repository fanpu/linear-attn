"""Follow-up (2026-09-26 night): do the held-out CIFAR-10 roads really fork?

5 seeds x 8 architectures, CIFAR-10, Adam lr 1e-3, batch 125, 30 epochs (the project's main CIFAR setting).
Seeds 0-1 (where they exist) come from cache/runs/cifar (original jobs); the rest from cache/runs_followup/cifar
(followup_train.py = train.py with a different output directory). SGD pilots and shuffled-label runs are excluded.

Pipeline is the project's, unchanged: analyze.bc_logsum (intensive Bhattacharyya), analyze.progress,
metrics.dtraj (Mao et al. eq. 5 at matched progress), the within-class shuffle null (one fixed within-class
permutation of the probe per run), InPCA via analyze.inpca, oriented by metrics.orient.

Quantification (both probes, so the train probe is the contrast):
  * 40x40 d_traj matrix (real) and the null matrix d0(A,B) = (d(A, perm B) + d(B, perm A)) / 2
  * pair categories: same arch (seed spread) / same valley, other arch / different valley
  * the hypothesised valleys (from the seed-0/1 plate): MLP = {logreg, MLP 1x256, 1x2048, 4x512};
    CONV-REC = {CNN, GRU}; DEEP = {ResNet-8, ViT}. (logreg is assigned where the original plate put it.)
  * average-linkage clustering of the real matrix, cut at k=3: adjusted Rand index vs the hypothesis
  * silhouette of the hypothesised partition, against all 3-way groupings of the 8 architectures of the same
    sizes (4/2/2): is this grouping the natural one?
  * robustness: matched-checkpoint distance (mean d_B at identical training step), the same over the last third
    of checkpoints ("late_third"), and d_B between final checkpoints ("endpoint"); each with its shuffle null
Writes cache/followup_fork_{tr,te}.npz (D, s, kinds, run, k) and cache/followup_fork.json.
"""
import glob, json, os, itertools
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from analyze import bc_logsum, progress, inpca, geodesic_bc_weights
from metrics import dtraj, orient

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHS = ["logreg", "mlp_256", "mlp_2048", "mlp_deep", "cnn", "gru", "resnet", "vit"]
VALLEY = {"logreg": 0, "mlp_256": 0, "mlp_2048": 0, "mlp_deep": 0, "cnn": 1, "gru": 1, "resnet": 2, "vit": 2}
VNAME = ["MLP", "CONV-REC", "DEEP"]


def load_runs():
    runs = []
    for a in ARCHS:
        for s in range(5):
            for d in ("runs", "runs_followup"):
                f = os.path.join(ROOT, "cache", d, "cifar", f"{a}_adam_true_s{s}.npz")
                if os.path.exists(f):
                    z = np.load(f); m = json.loads(str(z["meta"]))
                    assert m["epochs"] == 30 and m["opt"] == "adam" and m["labels"] == "true", f
                    runs.append(dict(meta=m, arch=a, seed=s, src=d, steps=z["steps"], lp_tr=z["lp_tr"], lp_te=z["lp_te"]))
                    break
    return runs


def ari(a, b):
    from math import comb
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    n = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua])
    sc = sum(comb(int(x), 2) for x in n.ravel()); sa = sum(comb(int(x), 2) for x in n.sum(1)); sb = sum(comb(int(x), 2) for x in n.sum(0))
    e = sa * sb / comb(len(a), 2); mx = (sa + sb) / 2
    return float((sc - e) / (mx - e)) if mx != e else 1.0


def silhouette(Dm, lab):
    lab = np.asarray(lab); s = []
    for i in range(len(lab)):
        same = (lab == lab[i]); same[i] = False
        a = Dm[i, same].mean() if same.any() else 0
        b = min(Dm[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def groupings():
    """All partitions of the 8 archs into unlabeled groups of sizes 4, 2, 2."""
    out = set()
    for four in itertools.combinations(ARCHS, 4):
        rest = [a for a in ARCHS if a not in four]
        for two in itertools.combinations(rest, 2):
            other = tuple(a for a in rest if a not in two)
            out.add(frozenset([frozenset(four), frozenset(two), frozenset(other)]))
    return [list(g) for g in out]


def build(runs, split, d, C, rng):
    y = d["ytr"][d["ptr"]] if split == "tr" else d["yte"][d["pte"]]
    N = len(y)
    pts, kinds, run, kk = [], [], [], []

    def add(sq, kind, ri=-1, k=-1):
        pts.append(sq.astype(np.float32)); kinds.append(kind); run.append(ri); kk.append(k)

    add(np.full((N, C), 1 / np.sqrt(C)), "P0"); add(np.eye(C)[y], "Pstar")
    for ri, r in enumerate(runs):
        lp = r["lp_" + split].astype(np.float64)
        for k in range(len(lp)):
            add(np.exp(lp[k] / 2), "run", ri, k)
    for ri, r in enumerate(runs):
        perm = np.arange(N)
        for c in range(C):
            idx = np.where(y == c)[0]; perm[idx] = rng.permutation(idx)
        lp = r["lp_" + split].astype(np.float64)
        for k in range(len(lp)):
            add(np.exp(lp[k][perm] / 2), "perm", ri, k)
    for al in np.linspace(0, 1, 41)[1:-1]:
        a, b = geodesic_bc_weights(al, C)
        add(a / np.sqrt(C) + b * np.eye(C)[y], "geo")
    S = np.stack(pts)
    lg, _ = bc_logsum(torch.from_numpy(S))
    D = -lg.numpy() / N
    D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0); D = np.maximum(D, 0)
    s = progress(S.astype(np.float64), y, C)
    return dict(D=D, s=s, kinds=np.array(kinds), run=np.array(run), k=np.array(kk))


def summarise(Dm, archs, valleys):
    n = len(archs); cats = {"seed": [], "valley": [], "between": []}
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(Dm[i, j]):
                continue
            c = "seed" if archs[i] == archs[j] else ("valley" if valleys[i] == valleys[j] else "between")
            cats[c].append(Dm[i, j])
    return {c: dict(n=len(v), median=float(np.median(v)), q10=float(np.quantile(v, .1)), q90=float(np.quantile(v, .9)),
                    min=float(np.min(v)), max=float(np.max(v))) for c, v in cats.items() if v}


def main():
    torch.set_num_threads(16)
    d = np.load(os.path.join(ROOT, "cache", "data_cifar.npz")); C = int(d["C"])
    runs = load_runs()
    archs = [r["arch"] for r in runs]; valleys = [VALLEY[a] for a in archs]
    print(len(runs), "runs:", {a: sum(1 for r in runs if r["arch"] == a) for a in ARCHS})
    rng = np.random.default_rng(0)
    out = dict(runs=[dict(arch=r["arch"], seed=r["seed"], src=r["src"], test_acc=r["meta"]["full_test_acc"]) for r in runs])
    G = groupings()
    for split in ("tr", "te"):
        L = build(runs, split, d, C, rng)
        np.savez(os.path.join(ROOT, "cache", f"followup_fork_{split}.npz"), **L)
        T = {}
        for kind in ("run", "perm"):
            for ri in range(len(runs)):
                idx = np.where((L["kinds"] == kind) & (L["run"] == ri))[0]
                T[(kind, ri)] = idx[np.argsort(L["k"][idx])]
        n = len(runs)
        R = np.full((n, n), np.nan); N0 = np.full((n, n), np.nan); K = np.zeros((n, n)); K0 = np.zeros((n, n))
        LT = np.zeros((n, n)); LT0 = np.zeros((n, n)); FE = np.zeros((n, n)); FE0 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                R[i, j] = R[j, i] = dtraj(L, T[("run", i)], T[("run", j)])[0]
                N0[i, j] = N0[j, i] = 0.5 * (dtraj(L, T[("run", i)], T[("perm", j)])[0] + dtraj(L, T[("run", j)], T[("perm", i)])[0])
                a, b = T[("run", i)], T[("run", j)]; m = min(len(a), len(b))
                K[i, j] = K[j, i] = L["D"][a[1:m], b[1:m]].mean()          # matched training step (skip step 0)
                pa, pb = T[("perm", i)], T[("perm", j)]
                K0[i, j] = K0[j, i] = 0.5 * (L["D"][a[1:m], pb[1:m]].mean() + L["D"][b[1:m], pa[1:m]].mean())
                h = (2 * m) // 3                                                  # late: last third of checkpoints (steps ~> 300)
                LT[i, j] = LT[j, i] = L["D"][a[h:m], b[h:m]].mean()
                LT0[i, j] = LT0[j, i] = 0.5 * (L["D"][a[h:m], pb[h:m]].mean() + L["D"][b[h:m], pa[h:m]].mean())
                FE[i, j] = FE[j, i] = L["D"][a[m - 1], b[m - 1]]                  # where the roads end
                FE0[i, j] = FE0[j, i] = 0.5 * (L["D"][a[m - 1], pb[m - 1]] + L["D"][b[m - 1], pa[m - 1]])
        np.fill_diagonal(R, 0); np.fill_diagonal(N0, 0)
        res = {}
        for name, Dm in (("dtraj", R), ("dtraj_null", N0), ("matched_step", K), ("matched_step_null", K0),
                         ("late_third", LT), ("late_third_null", LT0), ("endpoint", FE), ("endpoint_null", FE0)):
            Dz = np.nan_to_num(Dm, nan=np.nanmax(Dm))
            Z = linkage(squareform(Dz, checks=False), "average")
            cut = fcluster(Z, 3, "maxclust")
            cut8 = fcluster(Z, 8, "maxclust")
            sil_h = silhouette(Dz, valleys)
            sils = []
            for g in G:
                lab = [next(k for k, grp in enumerate(g) if a in grp) for a in archs]
                sils.append((silhouette(Dz, lab), [sorted(x) for x in g]))
            sils.sort(key=lambda t: -t[0])
            rank = 1 + sum(1 for sv, _ in sils if sv > sil_h + 1e-12)
            cats = summarise(Dm, archs, valleys)
            res[name] = dict(cats=cats,
                             ratio_between_over_seed=cats["between"]["median"] / cats["seed"]["median"],
                             ratio_between_over_valley=cats["between"]["median"] / cats["valley"]["median"],
                             ratio_valley_over_seed=cats["valley"]["median"] / cats["seed"]["median"],
                             ari_k3_vs_valleys=ari(cut, valleys), ari_k8_vs_arch=ari(cut8, archs),
                             k3_clusters={int(c): sorted(set(f"{archs[i]}" for i in range(n) if cut[i] == c)) for c in np.unique(cut)},
                             k3_members={int(c): [f"{archs[i]}_s{runs[i]['seed']}" for i in range(n) if cut[i] == c] for c in np.unique(cut)},
                             merge_heights_last5=[float(x) for x in Z[-5:, 2]],
                             silhouette_valleys=sil_h, silhouette_rank=rank, n_groupings=len(G),
                             best_grouping=sils[0][1], best_silhouette=sils[0][0])
            # separation margin: smallest between-valley distance vs largest seed-seed distance
            print(f"[{split}] {name:18s} seed {cats['seed']['median']:.3f} valley {cats['valley']['median']:.3f} "
                  f"between {cats['between']['median']:.3f}  B/seed {res[name]['ratio_between_over_seed']:.2f} "
                  f"B/valley {res[name]['ratio_between_over_valley']:.2f}  ARI3 {res[name]['ari_k3_vs_valleys']:.2f} "
                  f"ARI8 {res[name]['ari_k8_vs_arch']:.2f} sil {sil_h:.3f} rank {rank}/{len(G)}  k3 {res[name]['k3_clusters']}")
        # real vs null per category, paired
        for c in ("seed", "valley", "between"):
            sel = [(R[i, j], N0[i, j]) for i in range(n) for j in range(i + 1, n)
                   if (("seed" if archs[i] == archs[j] else ("valley" if valleys[i] == valleys[j] else "between")) == c)
                   and not np.isnan(R[i, j])]
            a = np.array(sel)
            res[f"real_vs_null_{c}"] = dict(median_ratio=float(np.median(a[:, 0] / a[:, 1])), frac_closer=float((a[:, 0] < a[:, 1]).mean()))
            print(f"   {c:8s} real/null median ratio {res[f'real_vs_null_{c}']['median_ratio']:.2f}, closer than null {res[f'real_vs_null_{c}']['frac_closer']:.2f}")
        # s range per arch (held-out progress stalls)
        res["s_end"] = {a: float(np.mean([np.maximum.accumulate(L["s"][T[("run", i)]])[-1] for i in range(n) if archs[i] == a])) for a in ARCHS}
        out[split] = res
        np.savez(os.path.join(ROOT, "cache", f"followup_fork_mats_{split}.npz"), R=R, N0=N0, K=K, K0=K0, LT=LT, LT0=LT0, FE=FE, FE0=FE0,
                 archs=np.array(archs), seeds=np.array([r["seed"] for r in runs]))
        # InPCA map for drawing: landmarks, straight road, real runs
        mask = np.isin(L["kinds"], ["P0", "Pstar", "geo", "run"])
        idx = np.where(mask)[0]
        X, lam, stress = inpca(L["D"][np.ix_(idx, idx)])
        i0 = int(np.where(L["kinds"][idx] == "P0")[0][0]); i1 = int(np.where(L["kinds"][idx] == "Pstar")[0][0])
        X = orient(X[:, :3], i0, i1)
        np.savez(os.path.join(ROOT, "cache", f"followup_fork_map_{split}.npz"), idx=idx, X=X, lam=lam[:20])
        out[split]["stress"] = [float(x) for x in stress[:5]]
        print(f"[{split}] InPCA stress top1-3 {stress[:3]}")
    json.dump(out, open(os.path.join(ROOT, "cache", "followup_fork.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
