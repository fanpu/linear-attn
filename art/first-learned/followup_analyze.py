"""FOLLOW-UP (2026-09-26 night): lanes or learning speed? (CPU only)

Recomputes first-learned's agreement statistics for two run sets
  before : linear, MLP, CNN, ResNet-8 (BN batch stats), ViT  -- all SGD 0.02, 8 epochs (cache/runs/)
  after  : linear, MLP, CNN reused; ResNet-8 and ViT retuned (cache/runs_followup/)
on two clocks
  step     : the original key (stable step on the declared eval grid, ties broken by mean correctness)
  progress : speed-normalised. Each run's own training progress p(t) = cummax train acc(t) / final train acc.
             Its correctness matrix is resampled at the first evaluation where p reaches each of a fixed
             ladder of levels (error-to-go 1-p geometric from 0.9 to 0.002, 60 levels, plus the final eval),
             so every run is read on the same progress grid with the same resolution. key = (stable level,
             -mean correctness over the ladder). Note: a per-run monotone relabelling of time alone cannot
             change a Spearman; what the ladder changes is the resolution/weighting (and hence ties and
             tie-breaks), which is exactly where a fast learner's order is coarsest.
Statistics: arch x arch mean Spearman (diag = seed pairs), within-class Spearman, class-only null,
last-learned-1% overlap. Writes cache/followup/agreement_followup.json.
"""
import glob, os, json
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(ROOT, "cache")
LADDER = np.geomspace(0.9, 0.002, 60)  # error-to-go levels (relative to own final acc)

BEFORE = {"logreg": "runs/mnist_logreg_true_s{}", "mlp_256": "runs/mnist_mlp_256_true_s{}", "cnn": "runs/mnist_cnn_true_s{}",
          "resnetbnb": "runs/mnist_resnetbnb_true_s{}", "vit": "runs/mnist_vit_true_s{}"}
AFTER = dict(BEFORE, resnetbnb="runs_followup/mnist_resnetbnb-sgd0.1cos8_true_s{}",
             vit="runs_followup/mnist_vit-adam1e-3cos10_true_s{}")


def keys(f):
    d = np.load(os.path.join(C, f + ".npz")); steps, pred, y = d["steps"], d["pred"], d["y"]
    cor = pred == y[None]; acc = cor.mean(1)
    def key_of(cor, t):
        K = len(cor); inc = ~cor
        last_inc = np.where(inc.any(0), K - 1 - np.argmax(inc[::-1], 0), -1)
        stable = np.where(last_inc == K - 1, np.inf, t[np.minimum(last_inc + 1, K - 1)])
        return rankdata(np.lexsort((-cor.mean(0), stable)).argsort()), stable
    k_step, stable = key_of(cor, steps)
    p = np.maximum.accumulate(acc) / acc[-1]
    ix = np.array([np.argmax(p >= 1 - e) for e in LADDER] + [len(p) - 1])
    k_prog, _ = key_of(cor[ix], np.arange(len(ix)).astype(float))
    fin = np.isfinite(stable)
    info = dict(final_acc=float(acc[-1]), test_acc=float(d["test_acc"]), T=int(steps[-1]),
                med_stable=float(np.median(stable[fin])),
                step_995=int(steps[np.argmax(acc >= 0.995)]) if (acc >= 0.995).any() else None,
                step_99=int(steps[np.argmax(acc >= 0.99)]) if (acc >= 0.99).any() else None)
    return k_step, k_prog, y, info


def wc(a, b, y):
    return float(np.mean([spearmanr(a[y == c], b[y == c])[0] for c in range(10)]))


def table(K, y, archs):
    runs = [(a, s) for a in archs for s in range(3)]
    q = int(0.01 * len(y)); last = {r: set(np.argsort(K[r])[-q:]) for r in runs}
    cm = {r: np.array([K[r][y == c].mean() for c in range(10)])[y] for r in runs}
    n = len(archs); AA, AW, AL, AN = (np.zeros((n, n)) for _ in range(4))
    for i, a in enumerate(archs):
        for j, b in enumerate(archs):
            pr = [(r1, r2) for r1 in runs for r2 in runs if r1[0] == a and r2[0] == b and r1 != r2 and (i != j or r1[1] < r2[1])]
            AA[i, j] = np.mean([spearmanr(K[r1], K[r2])[0] for r1, r2 in pr])
            AW[i, j] = np.mean([wc(K[r1], K[r2], y) for r1, r2 in pr])
            AL[i, j] = np.mean([len(last[r1] & last[r2]) / q for r1, r2 in pr])
            AN[i, j] = np.mean([spearmanr(K[r1], cm[r2])[0] for r1, r2 in pr])
    return dict(AA=AA.tolist(), AW=AW.tolist(), AL=AL.tolist(), AN=AN.tolist())


def summary(T, archs):
    g = [archs.index(a) for a in ["logreg", "mlp_256", "cnn"]]; o = [archs.index(a) for a in ["resnetbnb", "vit"]]
    out = {}
    for m in ["AA", "AW", "AL", "AN"]:
        M = np.array(T[m])
        out[m] = dict(seeds=float(np.mean(np.diag(M))),
                      group_cross=float(np.mean([M[i, j] for i in g for j in g if i < j])),
                      lanes_to_group=float(np.mean([M[i, j] for i in o for j in g])),
                      resnet_to_group=float(np.mean([M[o[0], j] for j in g])),
                      vit_to_group=float(np.mean([M[o[1], j] for j in g])),
                      resnet_vit=float(M[o[0], o[1]]),
                      seeds_resnet=float(M[o[0], o[0]]), seeds_vit=float(M[o[1], o[1]]))
    return out


if __name__ == "__main__":
    archs = ["logreg", "mlp_256", "cnn", "vit", "resnetbnb"]
    res = {"archs": archs, "ladder_error_to_go": LADDER.tolist()}
    for setname, S in [("before", BEFORE), ("after", AFTER)]:
        Ks, Kp, info = {}, {}, {}
        for a in archs:
            for s in range(3):
                ks, kp, y, inf = keys(S[a].format(s)); Ks[(a, s)] = ks; Kp[(a, s)] = kp; info[f"{a}_s{s}"] = inf
        res[setname] = dict(info=info)
        for clock, K in [("step", Ks), ("progress", Kp)]:
            T = table(K, y, archs); T["summary"] = summary(T, archs); res[setname][clock] = T
            print(f"== {setname} / {clock} clock")
            for m in ["AA", "AW", "AL"]:
                print(" ", m, {k: round(v, 3) for k, v in T["summary"][m].items()})
            print("  class-only null (cross-arch mean):", round(float(np.mean([np.array(T['AN'])[i, j] for i in range(5) for j in range(5) if i != j])), 3))
            for i, a in enumerate(archs):
                print(f"   {a:9s}", " ".join(f"{T['AA'][i][j]:.2f}/{T['AW'][i][j]:.2f}/{T['AL'][i][j]:.2f}" for j in range(5)))
        for k, v in info.items():
            print("  ", k, v)
    json.dump(res, open(os.path.join(C, "followup", "agreement_followup.json"), "w"), indent=1)
