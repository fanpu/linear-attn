"""Per-example learning statistics and agreement between runs (CPU only).

For each run (cache/runs/*.npz) and each of the 60,000 training examples:
  first   : first evaluation step at which the example is classified correctly
  stable  : earliest evaluation step from which it stays correct at every later evaluation
            (never correct at the final evaluation -> +inf, 'never learned')
  forget  : number of correct -> incorrect transitions between consecutive evaluations
            (full-pass evaluations on the declared grid, NOT Toneva's per-presentation count)
  speed   : mean correctness over the evaluation grid (a learning-speed score, Jiang et al. 2021
            proxy); used only to break ties in 'stable'
  key     : the ordering key = rank of (stable, -speed)  -- lower = learned earlier
Writes cache/stats.npz and prints the agreement tables.
"""
import glob, os, json
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(ROOT, "cache")


def per_run(f):
    d = np.load(f)
    steps, pred, y = d["steps"], d["pred"], d["y"]
    cor = pred == y[None]
    K, N = cor.shape
    first = np.where(cor.any(0), steps[np.argmax(cor, 0)], np.inf)
    # last incorrect index; stable = step after it
    inc = ~cor
    last_inc = np.where(inc.any(0), K - 1 - np.argmax(inc[::-1], 0), -1)
    stable = np.where(last_inc == K - 1, np.inf, steps[np.minimum(last_inc + 1, K - 1)])
    forget = (cor[:-1] & ~cor[1:]).sum(0)
    late = steps[1:] > 468  # after the first epoch: excludes the volatile first-pass flicker
    forget_late = (cor[:-1] & ~cor[1:])[late].sum(0)
    speed = cor.mean(0)
    key = rankdata(np.lexsort((-speed, stable)).argsort())  # reading-order rank
    return dict(forget_late=forget_late, first=first, stable=stable, forget=forget, speed=speed, key=key,
                final_pred=pred[-1], y=y, test_acc=float(d["test_acc"]), final_acc=cor[-1].mean())


def within_class_spearman(a, b, y):
    return np.mean([spearmanr(a[y == c], b[y == c])[0] for c in range(10)])


def main():
    files = sorted(glob.glob(os.path.join(C, "runs", "mnist_*.npz")))
    R = {os.path.basename(f)[6:-4]: per_run(f) for f in files}
    names = sorted(R)
    y = R[names[0]]["y"] if "true" in names[0] else None
    ytrue = next(R[n]["y"] for n in names if "_true_" in n)
    for n in names:
        r = R[n]
        print(f"{n:24s} final train acc {r['final_acc']:.4f} test {r['test_acc']:.4f} "
              f"never-learned {np.isinf(r['stable']).sum():6d}  median stable step {np.median(r['stable'][np.isfinite(r['stable'])]):.0f}  "
              f"unforgettable {np.mean(r['forget'] == 0):.3f} (after ep1 {np.mean(r['forget_late'] == 0):.3f})  mean forget {r['forget'].mean():.2f}")
    true = [n for n in names if "_true_" in n]
    arch = lambda n: n.rsplit("_true_", 1)[0] if "_true_" in n else n
    # pairwise agreement
    M = np.zeros((len(true), len(true))); W = np.zeros_like(M); Cn = np.zeros_like(M); J = np.zeros_like(M)
    classmean = {n: np.array([R[n]["key"][ytrue == c].mean() for c in range(10)])[ytrue] for n in true}
    q = int(0.01 * len(ytrue))
    last = {n: set(np.argsort(R[n]["key"])[-q:]) for n in true}
    for i, a in enumerate(true):
        for j, b in enumerate(true):
            M[i, j] = spearmanr(R[a]["key"], R[b]["key"])[0]
            W[i, j] = within_class_spearman(R[a]["key"], R[b]["key"], ytrue)
            Cn[i, j] = spearmanr(R[a]["key"], classmean[b])[0]
            J[i, j] = len(last[a] & last[b]) / q
    same = np.array([[arch(a) == arch(b) and a != b for b in true] for a in true])
    diff = np.array([[arch(a) != arch(b) for b in true] for a in true])
    out = {}
    for lab, X in [("spearman", M), ("within_class", W), ("class_only_null", Cn), ("last1pct_overlap", J)]:
        out[lab] = dict(same_arch=float(X[same].mean()), cross_arch=float(X[diff].mean()))
        print(f"{lab:18s} same-arch(seeds) {X[same].mean():.3f}   cross-arch {X[diff].mean():.3f}")
    archs = sorted(set(arch(n) for n in true))
    print("arch x arch mean spearman (off-diagonal seeds for diagonal) / within-class:")
    AA = np.zeros((len(archs), len(archs))); AW = np.zeros_like(AA)
    for i, a in enumerate(archs):
        for j, b in enumerate(archs):
            mk = np.array([[arch(p) == a and arch(q_) == b and p != q_ for q_ in true] for p in true])
            AA[i, j] = M[mk].mean(); AW[i, j] = W[mk].mean()
        print(f"  {a:8s} " + " ".join(f"{AA[i,j]:.3f}/{AW[i,j]:.3f}" for j in range(len(archs))))
    out["archs"] = archs; out["AA"] = AA.tolist(); out["AW"] = AW.tolist()
    # shuffled-label (memorisation) null
    for s in [n for n in names if "_shuf_" in n]:
        ys = R[s]["y"]
        rs = [spearmanr(R[s]["key"], R[t]["key"])[0] for t in true]
        print(f"null {s}: spearman with true-label runs {np.mean(rs):.3f} (range {min(rs):.3f}..{max(rs):.3f}); "
              f"memorised by end {R[s]['final_acc']:.3f}")
        out[f"null_{s}"] = float(np.mean(rs))
    # consensus orders
    net = [n for n in true if not n.startswith("logreg")]  # consensus = the neural networks; logreg is a baseline
    cons = np.mean([R[n]["key"] for n in net], 0)
    per_arch = {a: np.mean([R[n]["key"] for n in true if arch(n) == a], 0) for a in archs}
    forget_tot = np.sum([R[n]["forget"] for n in true], 0)
    never_frac = np.mean([np.isinf(R[n]["stable"]) for n in true], 0)
    wrong_same = np.mean([R[n]["final_pred"] != ytrue for n in true], 0)
    # consensus predicted label at the end (mode across runs)
    fp = np.stack([R[n]["final_pred"] for n in net])
    mode = np.array([np.bincount(fp[:, i], minlength=10).argmax() for i in range(fp.shape[1])])
    np.savez(os.path.join(C, "stats.npz"), cons=cons, forget_tot=forget_tot, never_frac=never_frac,
             wrong_frac=wrong_same, mode=mode, y=ytrue, true=np.array(true),
             keys=np.stack([R[n]["key"] for n in true]), forgets=np.stack([R[n]["forget"] for n in true]), forgets_late=np.stack([R[n]["forget_late"] for n in true]),
             stables=np.stack([R[n]["stable"] for n in true]), firsts=np.stack([R[n]["first"] for n in true]),
             **{f"arch_{a}": v for a, v in per_arch.items()},
             **{f"shufkey_{s}": R[s]["key"] for s in names if "_shuf_" in s})
    # agreement of per-arch consensus orders (3-seed means)
    print("per-arch 3-seed consensus spearman:")
    for a in archs:
        print("  ", a, " ".join(f"{spearmanr(per_arch[a], per_arch[b])[0]:.3f}" for b in archs))
    # trivial-feature baselines (no network): pixel distance to own-class mean image, and ink mass
    import struct
    with open(os.path.join(ROOT, "..", "data", "MNIST", "raw", "train-images-idx3-ubyte"), "rb") as f:
        f.read(16); X = np.frombuffer(f.read(), np.uint8).reshape(-1, 784).astype(np.float32) / 255
    cen = np.stack([X[ytrue == c].mean(0) for c in range(10)])
    dist = np.linalg.norm(X - cen[ytrue], axis=1); ink = X.sum(1)
    # margin to nearest other centroid (nearest-centroid classifier's own 'difficulty')
    D = np.stack([np.linalg.norm(X - cen[c], axis=1) for c in range(10)], 1)
    Do = D.copy(); Do[np.arange(len(ytrue)), ytrue] = np.inf
    ncm = dist - Do.min(1)
    for lab, feat in [("dist_to_class_mean", dist), ("ink_mass", ink), ("nearest_centroid_margin", ncm)]:
        r_all = np.mean([spearmanr(R[t]["key"], feat)[0] for t in true])
        r_wc = np.mean([within_class_spearman(R[t]["key"], feat, ytrue) for t in true])
        print(f"baseline {lab:24s} spearman with runs {r_all:.3f}   within-class {r_wc:.3f}")
        out[f"baseline_{lab}"] = dict(all=float(r_all), within=float(r_wc))
        out[f"baseline_{lab}"]["per_arch"] = {a: float(np.mean([spearmanr(R[t]["key"], feat)[0] for t in true if arch(t) == a])) for a in archs}
        print("     per arch:", {a: round(v, 3) for a, v in out[f"baseline_{lab}"]["per_arch"].items()})
    # tail agreement: overlap of the last-learned k% between run pairs, and with the NCM baseline's hardest k%
    out["tail"] = {}
    for frac in [0.01, 0.05, 0.10]:
        qq = int(frac * len(ytrue)); Ls = {n: set(np.argsort(R[n]["key"])[-qq:]) for n in true}
        Lncm = set(np.argsort(ncm)[-qq:])
        tab = {}
        for a in archs:
            for b in archs:
                v = [len(Ls[p] & Ls[q_]) / qq for p in true for q_ in true if arch(p) == a and arch(q_) == b and p != q_]
                tab[f"{a}|{b}"] = float(np.mean(v))
            tab[f"{a}|ncm"] = float(np.mean([len(Ls[p] & Lncm) / qq for p in true if arch(p) == a]))
        out["tail"][str(frac)] = tab
        print(f"last {frac:.0%} overlap (chance {frac:.0%}):", " ".join(f"{k}={v:.2f}" for k, v in tab.items()))
    json.dump(out, open(os.path.join(C, "agreement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
