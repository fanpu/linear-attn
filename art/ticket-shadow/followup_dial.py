"""Follow-up (2026-09-26 evening): the optimiser dial. Summarises cache/followup_*/ (adam_dial.py) and
the original raw-pixel Adam and SGD runs (imp.py, jobs 664 and 728) with the README's metrics:
  - mean connections kept per pixel at round R, binned by how many of the 55k training images light it
  - r(pixel std), r(ever lit) of the seed-mean shadow
  - plateau = kept(lit in 10-99 images) / kept(lit in >20k images): ~1 or more is Adam's support plateau,
    ~0.2 is SGD's graded core
  - round-0 displacement: mean |W_T - W_0| per pixel after the first dense training, and its r with std / ever-lit
Writes cache/followup_dial.json and prints a table.
"""
import glob, json, os
import numpy as np
from imp import load_dataset, SIZES

CACHE = "/home/fzeng/ml/research/art/ticket-shadow/cache"
BINS = [0, 1, 10, 100, 1000, 5000, 20000, 60000]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def load(cond, R):
    f = f"{CACHE}/{cond}/round_{R:02d}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f); z0 = np.load(f"{CACHE}/{cond}/round_00.npz")
    m1 = np.unpackbits(z["mask0"], axis=-1)[..., :300].astype(bool)
    labels = z["cfg"] if "cfg" in z else np.array([cond] * len(z["modes"]))
    modes = z["modes"]
    disp = np.abs(z0["W0"].astype(np.float64) - z0["init_W0"]).mean(2)
    return dict(labels=labels, modes=modes, shadow=m1.sum(2).astype(float), acc=z["es_test_acc"], disp=disp,
                acc0=z0["es_test_acc"])


def main(R=15):
    xtr, _, _, _ = load_dataset("mnist")
    x = xtr[:55000]; cnt = (x > 0).sum(0); std = x.std(0); lit = (cnt > 0).astype(float)
    groups = []
    for cond, rename in [("mnist_adam_raw_rw0", "adam_eps1e-8 (job 664)"), ("mnist_sgd_raw_rw0", "sgd_lr0.1 (job 728)")]:
        d = load(cond, R)
        idx = [i for i, m in enumerate(d["modes"]) if m == "imp"]
        groups.append((rename, d["shadow"][idx], d["acc"][idx], d["disp"][idx], d["acc0"][idx]))
    for cond in sorted(d for d in glob.glob(f"{CACHE}/followup_*") if os.path.isdir(d)):
        cond = os.path.basename(cond)
        if "smoke" in cond:
            continue
        d = load(cond, R)
        if d is None:
            print("missing round", R, "for", cond); continue
        for lab in dict.fromkeys(d["labels"].tolist()):
            idx = [i for i, l in enumerate(d["labels"]) if l == lab]
            new = (lab, d["shadow"][idx], d["acc"][idx], d["disp"][idx], d["acc0"][idx])
            prev = [g for g in groups if g[0] == lab]
            if prev:  # extra seeds of the same configuration from another job: concatenate
                p = prev[0]; groups.remove(p)
                new = (lab,) + tuple(np.concatenate([a, b]) for a, b in zip(p[1:], new[1:]))
            groups.append(new)
    out = {}
    print(f"round {R}; pixels per bin:", [int(((cnt >= BINS[i]) & (cnt < BINS[i + 1])).sum()) for i in range(len(BINS) - 1)])
    print(f"{'config':26s} n " + " ".join(f"{f'[{BINS[i]},{BINS[i+1]})':>11s}" for i in range(len(BINS) - 1))
          + "  plateau r(std) r(lit) | r_seeds | disp0 r(std) r(lit) | acc0 accR")
    for lab, sh, acc, disp, acc0 in groups:
        s = sh.mean(0)
        row = [float(s[(cnt >= BINS[i]) & (cnt < BINS[i + 1])].mean()) for i in range(len(BINS) - 1)]
        plateau = row[2] / row[6]
        rs = [corr(sh[i], sh[j]) for i in range(len(sh)) for j in range(i + 1, len(sh))]
        dm = disp.mean(0)
        o = dict(bins=row, plateau=plateau, r_std=corr(s, std), r_lit=corr(s, lit), n=len(sh),
                 r_seeds=float(np.mean(rs)) if rs else None,
                 plateau_seeds=[float(sh[i][(cnt >= 10) & (cnt < 100)].mean() / sh[i][cnt >= 20000].mean()) for i in range(len(sh))],
                 disp_r_std=corr(dm, std), disp_r_lit=corr(dm, lit), acc=float(acc.mean()), acc0=float(acc0.mean()),
                 disp_plateau=float(dm[(cnt >= 10) & (cnt < 100)].mean() / dm[cnt >= 20000].mean()),
                 shadow_mean=s.tolist())
        out[lab] = o
        print(f"{lab:26s} {len(sh)} " + " ".join(f"{v:11.1f}" for v in row)
              + f"  {plateau:6.2f} {o['r_std']:6.2f} {o['r_lit']:6.2f} | {o['r_seeds'] if o['r_seeds'] is None else round(o['r_seeds'], 2)} | "
              f"{o['disp_plateau']:5.2f} {o['disp_r_std']:6.2f} {o['disp_r_lit']:6.2f} | {100*o['acc0']:.1f} {100*o['acc']:.1f}")
    json.dump(out, open(f"{CACHE}/followup_dial_r{R}.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
