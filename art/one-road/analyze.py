"""InPCA of every training trajectory of a task, with landmarks and null controls (CPU).

Recipe follows Mao, Griniasty, ..., Sethna, Chaudhari (PNAS 2024), itself InPCA of Quinn et al. (2019):
  * a network = the product distribution of its softmax outputs on N probe examples
  * Bhattacharyya distance, intensive (per-example average):
        d_B(P_u, P_v) = -(1/N) sum_n log sum_c sqrt(p_u^n(c) p_v^n(c))
  * W = -L D L / 2 with L = I - 11^T/m ; eigendecompose W = U Lambda U^T
  * coordinates x_k = U_k sqrt|Lambda_k|, components ordered by |Lambda_k|; Lambda_k<0 are "time-like"
  * explained stress of the top d components: 1 - sqrt(sum_{k>d} Lambda_k^2 / sum_k Lambda_k^2)
  * progress s = argmin_alpha d_G(P, geodesic P0 -> P*(alpha)), d_G = (1/N) sum_n arccos BC_n
  * trajectory distance (Mao eq. 5): d_traj(u, v) = int d_B(P_u(s), P_v(s)) ds over the shared range of s

Nulls built here, all put through the identical pipeline:
  * geodesic    : the straight road P0 -> P* in the sqrt-probability sphere (per example)
  * mixture     : linear mixtures (1-a) uniform + a onehot  (label smoothing path)
  * perm(run)   : the run's own predictions with probe examples permuted *within their true class*.
                  Identical loss curve, identical d_B to P0 and to P* at every checkpoint, identical
                  per-class behaviour; only which-example-is-learned-when is scrambled. If the
                  architectures' roads coincide merely because the metric + loss curve force it,
                  then d(A, B) ~ d(A, perm B).
  * init        : 8 extra untrained random inits of every architecture
"""
import argparse, glob, json, os
import numpy as np
import torch
from models import build, IMAGE_ARCHS, MODADD_ARCHS

ROOT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(16)


def load_runs(task):
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "cache", "runs", task, "*.npz"))):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        runs.append(dict(meta=m, name=m["name"], arch=m["arch"], opt=m["opt"], labels=m["labels"],
                         seed=m["seed"], steps=z["steps"], lp_tr=z["lp_tr"], lp_te=z["lp_te"],
                         tr_loss=z["tr_loss"], te_loss=z["te_loss"], tr_acc=z["tr_acc"], te_acc=z["te_acc"]))
    return runs


@torch.no_grad()
def init_points(task, d, C, seeds=range(100, 108)):
    archs = MODADD_ARCHS if task == "modadd" else IMAGE_ARCHS
    img = task != "modadd"
    X = {}
    for split, key, pk in (("tr", "xtr", "ptr"), ("te", "xte", "pte")):
        x = d[key][d[pk]]
        X[split] = torch.tensor(x, dtype=torch.float32) if img else torch.tensor(x)
    out = []
    for a in archs:
        for s in seeds:
            torch.manual_seed(s)
            m = build(task, a, C, tuple(X["tr"].shape[1:]) if img else None).eval()
            out.append(dict(arch=a, seed=s, **{sp: torch.log_softmax(m(X[sp]).double(), -1).numpy()
                                             for sp in ("tr", "te")}))
    return out


def bc_logsum(A, B=None, chunk=8):
    """A: (M, N, C) sqrt-probs, float32 torch. Returns sum_n log BC_n(u, v) as (M, M') float64."""
    B = A if B is None else B
    M, N, _ = A.shape
    out = torch.zeros(M, B.shape[0], dtype=torch.float64)
    hel = torch.zeros(M, B.shape[0], dtype=torch.float64)
    At, Bt = A.transpose(0, 1).contiguous(), B.transpose(0, 1).contiguous()  # (N, M, C)
    for i in range(0, N, chunk):
        bc = torch.bmm(At[i:i + chunk], Bt[i:i + chunk].transpose(1, 2)).clamp_(1e-30, 1.0)
        out += torch.log(bc).sum(0).double()
        hel += bc.sum(0).double()
    return out, hel


def inpca(D):
    m = len(D)
    L = np.eye(m) - 1.0 / m
    W = -0.5 * L @ D @ L
    lam, U = np.linalg.eigh(W)
    o = np.argsort(-np.abs(lam))
    lam, U = lam[o], U[:, o]
    X = U * np.sqrt(np.abs(lam))
    tot = (lam ** 2).sum()
    stress = [1 - np.sqrt((lam[k:] ** 2).sum() / tot) for k in range(1, 11)]
    return X, lam, stress


def geodesic_bc_weights(alpha, C):
    """sqrt-prob of the geodesic point = a(alpha) sqrt(u) + b(alpha) e_y  (slerp on the sphere)."""
    th = np.arccos(1 / np.sqrt(C))
    a = np.sin((1 - alpha) * th) / np.sin(th)
    b = np.sin(alpha * th) / np.sin(th)
    return a, b


def progress(S, y, C, grid=np.linspace(0, 1, 401)):
    """S (M, N, C) sqrt-probs. Returns s = argmin_alpha d_G(P, geodesic(alpha))."""
    su = S.sum(-1) / np.sqrt(C)                                  # <sqrt p, sqrt u>
    sy = np.take_along_axis(S, y[None, :, None], 2)[..., 0]      # sqrt p_y
    best = np.full(len(S), np.inf); s = np.zeros(len(S))
    for al in grid:
        a, b = geodesic_bc_weights(al, C)
        dG = np.arccos(np.clip(a * su + b * sy, -1, 1)).mean(1)
        upd = dG < best; best[upd] = dG[upd]; s[upd] = al
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True)
    args = ap.parse_args(); task = args.task
    d = np.load(os.path.join(ROOT, "cache", f"data_{task}.npz")); C = int(d["C"])
    runs = load_runs(task)
    inits = init_points(task, d, C)
    print(task, len(runs), "runs")
    rng = np.random.default_rng(0)
    res = {}
    for split in ("tr", "te"):
        y = d["ytr"][d["ptr"]] if split == "tr" else d["yte"][d["pte"]]
        N = len(y)
        pts, info = [], []

        def add(sq, **kw):
            pts.append(sq.astype(np.float32)); info.append(kw)

        # landmarks
        add(np.full((N, C), 1 / np.sqrt(C)), kind="P0")
        add(np.eye(C)[y], kind="Pstar")
        # real runs
        for ri, r in enumerate(runs):
            lp = r["lp_" + split].astype(np.float64)
            for k in range(len(lp)):
                add(np.exp(lp[k] / 2), kind="run", run=ri, k=k)
        # perm nulls: within-class permutation of probe examples, one per true-label run
        perms = {}
        for ri, r in enumerate(runs):
            if r["labels"] != "true":
                continue
            perm = np.arange(N)
            for c in range(C):
                idx = np.where(y == c)[0]; perm[idx] = rng.permutation(idx)
            perms[ri] = perm
            lp = r["lp_" + split].astype(np.float64)
            for k in range(len(lp)):
                add(np.exp(lp[k][perm] / 2), kind="perm", run=ri, k=k)
        # geodesic and linear-mixture straight roads
        for al in np.linspace(0, 1, 41)[1:-1]:
            a, b = geodesic_bc_weights(al, C)
            add(a / np.sqrt(C) + b * np.eye(C)[y], kind="geo", alpha=al)
            add(np.sqrt((1 - al) / C + al * np.eye(C)[y]), kind="mix", alpha=al)
        # untrained random inits
        for ii, it in enumerate(inits):
            add(np.exp(it[split] / 2), kind="init", arch=it["arch"], seed=it["seed"])

        S = np.stack(pts)
        print(split, "points", S.shape)
        lg, bcs = bc_logsum(torch.from_numpy(S))
        D = -lg.numpy() / N
        # robustness metric: per-example Hellinger^2 averaged, = 1 - mean BC = squared Euclidean distance
        # between the concatenated sqrt-probability vectors / 2N. Bounded; its InPCA is plain PCA of sqrt(p).
        DH = 1.0 - bcs.numpy() / N
        DH = np.maximum(0.5 * (DH + DH.T), 0); np.fill_diagonal(DH, 0.0)
        D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0); D = np.maximum(D, 0)
        s = progress(S.astype(np.float64), y, C)
        kinds = np.array([i["kind"] for i in info])
        runidx = np.array([i.get("run", -1) for i in info]); kidx = np.array([i.get("k", -1) for i in info])
        np.savez(os.path.join(ROOT, "cache", f"D_{task}_{split}.npz"), D=D, DH=DH, s=s, kinds=kinds, run=runidx, k=kidx,
                 info=json.dumps(info, default=float))
        res[split] = dict(D=D, s=s, kinds=kinds, run=runidx, k=kidx)
    json.dump(dict(runs=[r["meta"] for r in runs]), open(os.path.join(ROOT, "cache", f"runs_{task}.json"), "w"))


if __name__ == "__main__":
    main()
