"""Analyse a teacher/student sweep: fit L(N) = c N^-alpha per (family, d), measure the intrinsic
dimension of each trained student's final hidden layer (TwoNN + Levina-Bickel MLE, 12k points,
as in Sharma & Kaplan), and the neighbour-count curves N(r) used for the 'd-line' pictures.

    python ts_analyze.py cache/main/ts_main.npz [--id_widths 16,45,90] [--device cuda]
Writes cache/main/ts_analysis.npz + .json (everything the renderers need).
"""
import argparse, json
import numpy as np
import torch
from idlib import knn_dists, twonn, mle_levina_bickel, neighbour_count_curve

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--id_widths", default="16,32,64")
p.add_argument("--n_id", type=int, default=12000)
p.add_argument("--device", default="cuda")
p.add_argument("--fit_lo", type=int, default=0, help="min N index to include in fit")
p.add_argument("--agg", default="median", choices=["median", "min"])
a = p.parse_args()
if a.device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.05)

z = np.load(a.npz, allow_pickle=False)
cfg = z["configs"]; dims = [int(d) for d in z["dims"]]; fams = [str(f) for f in z["fams"]]
D = int(z["D"])
widths = sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_test"))
Ns = np.array([int(z[f"w{n}_N"]) for n in widths])
test = np.stack([z[f"w{n}_test"] for n in widths])    # (W, M)
train = np.stack([z[f"w{n}_train"] for n in widths])
out = dict(widths=widths, Ns=Ns.tolist(), dims=dims, fams=fams, fits={}, ids={})
agg = np.median if a.agg == "median" else np.min

# ------------------------------------------------------------------ alpha fits
for f in fams:
    for d in dims:
        m = (cfg[:, 0] == f) & (cfg[:, 1] == str(d))
        L = agg(test[:, m], axis=1)
        x, y = np.log(Ns[a.fit_lo:]), np.log(L[a.fit_lo:])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        alpha = -coef[0]
        # bootstrap over seeds (resample seeds per width) for a CI on alpha
        rng = np.random.default_rng(0)
        al = []
        T = test[a.fit_lo:, m]
        for _ in range(400):
            idx = rng.integers(0, T.shape[1], T.shape[1])
            yy = np.log(agg(T[:, idx], axis=1))
            al.append(-np.linalg.lstsq(A, yy, rcond=None)[0][0])
        # local slopes between neighbouring N (curvature diagnostic)
        loc = -np.diff(np.log(L)) / np.diff(np.log(Ns))
        out["fits"][f"{f}_{d}"] = dict(alpha=float(alpha), logc=float(coef[1]),
                                       alpha_ci=np.percentile(al, [2.5, 97.5]).tolist(),
                                       four_over_alpha=float(4 / alpha), L=L.tolist(),
                                       L_all=test[:, m].tolist(), train=agg(train[:, m], axis=1).tolist(),
                                       local_slopes=loc.tolist())
        print(f"{f:6s} d={d:2d} alpha={alpha:.3f} 4/alpha={4/alpha:5.2f}  local:{np.round(loc,2)}", flush=True)

# ------------------------------------------------------------------ student-representation IDs
rng = np.random.default_rng(42)
radii = np.logspace(-3, 1.5, 60)
curves = {}
for n in [int(v) for v in a.id_widths.split(",") if int(v) in widths]:
    Ws = [(torch.as_tensor(z[f"w{n}_W{i}"]), torch.as_tensor(z[f"w{n}_b{i}"])) for i in range(3)]
    for j, (f, d, s) in enumerate(cfg):
        d = int(d)
        Q = torch.as_tensor(z[f"Q{d}"]).double()
        zz = torch.as_tensor(rng.uniform(-.5, .5, (a.n_id, d)))
        x = (zz @ Q.T).float()
        h = torch.relu(x @ Ws[0][0][j].T + Ws[0][1][j])
        h = torch.relu(h @ Ws[1][0][j].T + Ws[1][1][j])   # final hidden layer
        H = h.double().numpy()
        r = knn_dists(H, 20, device=a.device)
        tw = twonn(r)
        mle = {k: mle_levina_bickel(r, k)["d"] for k in (5, 10, 20)}
        key = f"{f}_{d}_{s}_w{n}"
        out["ids"][key] = dict(twonn=tw["d_fit"], twonn_ml=tw["d_mle"], n_zero=tw["n_zero"],
                               **{f"mle{k}": v for k, v in mle.items()}, dead=int((H.std(0) == 0).sum()))
        if s == "0":
            scale = np.median(r[:, 0])
            curves[key] = neighbour_count_curve(H / scale, radii, n_centres=400, device=a.device)
        print(key, {k: round(v, 2) if isinstance(v, float) else v for k, v in out["ids"][key].items()}, flush=True)

# data-manifold ID sanity check (input x = Q z is exactly d-dimensional)
for d in dims:
    zz = rng.uniform(-.5, .5, (a.n_id, d)) @ z[f"Q{d}"].T
    r = knn_dists(zz, 20, device=a.device)
    out["ids"][f"input_{d}"] = dict(twonn=twonn(r)["d_fit"], mle10=mle_levina_bickel(r, 10)["d"])
    curves[f"input_{d}"] = neighbour_count_curve(zz / np.median(r[:, 0]), radii, n_centres=400, device=a.device)

base = a.npz.replace(".npz", "_analysis")
json.dump(out, open(base + ".json", "w"), indent=1)
np.savez(base + ".npz", radii=radii, **{f"curve_{k}": v for k, v in curves.items()})
print("wrote", base)
