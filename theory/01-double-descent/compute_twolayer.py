"""Does the random-features formula still describe a two-layer ReLU net once its first layer is trained?  CPU, float32.

Same data model as the Mei-Montanari check: x ~ Unif(S^{d-1}(sqrt d)), linear target ||beta||^2 = 1, noise tau^2.
For each width N and seed we compare
  (rf)   frozen first layer, min-norm output weights (exact),
  (full) both layers trained from the same init by full-batch Adam on squared loss (no explicit regularization).
    python compute_twolayer.py  -> cache/twolayer.npz
"""
import os, pathlib, time, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from multiprocessing import Pool

import dd_core as C

HERE = pathlib.Path(__file__).resolve().parent
D, N_TRAIN, TAU2 = 20, 400, 0.25
WIDTHS = np.unique(np.round(np.geomspace(2, 2000, 26)).astype(int))
WIDTHS = np.unique(np.concatenate([WIDTHS, [12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 360, 380, 400, 420, 440]]))
STEPS = int(os.environ.get("TL_STEPS", 40000))


def job(args):
    N, seed = args
    torch.set_num_threads(1)
    rng = np.random.default_rng([N, seed])
    beta = rng.standard_normal(D); beta /= np.linalg.norm(beta)
    X = C.sphere(N_TRAIN, D, rng); Xt = C.sphere(4000, D, rng); Th = C.sphere(N, D, rng)
    y = X @ beta + np.sqrt(TAU2) * rng.standard_normal(N_TRAIN)
    ft = Xt @ beta
    # (rf) min-norm output weights on frozen features

    Z = np.maximum(X @ Th.T / np.sqrt(D), 0); Zt = np.maximum(Xt @ Th.T / np.sqrt(D), 0)
    a_rf = np.linalg.lstsq(Z, y, rcond=None)[0]
    rf_err = np.mean((Zt @ a_rf - ft) ** 2)
    # (full) train both layers from theta = Th, a = 0 (so f starts at 0), Adam, cosine lr
    Xt_t = torch.tensor(Xt, dtype=torch.float32); X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    th = torch.tensor(Th, dtype=torch.float32, requires_grad=True)
    a = torch.zeros(N, requires_grad=True)
    opt = torch.optim.Adam([th, a], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    scale = 1.0 / np.sqrt(D)
    f = lambda Xin: torch.relu(Xin @ th.T * scale) @ a
    for s in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = ((f(X_t) - y_t) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        train_mse = float(((f(X_t) - y_t) ** 2).mean())
        full_err = float(((f(Xt_t) - torch.tensor(ft, dtype=torch.float32)) ** 2).mean())
        move = float((th - torch.tensor(Th, dtype=torch.float32)).norm() / np.linalg.norm(Th))
    return rf_err, full_err, train_mse, move


if __name__ == "__main__":
    seeds = int(os.environ.get("TL_SEEDS", 3))
    t0 = time.time()
    tasks = [(int(N), s) for N in WIDTHS for s in range(seeds)]
    with Pool(4) as pool:
        res = []
        for i, r in enumerate(pool.imap(job, tasks)):
            res.append(r)
            if i % 10 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    res = np.array(res).reshape(len(WIDTHS), seeds, 4)
    psi1 = np.geomspace(0.05, 120, 500)
    th = np.array([C.mm_risk(p, N_TRAIN / D, 0.0, 1.0, TAU2)[0] if abs(p - N_TRAIN / D) > 1e-9 else np.nan for p in psi1])
    np.savez(HERE / "cache" / "twolayer.npz", widths=WIDTHS, res=res, D=D, n=N_TRAIN, tau2=TAU2, th_psi1=psi1, th=th, steps=STEPS)
    for N, r in zip(WIDTHS, res):
        print(f"N={N:5d}  N(d+1)={N*(D+1):6d}  rf {r[:,0].mean():.3g}  full {r[:,1].mean():.3g}  train_mse {r[:,2].mean():.2e}  move {r[:,3].mean():.3f}")
    print(f"done {time.time() - t0:.0f}s")
