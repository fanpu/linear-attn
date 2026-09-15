"""Best key-metric exponent s (M = Sigma^-s) for a delta-rule memory, as a function of context length T and drift q.
Objective = excess risk averaged over the context (what in-context training minimizes). See README.md."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "001-gdn-vs-kalman-theory"))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from sim import simulate
from pause import PAUSE_EXIT, should_pause

d, kappa, chunks = 32, 100, 8
Tmax = 64 * d
Ts = [d // 2, d, 2 * d, 4 * d, 16 * d, 64 * d]
svals = np.round(np.arange(0, 1.2501, 0.125), 3)
betas = np.geomspace(0.02, 1.9, 16)
out_path = HERE / "cache/metric.json"
res = json.load(open(out_path)) if out_path.exists() else {}


def prefix_mean(curve):          # [..., T, chunks] -> [..., T, chunks]
    return curve.cumsum(-2) / torch.arange(1, curve.shape[-2] + 1)[:, None]


for qd in [0.0, 0.01, 0.1, 1.0]:
    for s2 in [0.0, 0.1]:
        key = f"qd={qd},s2={s2}"
        if key in res:
            continue
        q = qd / d
        a = math.sqrt(1 - q)
        alphas = [a, 1 - 2 * (1 - a), 1 - (1 - a) / 2] if q > 0 else [1.0]
        best = np.zeros((len(svals), len(Ts), chunks))   # min over (alpha,beta) of context-avg risk, per chunk
        best_mean = np.zeros((len(svals), len(Ts)))
        kf = None
        for i, s in enumerate(svals):
            if should_pause(every_s=0):
                sys.exit(PAUSE_EXIT)
            o = simulate(d, q, s2, Tmax, batch=512, alphas=alphas, betas=betas, kappa=kappa, precond_s=float(s),
                         kalman=(i == 0), seed=40, chunks=chunks, device="cuda")
            pm = prefix_mean(o["gdn"])                         # [G, T, chunks]
            for j, T in enumerate(Ts):
                avg = pm[:, T - 1]                             # [G, chunks]
                best_mean[i, j] = avg.mean(-1).min().item()
                best[i, j] = avg.min(0).values.numpy()
            if i == 0:
                kf = [prefix_mean(o["kf"][:, None].squeeze(1))[T - 1].mean().item() if False else
                      float((o["kf"].cumsum(0) / torch.arange(1, Tmax + 1)[:, None])[T - 1].mean()) for T in Ts]

        def argmin_refined(r):           # parabola through the grid minimum and its neighbours, in log risk
            k = int(np.argmin(r))
            if 0 < k < len(r) - 1:
                y0, y1, y2 = np.log(r[k - 1:k + 2])
                den = y0 - 2 * y1 + y2
                if den > 0:
                    return float(svals[k] + 0.125 * 0.5 * (y0 - y2) / den)
            return float(svals[k])

        s_star = [argmin_refined(best_mean[:, j]) for j in range(len(Ts))]
        s_chunks = [[argmin_refined(best[:, j, c]) for c in range(chunks)] for j in range(len(Ts))]
        res[key] = dict(qd=qd, sigma2=s2, Ts=Ts, svals=svals.tolist(), best_mean=best_mean.tolist(), s_star=s_star,
                        s_star_sd=[float(np.std(x, ddof=1)) for x in s_chunks], kf=kf,
                        gdn_best=[float(best_mean[:, j].min()) for j in range(len(Ts))])
        json.dump(res, open(out_path, "w"))
        print(f"{key:16s} s*(T/d={[T/d for T in Ts]}) = {np.round(s_star, 2).tolist()} "
              f"sd={np.round(res[key]['s_star_sd'], 2).tolist()}  GDN*/KF={np.round(np.array(res[key]['gdn_best'])/np.array(kf), 3).tolist()}",
              flush=True)
