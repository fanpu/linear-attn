"""P2: isotropic inputs. Best gated delta rule (closed-form optimum) vs Kalman filter, steady state and transient.
Writes cache/iso.json. Prints one line per config."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from sim import simulate
from theory import gdn_excess_risk, optimal_gdn
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from pause import PAUSE_EXIT, should_pause

Path("cache").mkdir(exist_ok=True)
dev = "cuda" if "--cuda" in sys.argv else "cpu"
ds = [32, 128]
qds = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
s2s = [0.0, 0.1, 1.0]
out_path = Path('cache/iso.json')
res = json.load(open(out_path)) if out_path.exists() else []
done = {(r['d'], r['qd'], r['sigma2']) for r in res}
for d in ds:
    for qd in qds:
        for s2 in s2s:
            if (d, qd, s2) in done:
                continue
            if should_pause(every_s=0):
                sys.exit(PAUSE_EXIT)
            q = qd / d
            a_, b_, r_th = optimal_gdn(q, s2, d)
            # run for 20 relaxation times of the delta rule at its optimum (the slower of the two memories to settle;
            # the Kalman filter's gain is never smaller), so the last half is >= 10 relaxation times from the start
            tau = 1 / (1 - a_**2 * (1 - (2 * b_ - b_**2) / d))
            T = int(max(2000, 20 * tau))
            t0 = time.time()
            out = simulate(d, q, s2, T, batch=512, alphas=[a_], betas=[b_], seed=10, chunks=8, device=dev)
            half = slice(T // 2, T)
            g = out["gdn"][0, half].mean(0).numpy()   # per-chunk steady-state
            k = out["kf"][half].mean(0).numpy()
            ratio = g / k
            idx = np.unique(np.geomspace(1, T, 80).astype(int)) - 1
            res.append(dict(d=d, qd=qd, q=q, sigma2=s2, alpha=a_, beta=b_, gdn_theory=r_th,
                            gdn_sim=float(g.mean()), gdn_sim_sd=float(g.std(ddof=1) / math.sqrt(8)),
                            kf_sim=float(k.mean()), kf_sim_sd=float(k.std(ddof=1) / math.sqrt(8)),
                            ratio=float(g.mean() / k.mean()), ratio_chunks=ratio.tolist(), T=T,
                            curve_t=(idx + 1).tolist(), curve_gdn=out["gdn"][0].mean(-1)[idx].tolist(),
                            curve_kf=out["kf"].mean(-1)[idx].tolist()))
            r = res[-1]
            print(f"d={d:3d} qd={qd:5.2f} s2={s2:.1f}  a*={a_:.5f} b*={b_:.3f}  GDN th={r_th:.4g} sim={r['gdn_sim']:.4g}  "
                  f"KF={r['kf_sim']:.4g}  ratio={r['ratio']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"))  # checkpoint after every config
