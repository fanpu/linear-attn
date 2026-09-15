"""Noiseless isotropic limit: does the (best gated delta rule)/(Kalman) steady-state ratio approach pi/2 as d grows?
Age-model prediction: delta rule refreshes directions uniformly (exponential ages, mean d -> excess risk qd);
Kalman refreshes in proportion to uncertainty ~ age (Rayleigh ages, mean 2d/pi -> excess risk (2/pi) qd).
Writes cache/dscale.json; checkpoints after each config."""
import json, math, sys, time
from pathlib import Path
import torch
from sim import simulate
from theory import optimal_gdn
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pause import PAUSE_EXIT, should_pause

Path("cache").mkdir(exist_ok=True)
out_path = Path("cache/dscale.json")
res = json.load(open(out_path)) if out_path.exists() else []
done = {(r["d"], r["qd"]) for r in res}
for qd in [0.01, 0.1]:
    for d in [16, 32, 64, 128, 256, 512, 1024]:
        if (d, qd) in done:
            continue
        if should_pause(every_s=0):
            sys.exit(PAUSE_EXIT)
        q = qd / d
        a_, b_, r_th = optimal_gdn(q, 0.0, d)
        tau = 1 / (1 - a_**2 * (1 - (2 * b_ - b_**2) / d))
        T = int(max(2000, 20 * tau))
        batch = max(16, min(512, 2**20 // d**2 * 16))
        batch -= batch % 8
        t0 = time.time()
        o = simulate(d, q, 0.0, T, batch=batch, alphas=[a_], betas=[b_], seed=20, chunks=8, device="cuda")
        g = o["gdn"][0, T // 2:].mean(0)
        k = o["kf"][T // 2:].mean(0)
        ratio = (g / k).numpy()
        res.append(dict(d=d, qd=qd, alpha=a_, beta=b_, T=T, batch=batch, gdn_theory=r_th, gdn=float(g.mean()),
                        kf=float(k.mean()), kf_over_qd=float(k.mean() / qd), ratio=float(g.mean() / k.mean()),
                        ratio_sd=float(ratio.std(ddof=1) / math.sqrt(8))))
        r = res[-1]
        print(f"qd={qd} d={d:5d} B={batch:4d} T={T:6d}  GDN/qd={r['gdn']/qd:.4f} (th {r_th/qd:.4f})  "
              f"KF/qd={r['kf_over_qd']:.4f}  ratio={r['ratio']:.4f} +- {r['ratio_sd']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"))
print("pi/2 =", math.pi / 2, " 1/ln2 =", 1 / math.log(2))
