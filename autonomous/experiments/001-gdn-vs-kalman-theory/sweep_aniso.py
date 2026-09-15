"""P3 + P4: anisotropic inputs (condition number kappa) with static key preconditioning M = Sigma^{-s}.
The delta-rule optimum is found by grid search in simulation (the closed form assumes isotropy):
alpha in {a, 1-2(1-a), 1-(1-a)/2} (the isotropic optimum is alpha* = a to 5 digits), beta on a log grid.
Convergence check: steady state = mean of last quarter; 'drift' = relative change vs third quarter.
Writes cache/aniso.json; checkpoints after each config."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from sim import simulate, spectrum
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pause import PAUSE_EXIT, should_pause

Path("cache").mkdir(exist_ok=True)
out_path = Path("cache/aniso.json")
res = json.load(open(out_path)) if out_path.exists() else []
done = {(r["d"], r["kappa"], r["qd"], r["sigma2"]) for r in res}
d = 64
betas = np.geomspace(0.01, 1.9, 24)
for kappa in [1, 10, 100, 1000]:
    for qd in [0.01, 0.1, 1.0]:
        for s2 in [0.0, 0.1]:
            if (d, kappa, qd, s2) in done:
                continue
            if should_pause(every_s=0):
                sys.exit(PAUSE_EXIT)
            q = qd / d
            a = math.sqrt(1 - q)
            alphas = [a, 1 - 2 * (1 - a), 1 - (1 - a) / 2]
            lam = spectrum(d, kappa)
            slow = float(lam.mean() / lam.min())
            T = int(np.clip(20 * min(1 / q, d * slow), 4000, 100000))
            entry = dict(d=d, kappa=kappa, qd=qd, q=q, sigma2=s2, T=T, by_s={})
            t0 = time.time()
            for s in [0.0, 0.5, 1.0]:
                o = simulate(d, q, s2, T, batch=256, alphas=alphas, betas=betas, kappa=kappa, precond_s=s,
                             kalman=(s == 0.0), seed=30, chunks=8, device="cuda")
                last, third = slice(3 * T // 4, T), slice(T // 2, 3 * T // 4)
                G = o["gdn"][:, last].mean(1)                       # [G, chunks]
                gm = G.mean(-1).numpy()
                i = int(np.argmin(gm))
                g3 = float(o["gdn"][i, third].mean())
                entry["by_s"][str(s)] = dict(best_alpha=alphas[i // len(betas)], best_beta=float(betas[i % len(betas)]),
                                             gdn=float(gm[i]), gdn_chunks=G[i].tolist(),
                                             drift=abs(g3 - gm[i]) / gm[i],
                                             beta_edge=bool(i % len(betas) in (0, len(betas) - 1)))
                if s == 0.0:
                    k = o["kf"][last].mean(0)
                    entry["kf"], entry["kf_chunks"] = float(k.mean()), k.tolist()
                    entry["kf_drift"] = abs(float(o["kf"][third].mean()) - entry["kf"]) / entry["kf"]
            for s, e in entry["by_s"].items():
                e["ratio"] = e["gdn"] / entry["kf"]
            res.append(entry)
            json.dump(res, open(out_path, "w"))
            print(f"kappa={kappa:5d} qd={qd:4.2f} s2={s2:.1f} T={T:6d} KF={entry['kf']:.4g} | " + "  ".join(
                f"s={s}: ratio={e['ratio']:.3f} (b*={e['best_beta']:.3g}{' EDGE' if e['beta_edge'] else ''}, drift {e['drift']:.1%})"
                for s, e in entry["by_s"].items()) + f" ({time.time()-t0:.0f}s)", flush=True)
