"""Sweep eta, burn in, record every visited iterate.  Saves cache/bif_<name>.npz.

Usage:
  python compute_bifurcation.py prod2 [--n 16000 --burn 20000 --rec 2048]
  python compute_bifurcation.py prod4 | logistic | dln | prod2_multi | prod4_multi
"""
import argparse
import time
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"

# Declared settings.  Fixed inits are deliberately *unbalanced* so the approach to the
# balanced minimum (sharpness adaptivity) is part of what is measured.
SETTINGS = {
    "prod2": dict(k=2, lo=0.85, hi=2.0, x0=(1.1, 0.9)),
    "prod4": dict(k=4, lo=0.42, hi=1.21, x0=(1.1, 0.9, 1.05, 0.95)),
    "logistic": dict(k=1, lo=2.8, hi=4.0, x0=(0.3,)),
}


def sweep_numpy(k, etas, x0, burn, rec, lyap_tail=2000, seed=0, logistic=False):
    N = len(etas)
    x = np.broadcast_to(np.asarray(x0, float), (N, max(k, 1))).copy()
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(x.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if not logistic:
        u = np.full(N, float(np.prod(np.abs(x0))) ** (1.0 / k) * 1.0001)  # balanced-line reference
    P = np.empty((N, rec), np.float32)
    X1 = np.empty((N, rec), np.float32)
    LL = np.empty((N, rec), np.float32)
    lyap = np.zeros(N)
    lyap_bal = np.zeros(N)
    lyap_tr = np.zeros(N)
    alive = np.ones(N, bool)
    first_bad = np.full(N, -1)
    with np.errstate(all="ignore"):
        for t in range(burn + rec):
            recording = t >= burn
            if logistic:
                if recording:
                    d = nm.logistic_deriv(x[:, 0], etas)
                    lyap += np.log(np.abs(d))
                x = nm.logistic_step(x, etas[:, None])
            else:
                if t >= burn - lyap_tail:
                    v = nm.prod_tangent(x, v, etas)
                    nv = np.linalg.norm(v, axis=1)
                    if recording:
                        lyap += np.log(nv)
                    v /= nv[:, None]
                    if recording:
                        lyap_bal += np.log(np.abs(nm.bal_deriv(u, etas, k)))
                        lyap_tr += np.log(np.abs(nm.bal_transverse(u, etas, k)))
                    u = nm.bal_step(u, etas, k)
                else:
                    u = nm.bal_step(u, etas, k)
                x = nm.prod_step(x, etas)
            bad = ~np.isfinite(x).all(1) | (np.abs(x) > nm.BIG).any(1)
            newbad = bad & alive
            first_bad[newbad] = t
            alive &= ~bad
            x[bad] = 0.5  # park dead columns (masked later) so no overflow warnings propagate
            if recording:
                i = t - burn
                if logistic:
                    P[:, i] = x[:, 0]
                    X1[:, i] = x[:, 0]
                    LL[:, i] = np.nan
                else:
                    p = np.prod(x, 1)
                    P[:, i] = p
                    X1[:, i] = x[:, 0]
                    LL[:, i] = np.log10(0.5 * (p - 1.0) ** 2 + 1e-300)
    P[~alive] = np.nan
    X1[~alive] = np.nan
    LL[~alive] = np.nan
    out = dict(P=P, X1=X1, logloss=LL, lyap=lyap / rec, alive=alive, first_bad=first_bad)
    if not logistic:
        out.update(lyap_bal=lyap_bal / rec, lyap_trans=lyap_tr / rec,
                   final_x=x.astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--burn", type=int, default=20000)
    ap.add_argument("--rec", type=int, default=2048)
    ap.add_argument("--ninit", type=int, default=48)
    ap.add_argument("--tag", default="")
    ap.add_argument("--onlyP", action="store_true", help="store only P (hero density runs)")
    a = ap.parse_args()
    t0 = time.time()
    base = a.name.replace("_multi", "")
    S = SETTINGS[base]
    etas = np.linspace(S["lo"], S["hi"], a.n)
    logistic = base == "logistic"
    if a.name.endswith("_multi"):
        rng = np.random.default_rng(1)
        k = S["k"]
        # inits: log-normal coordinates around 1 (spread 0.35 in log), random signs off
        inits = np.exp(0.35 * rng.standard_normal((a.ninit, k)))
        Ps, X1s, alive = [], [], []
        for j, x0 in enumerate(inits):
            o = sweep_numpy(k, etas, x0, a.burn, a.rec, lyap_tail=1)
            Ps.append(o["P"])
            X1s.append(o["X1"])
            alive.append(o["alive"])
            print(f"init {j} {np.round(x0, 3)} alive {o['alive'].mean():.3f}  {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"{CACHE}/bif_{a.name}{a.tag}.npz", etas=etas, inits=inits, P=np.array(Ps),
                            X1=np.array(X1s), alive=np.array(alive), burn=a.burn, rec=a.rec)
    else:
        o = sweep_numpy(S["k"], etas, S["x0"], a.burn, a.rec, logistic=logistic)
        if a.onlyP:
            o.pop("X1"); o.pop("logloss")
        np.savez(f"{CACHE}/bif_{a.name}{a.tag}.npz", etas=etas, x0=np.array(S["x0"]), burn=a.burn,
                 rec=a.rec, k=S["k"], **o)
        print("alive frac", o["alive"].mean(), "lyap>0 frac", (o["lyap"][o["alive"]] > 0).mean())
    print(f"done {a.name} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
