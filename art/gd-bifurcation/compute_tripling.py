"""Self-similarity of the nested period-3 windows, measured.

For the tower of windows with base periods 3, 9, 27, 81, 243 (cache/windows.json, 'tower'), find the
superstable parameter s_m (Floquet multiplier of the period-3^m orbit = 0) by bisection with Newton
continuation, in numpy longdouble (IEEE quad on aarch64), on the balanced-line map of GD on
1/2(x1x2x3x4-1)^2.  The ratios  (s_m - s_{m-1}) / (s_{m+1} - s_m)  should approach the universal
period-tripling constant of quadratic unimodal maps (~55.25) if the windows are true renormalised copies.
The same is done for the logistic map as a control.  Output: cache/tripling.json
"""
import json
import numpy as np
from compute_feigenbaum import make_1d, orbit_newton, simulate

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
LD = np.longdouble


def superstable(name, lo, hi, p):
    S = make_1d(name, LD)
    S64 = make_1d(name)
    w = hi - lo
    # find a parameter inside the run where the period-p orbit is stable
    for f in [0.02, 0.05, 0.1, 0.2, 0.3, 0.01, 0.005]:
        e = lo + f * w
        xs = simulate(S64, e, S64.x_seed, 60000 + 400 * p)
        x, mu, res, _ = orbit_newton(S, LD(e), p, xs.astype(LD), tol=1e-26)
        if res < 1e-12 and -1 < mu < 1:
            break
    else:
        return None
    a, xa, mua = LD(e), x, mu
    # walk: mu decreases with eta inside the window (from +1 at the saddle-node to -1 at doubling)
    if mua > 0:
        b = a
        step = LD(0.02 * w)
        while True:
            b = b + step
            xb, mub, resb, _ = orbit_newton(S, b, p, xa, tol=1e-26)
            if not (resb < 1e-12):
                step /= 2
                b = a
                continue
            if mub < 0:
                break
            a, xa, mua = b, xb, mub
        lo_, hi_, xlo = a, b, xa
    else:
        b = a
        step = LD(0.02 * w)
        while True:
            b = b - step
            xb, mub, resb, _ = orbit_newton(S, b, p, xa, tol=1e-26)
            if not (resb < 1e-12):
                step /= 2
                b = a
                continue
            if mub > 0:
                break
            a, xa, mua = b, xb, mub
        lo_, hi_, xlo = b, a, xb
    for _ in range(200):
        mid = (lo_ + hi_) / 2
        if mid == lo_ or mid == hi_:
            break
        xm, mum, _, _ = orbit_newton(S, mid, p, xlo, tol=1e-26)
        if mum > 0:
            lo_, xlo = mid, xm
        else:
            hi_ = mid
    return (lo_ + hi_) / 2


def main():
    W = json.load(open(f"{CACHE}/windows.json"))
    out = {}
    tower = W["tower"]
    s = []
    for r in tower:
        v = superstable("bal4", r["lo"], r["hi"], r["base"])
        s.append(v)
        print(f"bal4 p={r['base']} superstable eta = {float(v):.18f}", flush=True)
    # control: logistic period-3 tower, superstable parameters found the same way from known windows
    ratios = [float((s[i] - s[i - 1]) / (s[i + 1] - s[i])) for i in range(1, len(s) - 1)]
    out["bal4"] = dict(periods=[r["base"] for r in tower], superstable=[repr(float(v)) for v in s],
                       superstable_str=[np.format_float_positional(v, precision=30) for v in s], ratios=ratios)
    print("bal4 tripling ratios", ratios)
    # logistic: period-3 window starts at 1+sqrt(8); nested windows located by a coarse scan
    from compute_windows import runs
    import npmaps as nm

    def scan_log(lo, hi, n, burn, maxp, tol):
        e = np.linspace(lo, hi, n)
        x = np.full_like(e, 0.3)
        for _ in range(burn):
            x = e * x * (1 - x)
        H = np.empty((maxp + 64, n))
        for i in range(maxp + 64):
            x = e * x * (1 - x)
            H[i] = x
        per = np.zeros(n, int)
        for p in range(1, maxp + 1):
            m = (per == 0) & (np.abs(H[p:p + 64] - H[:64]).max(0) < tol)
            per[m] = p
        return e, per
    lo, hi = 3.8284, 3.8575
    sl = []
    for m in range(1, 5):
        p = 3 ** m
        e, per = scan_log(lo, hi, 200000, 40000 + 200 * p, min(2 * p * 8, 1300), 1e-9)
        cand = [r for r in runs(e, per) if r["base"] == p]
        cand.sort(key=lambda r: -r["n"])
        r = cand[0]
        v = superstable("logistic", r["lo"], r["hi"], p)
        sl.append(v)
        print(f"logistic p={p} window [{r['lo']:.12f},{r['hi']:.12f}] superstable {float(v):.16f}", flush=True)
        w = r["hi"] - r["lo"]
        lo, hi = r["lo"], r["hi"] + 1.5 * w
    rl = [float((sl[i] - sl[i - 1]) / (sl[i + 1] - sl[i])) for i in range(1, len(sl) - 1)]
    out["logistic"] = dict(superstable=[np.format_float_positional(v, precision=30) for v in sl], ratios=rl)
    print("logistic tripling ratios", rl)
    json.dump(out, open(f"{CACHE}/tripling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
