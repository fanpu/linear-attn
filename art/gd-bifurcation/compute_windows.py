"""Find periodic windows inside the chaotic band of GD on 1/2(x1x2x3x4 - 1)^2.

Detection uses the exact balanced-line map u <- u - eta(u^4-1)u^3 (fast, 1D); every plate is later
recomputed with the full 4-coordinate GD (compute_plates.py).  The 1D map is valid here because the
balanced line is transversally attracting for eta < ~0.99 (measured lyap_trans < 0, see README).

A window = maximal eta-run where the orbit is periodic (period detected <= 256 at tol 1e-7 after 20000
burn-in steps), runs separated by < 3 grid cells merged if base periods divide.  Base period = smallest
period in the run.  Nested tower: inside the period-3 window's chaotic part, the widest period-9 window;
inside that, the widest period-27 window; and so on.

Output: cache/windows.json
"""
import json
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
K = 4


def scan(lo, hi, n, burn, maxp, tol, rec_extra=64):
    e = np.linspace(lo, hi, n)
    u = np.full_like(e, 1.05)
    with np.errstate(all="ignore"):
        for _ in range(burn):
            u = nm.bal_step(u, e, K)
        H = np.empty((maxp + rec_extra, n))
        lam = np.zeros(n)
        for i in range(maxp + rec_extra):
            lam += np.log(np.abs(nm.bal_deriv(u, e, K)) + 1e-300)
            u = nm.bal_step(u, e, K)
            H[i] = u
    lam /= maxp + rec_extra
    per = np.zeros(n, int)
    for p in range(1, maxp + 1):
        m = (per == 0) & (np.abs(H[p:p + rec_extra] - H[:rec_extra]).max(0) < tol)
        per[m] = p
    return e, per, lam, H


def runs(e, per, only=None):
    isp = per > 0 if only is None else np.isin(per, only)
    out = []
    i = 0
    n = len(e)
    while i < n:
        if isp[i]:
            j = i
            while j + 1 < n and isp[j + 1]:
                j += 1
            ps = per[i:j + 1]
            out.append(dict(lo=float(e[i]), hi=float(e[j]), n=int(j - i + 1), base=int(ps.min()),
                            periods=sorted(set(int(v) for v in ps))[:8]))
            i = j + 1
        else:
            i += 1
    # merge runs whose base periods divide and which are separated by <= 3 cells
    de = e[1] - e[0]
    merged = []
    for r in out:
        if merged and r["lo"] - merged[-1]["hi"] <= 3.5 * de and r["base"] % merged[-1]["base"] == 0:
            m = merged[-1]
            m["hi"] = r["hi"]
            m["n"] += r["n"]
            m["periods"] = sorted(set(m["periods"]) | set(r["periods"]))[:8]
        else:
            merged.append(r)
    return merged


def central_range(lo, hi, p, n=400):
    """Range of the branch nearest the critical point, phase aligned with period p, across [lo,hi]."""
    e = np.linspace(lo, hi, n)
    u = np.full_like(e, 1.05)
    for _ in range(20000):
        u = nm.bal_step(u, e, K)
    # critical point of the balanced map: eta(7u^6 - 3u^2) = 1  (u>0, Newton)
    uc = np.full_like(e, 0.9)
    for _ in range(60):
        f = e * (7 * uc ** 6 - 3 * uc ** 2) - 1
        df = e * (42 * uc ** 5 - 6 * uc)
        uc -= f / df
    H = []
    for _ in range(p * 64):
        H.append(u.copy())
        u = nm.bal_step(u, e, K)
    H = np.array(H)                                     # (T, n)
    t0 = np.argmin(np.abs(H[:p] - uc[None]), 0)         # phase closest to critical point
    vals = np.array([H[t0[i]::p, i] for i in range(n)])  # (n, 64)
    Pv = vals ** 4
    return float(np.nanmin(Pv)), float(np.nanmax(Pv)), float(np.median(uc ** 4))


def main():
    res = {}
    # ---------------- atlas: all windows in the main chaotic band
    eta_inf = 0.6599237962769293
    e, per, lam, _ = scan(eta_inf, 0.99, 400000, 20000, 256, 1e-7)
    W = runs(e, per)
    W.sort(key=lambda r: -r["n"])
    res["all_windows"] = W[:200]
    best = {}
    for r in W:
        b = r["base"]
        if b not in best and r["n"] >= 8:
            best[b] = r
    atlas = []
    for b in sorted(best):
        r = best[b]
        if b > 16:
            continue
        w = r["hi"] - r["lo"]
        ylo, yhi, pc = central_range(r["lo"] + 0.05 * w, r["hi"], b)
        atlas.append(dict(r, ylo=ylo, yhi=yhi, Pc=pc))
        print(f"atlas p={b}: [{r['lo']:.8f}, {r['hi']:.8f}] w={w:.2e} P-branch [{ylo:.5f},{yhi:.5f}] periods {r['periods']}", flush=True)
    res["atlas"] = atlas

    # ---------------- nested period-3 tower
    tower = []
    p3 = best[3]
    tower.append(dict(p3))
    lo, hi = p3["lo"], p3["hi"]
    for m in range(2, 6):
        p = 3 ** m
        w = hi - lo
        e, per, lam, _ = scan(lo, hi + 0.5 * w, 200000, 40000 + 200 * p, min(2 * p * 8, 4096), 1e-8, rec_extra=64)
        cand = [r for r in runs(e, per) if r["base"] == p]
        if not cand:
            print("tower stops at", p)
            break
        cand.sort(key=lambda r: -r["n"])
        r = cand[0]
        print(f"tower p={p}: [{r['lo']:.12f}, {r['hi']:.12f}] w={r['hi'] - r['lo']:.2e} n={r['n']} periods {r['periods']}", flush=True)
        tower.append(r)
        lo, hi = r["lo"], r["hi"]
    for r in tower:
        w = r["hi"] - r["lo"]
        r["ylo"], r["yhi"], r["Pc"] = central_range(r["lo"] + 0.05 * w, r["hi"], r["base"])
    res["tower"] = tower
    json.dump(res, open(f"{CACHE}/windows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
