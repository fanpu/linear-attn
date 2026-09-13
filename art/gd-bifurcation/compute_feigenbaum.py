"""Locate period-doubling points eta_n by bisection and measure Feigenbaum delta.

Method A (Newton-Floquet bisection): the stable period-p orbit (p = 2^(n-1)) is found by
Newton on G^p(x) - x, continued in eta; its Floquet multiplier mu (most negative eigenvalue
of the p-fold Jacobian) crosses -1 exactly where period 2p is born.  eta_n is the root of
mu(eta) + 1, bracketed and bisected to ~1e-15 relative.  Done in float64 AND in numpy
longdouble (IEEE quad on aarch64) for the 1D maps; the difference is the float64 error.

Method B (simulation bisection): from the declared fixed init, run T steps, and ask
"is the recorded orbit NOT periodic with period p (tolerance tol)?".  A batch of eta values
is evaluated per round and the bracket shrinks by the batch size each round.  This is
what one would do with a black-box training loop; it is biased by critical slowing down
and is reported for the scalar objectives, the logistic map, and the deep linear network.

Output: cache/feigenbaum.json
"""
import json
import sys
import time
import numpy as np
import npmaps as nm

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"


# ------------------------------------------------------------------ systems (single state)
class OneD:
    def __init__(self, name, step, deriv, eta1, x_seed, dtype=np.float64):
        self.name, self._s, self._d, self.eta1, self.x_seed, self.dt = name, step, deriv, eta1, x_seed, dtype
        self.m = 1

    def step(self, x, e):
        return self._s(x, e)

    def jac(self, x, e):
        return np.array([[self._d(x[0], e)]], dtype=self.dt)


def make_1d(name, dtype=np.float64):
    D = dtype
    if name == "logistic":
        return OneD(name, lambda x, r: D(r) * x * (D(1) - x), lambda x, r: D(r) * (D(1) - D(2) * x), 3.0,
                    np.array([0.3], D), D)
    k = int(name[3:])
    one = D(1)
    return OneD(name,
                lambda u, e: u - D(e) * (u ** k - one) * u ** (k - 1),
                lambda u, e: one - D(e) * ((2 * k - 1) * u ** (2 * k - 2) - (k - 1) * u ** (k - 2)),
                2.0 / k, np.array([1.05], D), D)


class Prod:
    """Full k-dimensional GD on 1/2(prod x - 1)^2."""

    def __init__(self, k, x_seed):
        self.name = f"prod{k}"
        self.k = self.m = k
        self.eta1 = 2.0 / k
        self.x_seed = np.array(x_seed, float)
        self.dt = np.float64

    def step(self, x, e):
        return nm.prod_step(x[None], np.array([e]))[0]

    def jac(self, x, e):
        return nm.prod_jac(x[None], np.array([e]))[0]


# ------------------------------------------------------------------ method A
def orbit_newton(S, e, p, x0, iters=60, tol=None):
    x = x0.copy()
    one = S.dt(1)
    tol = tol if tol is not None else (1e-14 if S.dt == np.float64 else 1e-28)
    for it in range(iters):
        y = x.copy()
        J = np.eye(S.m, dtype=S.dt)
        for _ in range(p):
            J = S.jac(y, e) @ J
            y = S.step(y, e)
        F = y - x
        if S.m == 1:
            dx = -F / (J[0, 0] - one)
        else:
            dx = np.linalg.lstsq((J - np.eye(S.m)).astype(float), -F.astype(float), rcond=None)[0]
        x = x + dx
        if np.max(np.abs(dx)) < tol * max(1.0, float(np.max(np.abs(x)))):
            break
    # multiplier at converged orbit
    y = x.copy()
    J = np.eye(S.m, dtype=S.dt)
    orb = []
    for _ in range(p):
        orb.append(y.copy())
        J = S.jac(y, e) @ J
        y = S.step(y, e)
    res = float(np.max(np.abs(y - x)))
    if S.m == 1:
        mu = float(J[0, 0])
    else:
        ev = np.linalg.eigvals(J.astype(float))
        real = ev[np.abs(ev.imag) < 1e-9].real
        mu = float(real.min()) if len(real) else np.nan
    return x, mu, res, np.array(orb)


def simulate(S, e, x0, T):
    x = x0.copy()
    for _ in range(T):
        x = S.step(x, e)
    return x


def find_bifurcations(S, nmax=12, verbose=True):
    etas = [S.eta1]
    # bracket for eta_2: just above eta_1 the period-2 orbit is stable
    gap = 0.2 * S.eta1  # generous first gap
    for n in range(2, nmax + 1):
        p = 2 ** (n - 1)
        e_prev = etas[-1]
        if n == 2:
            e_a = e_prev + 0.1 * gap
        else:
            e_a = e_prev + 0.4 * (etas[-1] - etas[-2]) / 4.67
        # seed: simulate long from a perturbed seed, then Newton
        xs = simulate(S, e_a, S.x_seed, 200000 if p < 64 else 400000)
        x, mu, res, _ = orbit_newton(S, e_a, p, xs)
        if not (res < 1e-8 and -1 < mu < 1):
            if verbose:
                print(f"  n={n}: seed orbit failed at eta={e_a} (mu={mu}, res={res})")
            break
        # march upward with continuation until mu < -1
        step = (etas[-1] - etas[-2]) / 4.67 * 0.1 if n > 2 else 0.01 * gap
        e_lo, x_lo, mu_lo = e_a, x, mu
        e_hi = None
        for _ in range(400):
            e_try = e_lo + step
            xt, mut, rest, _ = orbit_newton(S, e_try, p, x_lo)
            if rest > 1e-8 or not np.isfinite(mut):
                step *= 0.5
                continue
            if mut < -1:
                e_hi = e_try
                break
            e_lo, x_lo, mu_lo = e_try, xt, mut
        if e_hi is None:
            if verbose:
                print(f"  n={n}: no crossing found above {e_lo} (mu={mu_lo})")
            break
        # bisection on mu + 1
        for _ in range(80):
            mid = 0.5 * (e_lo + e_hi)
            if mid == e_lo or mid == e_hi:
                break
            xm, mum, resm, _ = orbit_newton(S, mid, p, x_lo)
            if mum > -1:
                e_lo, x_lo = mid, xm
            else:
                e_hi = mid
        etas.append(0.5 * (e_lo + e_hi))
        if verbose:
            print(f"  {S.name} n={n} (period {p}->{2 * p}) eta_n={etas[-1]:.16g} bracket={e_hi - e_lo:.1e}", flush=True)
    return etas


def find_bifurcations_ld(name, eta64, nmax):
    """Re-bisect each eta_n in longdouble inside a small bracket around the float64 value."""
    S = make_1d(name, np.longdouble)
    out = [np.longdouble(S.eta1)]
    for n in range(2, min(nmax, len(eta64)) + 1):
        p = 2 ** (n - 1)
        gapn = eta64[n - 1] - eta64[n - 2]
        lo = np.longdouble(eta64[n - 1]) - np.longdouble(0.2 * gapn)
        hi = np.longdouble(eta64[n - 1]) + np.longdouble(0.05 * gapn)
        xs = simulate(make_1d(name), float(lo), make_1d(name).x_seed, 400000).astype(np.longdouble)
        x, mu, res, _ = orbit_newton(S, lo, p, xs)
        if not (-1 < mu < 1):
            out.append(np.longdouble(np.nan))
            continue
        _, muh, _, _ = orbit_newton(S, hi, p, x)
        if not muh < -1:
            out.append(np.longdouble(np.nan))
            continue
        for _ in range(120):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            xm, mum, _, _ = orbit_newton(S, mid, p, x)
            if mum > -1:
                lo, x = mid, xm
            else:
                hi = mid
        out.append((lo + hi) / 2)
        print(f"  [longdouble] {name} n={n} eta_n={float(out[-1]):.16g} diff64={float(out[-1]) - eta64[n - 1]:.2e}",
              flush=True)
    return out


def deltas(eta, sig):
    eta = np.asarray(eta, float)
    sig = np.asarray(sig, float)
    d, ds = [], []
    for i in range(1, len(eta) - 1):
        a = eta[i] - eta[i - 1]
        b = eta[i + 1] - eta[i]
        v = a / b
        # propagate independent errors on the three points
        g0 = -1 / b
        g1 = 1 / b + a / b ** 2
        g2 = -a / b ** 2
        s = np.sqrt((g0 * sig[i - 1]) ** 2 + (g1 * sig[i]) ** 2 + (g2 * sig[i + 1]) ** 2)
        d.append(v)
        ds.append(s)
    return d, ds


# ------------------------------------------------------------------ method B (simulation)
def sim_predicate_batch(stepfun, obsfun, x0, etas, p, T, W, tol):
    """True where the orbit is NOT p-periodic after T steps (checked over a window of W steps)."""
    x = stepfun.init(x0, len(etas))
    for _ in range(T):
        x = stepfun(x, etas)
    hist = []
    for _ in range(W + p):
        hist.append(obsfun(x))
        x = stepfun(x, etas)
    H = np.array(hist)
    dev = np.abs(H[p:] - H[:-p]).max(0)
    alive = np.isfinite(H).all(0)
    return (dev > tol) & alive, alive


def sim_bisect(stepfun, obsfun, x0, e_lo, e_hi, p, T, W, tol, rounds=3, B=64):
    for _ in range(rounds):
        grid = np.linspace(e_lo, e_hi, B)
        pred, alive = sim_predicate_batch(stepfun, obsfun, x0, grid, p, T, W, tol)
        idx = np.argmax(pred)
        if not pred.any() or idx == 0:
            return np.nan, np.nan
        e_lo, e_hi = grid[idx - 1], grid[idx]
    return 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)


class NpStep:
    def __init__(self, kind, k=None):
        self.kind, self.k = kind, k

    def init(self, x0, N):
        return np.broadcast_to(np.asarray(x0, float), (N, len(x0))).copy()

    def __call__(self, x, e):
        with np.errstate(all="ignore"):
            if self.kind == "logistic":
                return nm.logistic_step(x, e[:, None])
            return nm.prod_step(x, e)


def main():
    t0 = time.time()
    res = {}
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    # ---------------- method A
    for name in ["logistic", "bal2", "bal3", "bal4"]:
        if only and name not in only:
            continue
        print("== A", name, flush=True)
        S = make_1d(name)
        e64 = find_bifurcations(S, nmax=13)
        eld = find_bifurcations_ld(name, e64, nmax=len(e64))
        err = [abs(float(a) - b) if np.isfinite(float(a)) else np.nan for a, b in zip(eld, e64)]
        sig = [max(e, 1e-16 * b) for e, b in zip(err, e64)]
        d, ds = deltas(e64, sig)
        res[f"A_{name}"] = dict(eta=e64, eta_longdouble=[float(v) for v in eld], sigma=sig, delta=d, delta_sigma=ds)
        print("   delta:", np.round(d, 5), flush=True)
    for k, xs in [(2, (1.1, 0.9)), (4, (1.1, 0.9, 1.05, 0.95))]:
        name = f"prod{k}"
        if only and name not in only:
            continue
        print("== A", name, flush=True)
        S = Prod(k, xs)
        e = find_bifurcations(S, nmax=11)
        # compare to the balanced reduction: the float64 difference is the "is it 1D" test
        b = res.get(f"A_bal{k}", {}).get("eta")
        sig = [1e-15 * v for v in e]
        if b is not None:
            m = min(len(b), len(e))
            sig = [max(abs(e[i] - b[i]), 1e-15 * e[i]) for i in range(m)]
            e = e[:m]
        d, ds = deltas(e, sig)
        res[f"A_{name}"] = dict(eta=e, sigma=sig, delta=d, delta_sigma=ds)
        print("   delta:", np.round(d, 5), flush=True)
    json.dump(res, open(f"{CACHE}/feigenbaum_A.json", "w"), indent=1)

    # ---------------- method B (simulation bisection on the actual fixed-init training loop)
    if only and "B" not in only:
        print(f"done {time.time() - t0:.0f}s")
        return
    resB = {}
    for name, stepper, obs, x0, key in [
        ("logistic", NpStep("logistic"), lambda x: x[:, 0], (0.3,), "A_logistic"),
        ("prod2", NpStep("prod", 2), lambda x: np.prod(x, 1), (1.1, 0.9), "A_bal2"),
        ("prod4", NpStep("prod", 4), lambda x: np.prod(x, 1), (1.1, 0.9, 1.05, 0.95), "A_bal4"),
    ]:
        ref = res[key]["eta"]
        for T, tol in [(20000, 1e-6), (200000, 1e-9)]:
            out, outs = [], []
            for n in range(1, 8):
                p = 2 ** (n - 1)
                lo = ref[n - 1] - (ref[n - 1] - (ref[n - 2] if n > 1 else 0.8 * ref[0])) * 0.3
                hi = ref[n - 1] + (ref[n] - ref[n - 1]) * 0.3 if n < len(ref) else ref[n - 1] * 1.001
                e, s = sim_bisect(stepper, obs, x0, lo, hi, p, T, 4 * p + 64, tol)
                out.append(e)
                outs.append(s)
            d, ds = deltas(out, outs)
            resB[f"B_{name}_T{T}"] = dict(eta=out, halfwidth=outs, delta=d, delta_sigma=ds, T=T, tol=tol,
                                          bias_vs_A=[o - r for o, r in zip(out, ref)])
            print(f"== B {name} T={T}: eta_n - eta_n(A) =", np.array(resB[f'B_{name}_T{T}']['bias_vs_A']),
                  " delta", np.round(d, 3), flush=True)
    json.dump(resB, open(f"{CACHE}/feigenbaum_B.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
