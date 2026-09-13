"""Checks for dd_core: closed forms agree with each other and with simulation.   python test_core.py"""
import numpy as np
from scipy import integrate, optimize

import dd_core as C


def test_stieltjes_vs_density():
    for g in [0.3, 0.8, 1.5, 4.0]:
        for lam in [0.05, 0.5, 2.0]:
            a, b = C.mp_edges(g)
            cont = integrate.quad(lambda s: C.mp_density(s, g) / (s + lam), a, b, limit=400)[0]
            cont2 = integrate.quad(lambda s: C.mp_density(s, g) / (s + lam) ** 2, a, b, limit=400)[0]
            atom = max(0.0, 1 - 1 / g)
            m, dm = C.mp_stieltjes_neg(lam, g)
            assert abs(m - (cont + atom / lam)) < 1e-6, (g, lam, m, cont + atom / lam)
            assert abs(dm - (cont2 + atom / lam ** 2)) < 1e-5, (g, lam, dm, cont2 + atom / lam ** 2)
    print("ok  Stieltjes transform and derivative match integrals of the MP density")


def test_ridge_to_ridgeless():
    for g in [0.2, 0.7, 1.3, 5.0]:
        r0 = C.ridgeless_risk(g, 1.0, 0.5)
        rl = C.ridge_risk(g, 1e-7, 1.0, 0.5)
        assert np.allclose(r0, rl, rtol=1e-4), (g, r0, rl)
    print("ok  ridge risk -> ridgeless closed form as lam -> 0")


def test_general_matches_isotropic():
    p = 500
    for g in [0.5, 2.0]:
        for lam in [0.0, 0.1, 1.0]:
            got = C.general_ridge_risk(np.ones(p), np.full(p, 1.0 / p), g, lam, sigma2=0.5)
            want = C.ridgeless_risk(g, 1.0, 0.5) if lam == 0 else C.ridge_risk(g, lam, 1.0, 0.5)
            assert np.allclose(got, want, rtol=1e-6, atol=1e-9), (g, lam, got, want)
    print("ok  general-covariance deterministic equivalent reduces to the isotropic formulas")


def test_optimal_lambda():
    for g in [0.5, 1.0, 3.0]:
        for snr in [0.5, 4.0]:
            f = lambda ll: C.ridge_risk(g, np.exp(ll), snr, 1.0)[0]
            res = optimize.minimize_scalar(f, bounds=(-10, 5), method="bounded", options=dict(xatol=1e-9))
            assert abs(np.exp(res.x) - C.optimal_lambda(g, snr, 1.0)) < 1e-3 * C.optimal_lambda(g, snr, 1.0), (g, snr, np.exp(res.x))
    print("ok  lam* = sigma^2 gamma / r^2 minimizes the limiting ridge risk")


def test_simulation_matches():
    rng = np.random.default_rng(0)
    n = 400
    for g, lam in [(0.5, 0.0), (2.0, 0.0), (0.9, 0.2), (1.0, 0.5), (3.0, 1.0)]:
        p = int(g * n)
        sims = [C.simulate_linear(n, p, [lam], r2=1.0, sigma2=0.25, rng=rng) for _ in range(40)]
        risk = np.mean([s["risk"][0] for s in sims])
        cond = np.mean([s["bias"][0] + s["var"][0] for s in sims])
        want = (C.ridgeless_risk(g, 1.0, 0.25) if lam == 0 else C.ridge_risk(g, lam, 1.0, 0.25))[0]
        se = np.std([s["risk"][0] for s in sims]) / np.sqrt(len(sims))
        assert abs(cond - want) < 0.05 * want + 0.01, (g, lam, cond, want)
        assert abs(risk - want) < 4 * se + 0.02 * want, (g, lam, risk, want, se)
        print(f"    gamma={g} lam={lam}: sim {risk:.4f} (cond {cond:.4f}) +- {se:.4f}  theory {want:.4f}")
    print("ok  simulated ridge / ridgeless risk matches closed form (n=400)")


def test_anisotropic_simulation():
    rng = np.random.default_rng(1)
    p = 300
    e = np.arange(1, p + 1, dtype=float) ** -1.0
    e *= p / e.sum()
    beta = rng.standard_normal(p) / np.sqrt(p)
    for g, lam in [(0.5, 0.0), (2.0, 0.0), (1.2, 0.05)]:
        n = int(p / g)
        sims = [C.simulate_linear(n, p, [lam], sigma2=0.25, rng=rng, cov_sqrt=np.sqrt(e), beta=beta) for _ in range(40)]
        cond = np.mean([s["bias"][0] + s["var"][0] for s in sims])
        want = C.general_ridge_risk(e, beta ** 2, g, lam, sigma2=0.25)[0]
        assert abs(cond - want) < 0.08 * want, (g, lam, cond, want)
        print(f"    aniso gamma={g} lam={lam}: sim {cond:.4f} theory {want:.4f}")
    print("ok  anisotropic deterministic equivalent matches simulation (p=300)")


def test_mm_rf():
    zeta = C.RELU_MU["mu1"] / C.RELU_MU["mustar"]
    # ridgeless theorem 3 vs definition 1 at tiny lambda
    for p1 in [1.0, 2.5, 6.0]:
        Bl, Vl = C.mm_bias_var_ridgeless(p1, 3.0, zeta)
        B, V = C.mm_bias_var(p1, 3.0, 1e-7, zeta)
        assert abs(Bl - B) < 1e-3 * abs(Bl) + 1e-4 and abs(Vl - V) < 1e-3 * abs(Vl) + 1e-4, (p1, Bl, B, Vl, V)
    # simulation (d = 100, n = 300, lam = 1e-3, tau2 = 0.5)
    rng = np.random.default_rng(5)
    for N in [100, 300, 900]:
        sims = [C.simulate_rf(100, 300, N, [1e-3], F1sq=1.0, tau2=0.5, rng=rng)[0] for _ in range(8)]
        th = C.mm_risk(N / 100, 3.0, 1e-3, F1sq=1.0, tau2=0.5)[0]
        se = np.std(sims) / np.sqrt(len(sims))
        print(f"    RF N={N}: sim {np.mean(sims):.3f} +- {se:.3f}  theory {th:.3f}")
        assert abs(np.mean(sims) - th) < 0.1 * th + 3 * se
    print("ok  Mei-Montanari ridgeless limit consistent; RF simulation matches asymptotics (d=100)")


if __name__ == "__main__":
    test_stieltjes_vs_density()
    test_ridge_to_ridgeless()
    test_general_matches_isotropic()
    test_optimal_lambda()
    test_simulation_matches()
    test_anisotropic_simulation()
    test_parts_match_full()
    test_mm_rf()


def test_parts_match_full():
    for n, p in [(50, 30), (50, 120)]:
        a, _ = C.simulate_linear_parts(n, p, [0.0, 0.3], rng=7)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n, p)); beta = rng.standard_normal(p); beta /= np.linalg.norm(beta); eps = rng.standard_normal(n)
        for i, lam in enumerate([0.0, 0.3]):
            for sigma in [0.0, 0.7]:
                y = X @ beta + sigma * eps
                bh = np.linalg.pinv(X) @ y if lam == 0 else np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)
                want = np.sum((bh - beta) ** 2)
                got = a[i, 0] + 2 * sigma * a[i, 1] + sigma ** 2 * a[i, 2]
                assert abs(got - want) < 1e-8 * max(1, want), (n, p, lam, sigma, got, want)
    print("ok  decomposed simulation reproduces direct pinv / ridge solutions to 1e-8")
