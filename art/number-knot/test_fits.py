"""Synthetic checks for fits.py: a planted helix must be found at its period and not at others,
shuffled labels must give ~chance, and a planted calendar circle must beat an order shuffle."""
import numpy as np

import fits as F

rng = np.random.default_rng(0)


def planted(N=1000, D=300, T=10, amp=10.0, noise=1.0):
    a = np.arange(N)
    dirs = np.linalg.qr(rng.standard_normal((D, 3)))[0]
    sig = np.stack([a / N * 2, amp * np.cos(2 * np.pi * a / T), amp * np.sin(2 * np.pi * a / T)], 1)
    return sig @ dirs.T + noise * rng.standard_normal((N, D)), a, dirs


def test_helix_period_recovered():
    X, a, _ = planted()
    Y, *_ = F.pca(X, 100)
    perms = np.array([rng.permutation(len(a)) for _ in range(50)])
    r2, null = F.helix_table(Y, a, perms)
    d = {T: r2[T] - r2["lin"] for T in F.PERIODS}
    assert d[10] > 0.3, d
    assert max(d[2], d[5], d[100]) < 0.02, d
    assert np.quantile(null[10], 0.99) < 0.02


def test_fourier_peak():
    X, a, _ = planted(T=5)
    Y, *_ = F.pca(X, 100)
    P = F.fourier_power(Y, a, detrend=True)
    assert np.argmax(P) == 1000 // 5


def test_helix_coords_circle():
    X, a, _ = planted(noise=0.1)
    Y, *_ = F.pca(X, 100)
    xyz, C, c0 = F.helix_coords(Y, a, 10)
    assert np.corrcoef(xyz[:, 1], np.cos(2 * np.pi * a / 10))[0, 1] > 0.99


def test_r2_perm_matches_refit():
    X, a, _ = planted(N=200)
    Y, *_ = F.pca(X, 50)
    p = rng.permutation(200)
    B = F.basis(a, 10)
    # refit with label a[j] attached to row p[j]
    a_sh = np.empty(200); a_sh[p] = a
    assert np.isclose(F.r2_perm(Y, B, p[None])[0], F.r2_fit(Y, F.basis(a_sh, 10)))


def test_calendar_circle():
    K, nt, D = 7, 20, 100
    ang = 2 * np.pi * np.arange(K) / K
    dirs = np.linalg.qr(rng.standard_normal((D, 2)))[0]
    lab = np.tile(np.arange(K), nt); tmpl = np.repeat(np.arange(nt), K)
    H = 3 * np.stack([np.cos(ang[lab]), np.sin(ang[lab])], 1) @ dirs.T + rng.standard_normal((K * nt, D))
    s = F.calendar_scores(H, lab, tmpl, K)
    assert s["cyclic"] and s["r2_ho"] > 0.6, s["r2_ho"]
    bad = F.calendar_scores(H, lab, tmpl, K, order=np.array([0, 3, 6, 2, 5, 1, 4]))
    assert bad["r2_in"] < s["r2_in"] - 0.3 and not bad["cyclic"]


def test_mod_basis_contains_circles():
    X, a, _ = planted(T=10)
    Y, *_ = F.pca(X, 100)
    r_mod = F.r2_fit(Y, F.basis_mod(a, 10))
    assert r_mod >= F.r2_fit(Y, F.basis(a, 10)) - 1e-9
    assert F.basis_mod(a, 10).shape[1] == 11 and F.basis_poly3(a).shape[1] == 4
