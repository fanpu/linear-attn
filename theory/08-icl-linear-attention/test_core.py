"""Small tests for icl_core closed forms.  Run: .venv/bin/python 08-icl-linear-attention/test_core.py"""
import math
import torch
from icl_core import *

torch.manual_seed(0)
D = torch.float64


def close(a, b, tol, msg):
    a, b = float(a), float(b)
    ok = abs(a - b) <= tol * max(1.0, abs(b))
    print(f"[{'ok' if ok else 'FAIL'}] {msg}: {a:.6g} vs {b:.6g}")
    assert ok


def test_rls_equals_ridge():
    X, y, *_ = sample_prompts(64, 30, 8, sigma=0.5)
    W = rls_prefix_w(X, y, lam=0.25)
    for t in [1, 5, 8, 17, 30]:
        wr = ridge_w(X[:, :t], y[:, :t], 0.25)
        close((W[:, t] - wr).abs().max(), 0.0, 1e-9, f"RLS == ridge on prefix {t}")


def test_lsa_at_zfb_optimum_is_precond_gd():
    d, N, M = 6, 20, 13
    Lam = make_cov("ar1", d, 0.7)
    Wpv, Wkq = zfb_global_min(Lam, N)
    X, y, xq, yq, w = sample_prompts(128, M, d, Lam)
    yh = lsa_predict(embed(X, y, xq), Wpv, Wkq)
    Gi = torch.linalg.inv(zfb_gamma(Lam, N))
    ref = torch.einsum("bd,de,be->b", xq, Gi, torch.einsum("bmd,bm->bd", X, y) / M)
    close((yh - ref).abs().max(), 0.0, 1e-10, "LSA(W*) prediction == x_q^T Gamma^-1 (1/M) sum y x")
    Beff = effective_preconditioner(Wpv, Wkq, d)
    close((Beff - Gi).abs().max(), 0.0, 1e-12, "effective preconditioner of W* == Gamma^-1")


def test_risk_formula_monte_carlo():
    d, M = 5, 11
    Lam = make_cov("ar1", d, 0.6)
    Lq = 1.7 * make_cov("ar1", d, 0.3)
    A = torch.randn(d, d, dtype=D) * 0.3 + torch.eye(d, dtype=D)
    sigma = 0.4
    r_cf = risk_precond_gd1(A, Lam, M, sigma=sigma, Lam_q=Lq)
    acc, n = 0.0, 0
    for _ in range(20):
        X, y, _, _, w = sample_prompts(200_000, M, d, Lam, sigma=sigma)
        xq = torch.randn(200_000, d, dtype=D) @ torch.linalg.cholesky(Lq).T
        what = (torch.einsum("bmd,bm->bd", X, y) / M) @ A.T
        acc += (((what - w) * xq).sum(-1) ** 2).sum().item()
        n += 200_000
    close(acc / n, r_cf, 1e-2, "precond-GD1 risk: closed form vs Monte Carlo (4M prompts)")


def test_matches_zfb_theorem_4_2():
    d, N, M = 7, 15, 9
    Lam = make_cov("ar1", d, 0.5) * torch.linspace(0.5, 2, d, dtype=D).sqrt()[:, None] * torch.linspace(0.5, 2, d, dtype=D).sqrt()[None]
    G = zfb_gamma(Lam, N)
    Gi = torch.linalg.inv(G)
    Gi2 = Gi @ Gi
    tr = torch.trace
    trL = tr(Lam)
    # ZFB eq (4.6) specialised to noiseless w ~ N(0, I): error of best linear predictor = 0,
    # Sigma = |w|_Lam^2 Lam + Lam w w^T Lam,  a = w
    L2, L3 = Lam @ Lam, Lam @ Lam @ Lam
    t_M = (tr(Lam) * tr(Gi2 @ L2) + tr(L2 @ Gi2 @ Lam)) / M
    t_N = (tr(Gi2 @ L3) + 2 * trL * tr(Gi2 @ L2) + trL**2 * tr(Gi2 @ Lam)) / N**2
    close(risk_precond_gd1(Gi, Lam, M), t_M + t_N, 1e-10, "our closed form == ZFB Thm 4.2 (noiseless, w~N(0,I))")


def test_dmmse_concentrates():
    d, M = 4, 12
    tasks = torch.randn(16, d, dtype=D)
    X = torch.randn(32, M, d, dtype=D)
    idx = torch.randint(0, 16, (32,))
    y = torch.einsum("bmd,bd->bm", X, tasks[idx]) + 0.01 * torch.randn(32, M, dtype=D)
    W = dmmse_prefix_w(X, y, tasks, sigma=0.1)
    close((W[:, -1] - tasks[idx]).abs().max(), 0.0, 1e-6, "dMMSE posterior mean -> true pool task")
    close((W[:, 0] - tasks.mean(0)).abs().max(), 0.0, 1e-12, "dMMSE with empty context == pool mean")


def test_mp_mass():
    for g in [0.3, 1.0, 2.5]:
        s, wts, atom = mp_quadrature(g)
        close(wts.sum() + atom, 1.0, 2e-3, f"MP mass at gamma={g}")
        close((wts * s).sum(), 1.0, 2e-3, f"MP mean at gamma={g}")


if __name__ == "__main__":
    test_rls_equals_ridge()
    test_lsa_at_zfb_optimum_is_precond_gd()
    test_risk_formula_monte_carlo()
    test_matches_zfb_theorem_4_2()
    test_dmmse_concentrates()
    test_mp_mass()
    print("all tests passed")
