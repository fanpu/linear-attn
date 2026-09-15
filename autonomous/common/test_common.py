"""Correctness checks for common/. Run: ../.venv/bin/python -m pytest -q common/  (from autonomous/)

CPU tests check the references against each other in float64.
GPU tests check the fla kernels against the references in float32 (skipped without CUDA).
"""
import pytest
import torch

from recurrences import delta_par, delta_ref, fla_delta, fla_linear, linear_ref
from tasks import correlated_keys, icl_regression, mqar


def _inputs(B=2, T=37, H=3, D=8, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    kw = dict(generator=g, dtype=dtype, device=device)
    q, k, v = (torch.randn(B, T, H, D, **kw) for _ in range(3))
    k = k / k.norm(dim=-1, keepdim=True)  # delta rule is only stable for beta ||k||^2 < 2
    beta = torch.rand(B, T, H, **kw)
    log_a = -torch.rand(B, T, H, **kw) * 0.3
    return q, k, v, beta, log_a


def test_delta_with_beta0_is_decay_only():
    q, k, v, beta, log_a = _inputs()
    S0 = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    o, S = delta_ref(q, k, v, torch.zeros_like(beta), log_a, S0=S0)
    decay = log_a.sum(1).exp()[..., None, None]
    assert torch.allclose(S, S0 * decay, atol=1e-12)


def test_delta_single_step_writes_exactly():
    """With beta = 1 and unit keys, one write makes the memory return v exactly for that key."""
    q, k, v, beta, _ = _inputs(T=1)
    _, S = delta_ref(q, k, v, torch.ones_like(beta))
    assert torch.allclose(torch.einsum("bhvk,bhk->bhv", S, k[:, 0]), v[:, 0], atol=1e-12)


@pytest.mark.parametrize("gated", [False, True])
def test_delta_par_matches_ref(gated):
    q, k, v, beta, log_a = _inputs()
    la = log_a if gated else None
    o_ref, _ = delta_ref(q, k, v, beta, la)
    assert torch.allclose(delta_par(q, k, v, beta, la), o_ref, atol=1e-10)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel_err(x, ref):
    return ((x.double().cpu() - ref).abs().max() / ref.abs().max()).item()


# Measured float32 accuracy on the GB10 (IEEE matmul; see journal 2026-09-15): the recurrent kernel is ~1e-6 relative,
# the chunk kernel ~5e-5 (gated) to ~6e-4 (ungated) because of float32 triangular solves inside each chunk.
# With Triton's default TF32 the chunk kernel degrades to ~2e-3, which these bounds catch.
TOL = {"recurrent": 1e-5, "chunk": 1e-3}


@cuda
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_fla_delta_matches_ref(gated, mode):
    q, k, v, beta, log_a = _inputs(T=130, D=16)  # T > chunk size 64, so chunk boundaries are exercised
    la = log_a if gated else None
    o_ref, S_ref = delta_ref(q, k, v, beta, la)
    dev = [t.float().cuda() if t is not None else None for t in (q, k, v, beta, la)]
    o, S = fla_delta(*dev, mode=mode)
    assert _rel_err(o, o_ref) < TOL[mode] and _rel_err(S, S_ref) < TOL[mode]


@cuda
def test_triton_matmul_is_ieee():
    """The gated chunk kernel sits at ~5e-5 with IEEE float32; TF32 would put it near 2e-3."""
    q, k, v, beta, log_a = _inputs(B=2, T=128, H=2, D=64)
    o_ref, _ = delta_ref(q, k, v, beta, log_a)
    o, _ = fla_delta(*[t.float().cuda() for t in (q, k, v, beta, log_a)], mode="chunk")
    assert _rel_err(o, o_ref) < 3e-4


@cuda
@pytest.mark.parametrize("gated", [False, True])
def test_fla_linear_matches_ref(gated):
    q, k, v, _, log_a = _inputs(T=130, D=16)
    la = log_a if gated else None
    o_ref, S_ref = linear_ref(q, k, v, la)
    dev = [t.float().cuda() if t is not None else None for t in (q, k, v, la)]
    o, S = fla_linear(*dev)
    assert _rel_err(o, o_ref) < 1e-4 and _rel_err(S, S_ref) < 1e-4


def test_icl_regression_drift_keeps_task_norm_and_decorrelates():
    g = torch.Generator().manual_seed(0)
    q = 0.05
    x, y, w = icl_regression(4000, 60, 10, sigma=0.0, drift=q, gen=g)
    norms = (w**2).sum(-1).mean(0)  # E||w_t||^2 = 1 for all t
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.05)
    lag = 20  # E<w_t, w_{t+lag}> = (1-q)^{lag/2}
    corr = (w[:, 0] * w[:, lag]).sum(-1).mean()
    assert abs(corr - (1 - q) ** (lag / 2)) < 0.05
    assert torch.allclose(y, (w * x).sum(-1))


def test_mqar_targets_are_paired_values():
    g = torch.Generator().manual_seed(0)
    tok, tgt = mqar(8, n_pairs=10, n_queries=5, vocab=64, gen=g)
    for b in range(8):
        table = {int(tok[b, 2 * i]): int(tok[b, 2 * i + 1]) for i in range(10)}
        assert len(table) == 10
        for t in range(20, 25):
            assert table[int(tok[b, t])] == int(tgt[b, t])
    assert (tgt[:, :20] == -100).all()


def test_correlated_keys_mean_overlap():
    g = torch.Generator().manual_seed(0)
    k = correlated_keys(200, 30, 256, rho=0.3, gen=g)
    G = k @ k.transpose(1, 2)
    off = G[:, ~torch.eye(30, dtype=torch.bool)].mean()
    assert abs(off - 0.3) < 0.03 and torch.allclose(k.norm(dim=-1), torch.ones(200, 30, dtype=torch.float64))
