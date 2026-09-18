"""Tests for the delta-rule reference. [AI harness; you fill the adapter]

after fla/tests/ops/test_delta.py in shape; oracles are the hand-traced
example and algebraic properties (see test_naive_linear_attn.py).
"""
import pytest
import torch

from .adapters import run_naive_delta_rule, run_naive_linear_attn
from .test_naive_linear_attn import two_write_inputs


def test_naive_delta_rule_two_writes():
    """beta = (1, 1, 0), scale = 1: o_1 = (1, 0); o_2 = (0, 1), exact, because the write
    of k2 first subtracts what the state predicted for k2; o_3, querying k1 afterwards,
    = (0.64, 0.6): the second write disturbed the first by (k1·k2) times its error.
    Final S = [[0.64, 0.6], [-0.48, 0.8]]."""
    q, k, v = two_write_inputs()
    beta = torch.tensor([[1.0, 1.0, 0.0]]).reshape(1, 3, 1)
    o, S = run_naive_delta_rule(q, k, v, beta, 1.0)
    expect = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.64, 0.6]]).reshape(1, 3, 1, 2)
    torch.testing.assert_close(o, expect, atol=1e-6, rtol=0)
    torch.testing.assert_close(S, torch.tensor([[0.64, 0.6], [-0.48, 0.8]]).reshape(1, 1, 2, 2), atol=1e-6, rtol=0)


def test_naive_delta_rule_beta_zero_writes_nothing():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 10, 2, 4) for _ in range(3))
    o, S = run_naive_delta_rule(q, k, v, torch.zeros(2, 10, 2), None)
    assert torch.count_nonzero(o) == 0 and torch.count_nonzero(S) == 0


@pytest.mark.parametrize("B,T,H,D", [(2, 24, 2, 8), (1, 40, 1, 16)])
def test_naive_delta_rule_latest_write_is_exact(B, T, H, D):
    """With beta = 1 and unit-norm keys (not orthogonal), querying the key just written
    returns its value exactly, however many writes came before: the delta rule's
    guarantee. The additive rule does not have it."""
    torch.manual_seed(2)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D), dim=-1)
    v = torch.randn(B, T, H, D)
    o, _ = run_naive_delta_rule(k, k, v, torch.ones(B, T, H), 1.0)
    torch.testing.assert_close(o, v, atol=1e-4, rtol=1e-4)
    o_add, _ = run_naive_linear_attn(k, k, v, 1.0)
    assert (o_add - v).abs().max() > 0.1


def test_naive_delta_rule_equals_additive_on_orthonormal_keys():
    """When keys are orthonormal the state predicts zero for every new key, so the
    subtraction is a no-op and both rules coincide (beta = 1, scale = 1)."""
    torch.manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(1, 1, 8, 8))
    k = Q[0, 0].T.reshape(1, 8, 1, 8)
    v = torch.randn(1, 8, 1, 8)
    o_d, S_d = run_naive_delta_rule(k, k, v, torch.ones(1, 8, 1), 1.0)
    o_a, S_a = run_naive_linear_attn(k, k, v, 1.0)
    torch.testing.assert_close(o_d, o_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S_d, S_a, atol=1e-5, rtol=1e-5)


def test_naive_delta_rule_is_causal():
    torch.manual_seed(4)
    q, k, v = (torch.randn(2, 12, 2, 4) for _ in range(3))
    beta = torch.rand(2, 12, 2)
    o1, _ = run_naive_delta_rule(q, k, v, beta, None)
    k2, v2 = k.clone(), v.clone()
    k2[:, 7:], v2[:, 7:] = torch.randn_like(k2[:, 7:]), torch.randn_like(v2[:, 7:])
    o2, _ = run_naive_delta_rule(q, k2, v2, beta, None)
    torch.testing.assert_close(o1[:, :7], o2[:, :7], atol=0, rtol=0)


def test_naive_delta_rule_dtypes():
    q, k, v = (torch.randn(2, 8, 2, 4).to(torch.bfloat16) for _ in range(3))
    o, S = run_naive_delta_rule(q, k, v, torch.rand(2, 8, 2).to(torch.bfloat16), None)
    assert o.dtype == torch.bfloat16 and o.shape == (2, 8, 2, 4)
    assert S.dtype == torch.float32 and S.shape == (2, 2, 4, 4)
