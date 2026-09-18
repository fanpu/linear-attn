"""Tests for the additive linear-attention reference. [AI harness; you fill the adapter]

after fla/tests/ops/test_linear_attn.py in shape (parameterized over shapes
and dtypes, assert_close with stated tolerances); the oracles here are
hand-traced numbers and algebraic properties, not another implementation.
"""
import pytest
import torch

from .adapters import run_naive_linear_attn

# The two-write example of the handout, per head: k1 = (1, 0), k2 = (0.6, 0.8),
# v1 = (1, 0), v2 = (0, 1); a third step queries k1 again and writes a zero value.
K1, K2 = [1.0, 0.0], [0.6, 0.8]
V1, V2, V0 = [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]


def two_write_inputs(dtype=torch.float32):
    q = torch.tensor([[K1, K2, K1]], dtype=dtype).reshape(1, 3, 1, 2)
    k = torch.tensor([[K1, K2, K1]], dtype=dtype).reshape(1, 3, 1, 2)
    v = torch.tensor([[V1, V2, V0]], dtype=dtype).reshape(1, 3, 1, 2)
    return q, k, v


def test_naive_linear_attn_two_writes():
    """With scale = 1: o_1 = (1, 0); o_2 = (0.6, 1.0) (interference 0.6 = k1·k2 on the first
    coordinate); o_3, querying k1 after both writes, = (1, 0.6). Final S = [[1, 0.6], [0, 0.8]]."""
    q, k, v = two_write_inputs()
    o, S = run_naive_linear_attn(q, k, v, 1.0)
    expect = torch.tensor([[1.0, 0.0], [0.6, 1.0], [1.0, 0.6]]).reshape(1, 3, 1, 2)
    torch.testing.assert_close(o, expect, atol=1e-6, rtol=0)
    torch.testing.assert_close(S, torch.tensor([[1.0, 0.6], [0.0, 0.8]]).reshape(1, 1, 2, 2), atol=1e-6, rtol=0)


def test_naive_linear_attn_default_scale():
    """scale=None means K ** -0.5: outputs shrink by 1/sqrt(2) here."""
    q, k, v = two_write_inputs()
    o_unit, _ = run_naive_linear_attn(q, k, v, 1.0)
    o_def, _ = run_naive_linear_attn(q, k, v, None)
    torch.testing.assert_close(o_def, o_unit / 2 ** 0.5, atol=1e-6, rtol=0)


@pytest.mark.parametrize("B,T,H,D", [(2, 16, 2, 8), (1, 33, 3, 4)])
def test_naive_linear_attn_orthonormal_keys_retrieve_exactly(B, T, H, D):
    """Writes with mutually orthonormal keys do not interfere: querying k_t at t returns
    v_t exactly (scale = 1), for every t up to D writes per head."""
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(B, H, D, D))          # orthonormal columns per (b, h)
    n = min(T, D)
    k = torch.zeros(B, T, H, D)
    k[:, :n] = Q[:, :, :, :n].permute(0, 3, 1, 2)             # k_t = column t
    v = torch.randn(B, T, H, D)
    o, _ = run_naive_linear_attn(k, k, v, 1.0)
    torch.testing.assert_close(o[:, :n], v[:, :n], atol=1e-5, rtol=1e-5)


def test_naive_linear_attn_is_causal():
    """Changing inputs after position t0 must not change outputs at or before t0."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 12, 2, 4) for _ in range(3))
    o1, _ = run_naive_linear_attn(q, k, v, None)
    k2, v2 = k.clone(), v.clone()
    k2[:, 7:], v2[:, 7:] = torch.randn_like(k2[:, 7:]), torch.randn_like(v2[:, 7:])
    o2, _ = run_naive_linear_attn(q, k2, v2, None)
    torch.testing.assert_close(o1[:, :7], o2[:, :7], atol=0, rtol=0)
    assert not torch.allclose(o1[:, 7:], o2[:, 7:])


def test_naive_linear_attn_dtypes():
    """bf16 inputs give a bf16 output and an fp32 state of shape [B, H, K, V]."""
    q, k, v = (torch.randn(2, 8, 2, 4).to(torch.bfloat16) for _ in range(3))
    o, S = run_naive_linear_attn(q, k, v, None)
    assert o.dtype == torch.bfloat16 and o.shape == (2, 8, 2, 4)
    assert S.dtype == torch.float32 and S.shape == (2, 2, 4, 4)
