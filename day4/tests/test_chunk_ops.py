"""GPU tests: fla's chunked kernels against your references, at today's shapes. [AI harness]

after fla/tests/ops/test_delta.py: parameterized over (B, T, H, D, dtype),
`torch.testing.assert_close` with explicit tolerances, output and gradient
checks separate, plus a bitwise determinism test for each kernel.
Tolerances: bf16 has 8 bits of mantissa (unit roundoff 2^-8 = 3.9e-3); the
chunked algorithm sums up to T terms, so we allow 2e-2 absolute on outputs
of order 1, about five units, and 2e-2 relative on the state.
"""
import pytest
import torch

from testbed.ops.delta_rule.chunk import chunk_delta_rule
from testbed.ops.linear_attn.chunk import chunk_linear_attn

from .adapters import run_naive_delta_rule, run_naive_linear_attn

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SHAPES = [(2, 512, 2, 32), (2, 512, 2, 64), (1, 512, 2, 128), (1, 512, 2, 256)]   # (B, T, H, D): the four per-head dims trained today


def inputs(B, T, H, D, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, device="cuda"), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, device="cuda").to(dtype)
    beta = torch.rand(B, T, H, device="cuda").to(dtype)
    return q, k, v, beta


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("B,T,H,D", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_linear_attn_matches_naive(B, T, H, D, dtype):
    q, k, v, _ = inputs(B, T, H, D, dtype)
    o_ref, S_ref = run_naive_linear_attn(q, k, v, None)
    o, S = chunk_linear_attn(q, k, v, None)
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(S.float(), S_ref.float(), atol=2e-2, rtol=2e-2)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("B,T,H,D", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_delta_rule_matches_naive(B, T, H, D, dtype):
    q, k, v, beta = inputs(B, T, H, D, dtype)
    o_ref, S_ref = run_naive_delta_rule(q, k, v, beta, None)
    o, S = chunk_delta_rule(q, k, v, beta, None)
    torch.testing.assert_close(o.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(S.float(), S_ref.float(), atol=2e-2, rtol=2e-2)


@gpu
@pytest.mark.gpu
def test_chunk_delta_rule_grads_match_naive():
    """Gradients w.r.t. q, k, v, beta, each checked separately (fp32 inputs so the
    reference is not the bottleneck; fp32 inside Triton runs as TF32, so 1e-2)."""
    q, k, v, beta = (t.float().requires_grad_(True) for t in inputs(1, 256, 2, 64, torch.float32))
    o, _ = chunk_delta_rule(q, k, v, beta, None)
    grads = torch.autograd.grad(o.square().sum(), (q, k, v, beta))
    q2, k2, v2, b2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v, beta))
    o2, _ = run_naive_delta_rule(q2, k2, v2, b2, None)
    grads_ref = torch.autograd.grad(o2.square().sum(), (q2, k2, v2, b2))
    for name, g, gr in zip("qkv beta".split(), grads, grads_ref):
        torch.testing.assert_close(g, gr, atol=1e-2, rtol=1e-2, msg=f"grad {name}")


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("op", ["linear", "delta"])
def test_chunk_ops_bitwise_deterministic(op):
    """The same inputs twenty times must give bit-identical outputs (the Day 1 rule:
    a race in a kernel shows up here before it shows up as a wrong loss curve)."""
    q, k, v, beta = inputs(2, 512, 2, 64, torch.bfloat16)
    fn = (lambda: chunk_linear_attn(q, k, v, None)[0]) if op == "linear" else (lambda: chunk_delta_rule(q, k, v, beta, None)[0])
    first = fn().clone()
    for _ in range(20):
        assert torch.equal(fn(), first)
