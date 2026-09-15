"""Run from this dir: ../../.venv/bin/python -m pytest -q test_models.py"""
import math
import torch
import pytest
from models import DeltaNetWB, LinAttnWB, make_batch, spectrum, fit_exponent


def nlms(x, y, M, alpha, beta):
    """float64 reference in fla's gate order: predict y_t with w, then u = alpha w, w <- u + beta (y - u.x) M x / x^T M x."""
    B, T, d = x.shape
    w = torch.zeros(B, d, dtype=torch.float64)
    out = []
    for t in range(T):
        xt = x[:, t]
        out.append((w * xt).sum(-1))
        Mx = xt @ M.T
        u = alpha * w
        r = y[:, t] - (u * xt).sum(-1)
        w = u + beta * (r / (xt * Mx).sum(-1))[:, None] * Mx
    return torch.stack(out, 1)


@pytest.mark.parametrize("learn_alpha", [False, True])
def test_deltanet_wb_is_preconditioned_nlms_float64(learn_alpha):
    """Semantics: the interleaved-token gated delta rule equals preconditioned NLMS (fla gate order), exactly."""
    torch.manual_seed(0)
    d, T = 8, 70
    x, y, _ = make_batch(4, T, d, spectrum(d, 10), 0.05 if learn_alpha else 0.0, 0.1, "cpu")
    A0 = torch.randn(d, d) / math.sqrt(d) + torch.eye(d)
    m = DeltaNetWB(d, A0, learn_alpha=learn_alpha, alpha0=0.9, beta0=0.7, impl="ref").double()
    yh = m(x.double(), y.double()).detach()
    ref = nlms(x.double(), y.double(), m.metric(), 0.9 if learn_alpha else 1.0, 0.7)
    assert (yh - ref).abs().max() < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("learn_alpha", [False, True])
def test_deltanet_wb_fla_kernel_close(learn_alpha):
    """The float32 chunk kernel agrees with the float64 path to ~3e-3 absolute (unnormalized queries amplify its error)."""
    torch.manual_seed(0)
    d, T = 8, 70
    x, y, _ = make_batch(4, T, d, spectrum(d, 10), 0.05 if learn_alpha else 0.0, 0.1, "cpu")
    A0 = torch.randn(d, d) / math.sqrt(d) + torch.eye(d)
    ref = DeltaNetWB(d, A0, learn_alpha=learn_alpha, alpha0=0.9, beta0=0.7, impl="ref").double()(x.double(), y.double())
    yf = DeltaNetWB(d, A0, learn_alpha=learn_alpha, alpha0=0.9, beta0=0.7).cuda()(x.cuda(), y.cuda())
    assert (yf.double().cpu() - ref.detach()).abs().max() < 1e-2


def test_linattn_wb_matches_loop():
    torch.manual_seed(0)
    d, T = 6, 20
    x, y, _ = make_batch(3, T, d, spectrum(d, 10), 0.0, 0.0, "cpu")
    A0 = torch.randn(d, d)
    m = LinAttnWB(d, A0)
    yh = m(x, y).detach()
    M = A0.T @ A0
    for t in range(1, T):
        ref = sum(y[:, i, None] * x[:, i] for i in range(t)) @ M @ x[:, t].T if False else \
            torch.stack([sum(y[b, i] * x[b, i] @ M @ x[b, t] for i in range(t)) / t for b in range(3)])
        assert torch.allclose(yh[:, t], ref, rtol=1e-4, atol=1e-4)


def test_fit_exponent_recovers_power():
    lam = spectrum(32, 100)
    s, r2, off = fit_exponent(torch.diag(lam ** -0.37) * 3.0, lam)
    assert abs(s - 0.37) < 1e-6 and r2 > 0.999999 and off < 1e-9
