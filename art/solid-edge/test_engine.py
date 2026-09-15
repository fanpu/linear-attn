"""CPU tests for se_engine (float64, uncompiled).  OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_engine.py"""
import math
import sys

import numpy as np
import torch

import se_engine as se

sys.path.insert(0, '/home/fzeng/ml/research/art/trainability-fractal')
import tfractal as tf  # noqa: E402

D64 = torch.float64


def _loglr(P, g, lo=-3, hi=6):
    return torch.pow(10.0, lo + (hi - lo) * torch.rand(P, generator=g, dtype=D64))


def test_net2_matches_source_trainer():
    g = torch.Generator().manual_seed(1)
    P = 300
    l0, l1 = _loglr(P, g), torch.pow(10.0, 1.0 + 2.0 * torch.rand(P, generator=g, dtype=D64))
    sig = torch.pow(10.0, -2 + 4 * torch.rand(P, generator=g, dtype=D64))
    pr_src = tf.make_problem(0, device='cpu')
    ref = tf.train_chunk(pr_src, l0, l1, steps=150, sigma0=sig, sigma1=sig, compiled=False)['measure'].numpy()
    pr = se.make_problem('net2', device='cpu', dtype=D64)
    out = se.train_chunk(pr, [l0, l1], sigma=sig, steps=150, compiled=False, bucket_min=8)
    m = out['measure'].numpy()
    assert (np.sign(m) == np.sign(ref)).all()
    assert np.allclose(m, ref, rtol=1e-9, atol=0)
    assert len(out['shapes']) > 1, 'compaction path not exercised'


def _autograd_check(kind):
    pr = se.make_problem(kind, device='cpu', dtype=D64)
    P = 3
    g = torch.Generator().manual_seed(2)
    Ws = [W.expand(P, *W.shape).clone() + 0.1 * torch.randn(P, *W.shape, generator=g, dtype=D64) for W in pr['Ws']]
    lg, _ = se.get_step(kind, 16, compiled=False)
    loss, gs = lg(Ws, pr['data'])
    Wr = [W.clone().requires_grad_(True) for W in Ws]
    if kind == 'net3':
        X, Y = pr['data']
        a0 = 1 / math.sqrt(16)
        h0 = torch.tanh(math.sqrt(2) * a0 * X @ Wr[0])
        h1 = torch.tanh(math.sqrt(2) * a0 * h0 @ Wr[1])
        L = ((h1 @ Wr[2] / 16 - Y) ** 2).mean(dim=(1, 2))
    else:
        F0, F1, F2, Y = pr['data']
        L = ((F0 @ Wr[0] + F1 @ Wr[1] + F2 @ Wr[2] - Y) ** 2).mean(dim=(1, 2))
    L.sum().backward()
    assert torch.allclose(loss, L.detach(), rtol=1e-12)
    for a, b in zip(gs, Wr):
        assert torch.allclose(a, b.grad, rtol=1e-10, atol=1e-14)


def test_net3_grad():
    _autograd_check('net3')


def test_quad3_grad():
    _autograd_check('quad3')


def test_compaction_is_label_exact_net3():
    g = torch.Generator().manual_seed(3)
    P = 200
    lrs = [_loglr(P, g, -1, 4) for _ in range(3)]
    pr = se.make_problem('net3', device='cpu', dtype=D64)
    a = se.train_chunk(pr, lrs, steps=100, compiled=False, bucket_min=8)
    b = se.train_chunk(pr, lrs, steps=100, compiled=False, early_exit=False)
    ma, mb = a['measure'].numpy(), b['measure'].numpy()
    assert len(a['shapes']) > 1
    assert (np.sign(ma) == np.sign(mb)).all()
    # frozen rows differ only in the negligible 1/v tail (<= 1e-6 per remaining step)
    assert np.allclose(ma, mb, rtol=1e-3)


def test_overview_axis_matches_log_grid():
    hx, hy = tf.log_grid(1.5, 1.5, 4.5, 1024, dev='cpu')
    vals, pix = se.overview_lr_axis(64)
    row = hx.view(1024, 1024)[0]
    assert torch.equal(vals, row[torch.as_tensor(pix)])
    assert torch.equal(hy.view(1024, 1024)[:, 0][torch.as_tensor(pix)], vals)
    assert pix[0] == 8 and pix[-1] == 1016


def test_sigma_plane_is_one():
    s, lg, k1 = se.sigma_axis(64)
    assert k1 == 26 and s[k1].item() == 1.0
    assert abs(lg[0] + 2.4375) < 1e-12


def test_quad2_matches_source_null():
    g = torch.Generator().manual_seed(4)
    P = 300
    l0, l1 = _loglr(P, g, -1, 3), _loglr(P, g, 0, 4)
    ref = tf.train_chunk_quadratic(tf.make_problem(0, nonlin='quadratic', device='cpu'), l0, l1, steps=200)['measure'].numpy()
    pr = se.make_problem('quad2', device='cpu', dtype=D64)
    m = se.train_chunk(pr, [l0, l1], steps=200, compiled=False, bucket_min=8)['measure'].numpy()
    assert (np.sign(m) == np.sign(ref)).all()
    conv = ref < 0
    assert np.allclose(m[conv], ref[conv], rtol=1e-9)


def test_overflow_is_diverged_in_float32():
    """sigma = 10^3.3 makes l0 > 1e6; a run that overflows float32 must be labelled diverged,
    matching float64 (the M2 measure fix)."""
    P = 64
    lr0 = torch.full((P,), 1e4, dtype=D64); lr1 = torch.pow(10.0, torch.linspace(-3, 6, P, dtype=D64))
    sig = torch.full((P,), 10 ** 3.3, dtype=D64)
    out = {}
    for dt in (torch.float32, D64):
        pr = se.make_problem('quad2', device='cpu', dtype=dt)
        out[dt] = se.train_chunk(pr, [lr0, lr1], sigma=sig, steps=500, compiled=False, bucket_min=8)['measure'].numpy()
    assert (np.sign(out[torch.float32]) == np.sign(out[D64])).mean() > 0.95
