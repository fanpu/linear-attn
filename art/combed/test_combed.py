"""Tests for combed_common. Run: art/.venv/bin/python -m pytest -q art/combed/test_combed.py"""
import math

import numpy as np
import torch

import combed_common as C


def test_training_sets_are_nested_prefixes_and_on_the_knot():
    big = C.training_set(4096)
    for n in C.NS:
        assert np.array_equal(C.training_set(n), big[:n])
    # every point satisfies the trefoil parametrisation for some s: check radius bounds
    assert np.all(np.abs(big[:, 2]) <= 1 / 3 + 1e-12)


def test_closed_form_matches_eq6_direct_sum():
    data = torch.tensor(C.training_set(64), dtype=torch.float64)
    x = torch.randn(10, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    t = 0.7
    v = C.closed_form_velocity(x, t, data)
    logits = -((x[:, None] - t * data[None]) ** 2).sum(-1) / (2 * (1 - t) ** 2)
    w = torch.softmax(logits, 1)
    v_ref = ((data[None] - x[:, None]) * w[..., None]).sum(1) / (1 - t)
    assert torch.allclose(v, v_ref, rtol=1e-10, atol=1e-10)


def test_closed_form_satisfies_continuity_equation():
    """d p_t / dt + div(p_t v*) = 0 by central finite differences at 20 random (x, t), to 1e-4 relative."""
    rng = np.random.default_rng(123)
    data = C.training_set(16)
    data_t = torch.tensor(data, dtype=torch.float64)
    h = 1e-5

    def v(x, t):
        return C.closed_form_velocity(torch.tensor(x, dtype=torch.float64), t, data_t).numpy()

    worst = 0.0
    for _ in range(20):
        t = rng.uniform(0.05, 0.95)
        i = rng.integers(len(data))
        x = t * data[i] + (1 - t) * rng.standard_normal(3)  # a point where p_t is not negligible
        x = x[None]
        dpdt = (C.mixture_density(x, t + h, data) - C.mixture_density(x, t - h, data))[0] / (2 * h)
        div = 0.0
        for a in range(3):
            e = np.zeros((1, 3)); e[0, a] = h
            fp = C.mixture_density(x + e, t, data)[0] * v(x + e, t)[0, a]
            fm = C.mixture_density(x - e, t, data)[0] * v(x - e, t)[0, a]
            div += (fp - fm) / (2 * h)
        rel = abs(dpdt + div) / max(abs(dpdt), abs(div))
        worst = max(worst, rel)
        assert rel < 1e-4, (t, x, dpdt, div, rel)
    print("worst relative continuity residual", worst)


def test_rk4_exact_on_a_conditional_straight_line():
    # With one training point the flow is x_t = (1-t) x0 + t x1 exactly; RK4 reproduces linear solutions.
    x1 = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)
    x0 = torch.randn(5, 3, dtype=torch.float64)
    xT, _ = C.rk4(lambda x, t: C.closed_form_velocity(x, t, x1), x0, 64)
    assert torch.allclose(xT, (1 - C.T_STOP) * x0 + C.T_STOP * x1, atol=1e-12)


def test_keep_indices_and_memorisation_criterion():
    assert len(C.keep_indices(256)) == 64 and len(C.keep_indices(512)) == 64
    data = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 5.0, 0]])
    pts = np.array([[0.2, 0, 0], [0.3, 0, 0], [0.26, 0, 0]])  # d1/d2 = 0.25, 0.43, 0.35
    assert C.memorised(pts, data).tolist() == [True, False, False]


def test_mlp_shapes():
    m = C.VelocityMLP()
    assert m(torch.zeros(7, 3), torch.rand(7)).shape == (7, 3)
    n_hidden = sum(isinstance(l, torch.nn.Linear) for l in m.net) - 1
    assert n_hidden == 4


def test_tail_grid_is_geometric_and_rk4_grid_exact_on_line():
    ts = C.tail_grid(96)
    assert abs(ts[0] - (1 - 1e-3)) < 1e-15 and abs(ts[-1] - (1 - 1e-6)) < 1e-15
    r = (1 - ts[1:]) / (1 - ts[:-1])
    assert np.allclose(r, r[0])
    x1 = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)
    x0 = torch.randn(5, 3, dtype=torch.float64)
    xa = (1 - ts[0]) * x0 + ts[0] * x1
    xb = C.rk4_grid(lambda x, t: C.closed_form_velocity(x, t, x1), xa, ts)
    assert torch.allclose(xb, (1 - ts[-1]) * x0 + ts[-1] * x1, atol=1e-12)
