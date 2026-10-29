"""Tests for the rescaling symmetry.  [AI-owned; do not edit]

Problem (noether): ψ_α (2), its generator X (1), and the conserved quantity Q.
Q is checked along accurate solutions of gradient flow for three losses that depend on (a, b)
only through ab, so any function of a² - b² that is not constant passes.
"""
import numpy as np
import pytest
from adapters import run_conserved, run_generator, run_rescale

from tdyn.integrate import solve_flow
from tdyn.models import anchor_hessian

RNG = np.random.default_rng(0)
PTS = RNG.uniform(-2, 2, size=(200, 2))

# Losses of the form ℓ(ab): their gradients are ℓ'(ab) (b, a).
X_DATA = np.random.default_rng(1).standard_normal(16)
Y_DATA = 0.7 * X_DATA + 0.3 * np.random.default_rng(2).standard_normal(16)
LOSS_DERIVS = {
    "squared, target 2": lambda p: p - 2.0,                                     # ½(p - 2)²
    "logistic": lambda p: -1.0 / (1.0 + np.exp(p)),                             # log(1 + e^{-p})
    "regression data": lambda p: np.mean(X_DATA * (p * X_DATA - Y_DATA)),       # ½ mean (p x_i - y_i)²
}


def _grad(dl):
    def g(th):
        a, b = th[..., 0], th[..., 1]
        d = np.vectorize(dl)(a * b)
        return np.stack([d * b, d * a], axis=-1)
    return g


def test_rescale_preserves_product_and_identity():
    for alpha in (0.3, 1.0, 2.5, -1.7):
        out = np.asarray(run_rescale(PTS, alpha))
        assert out.shape == PTS.shape
        np.testing.assert_allclose(out[:, 0] * out[:, 1], PTS[:, 0] * PTS[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(run_rescale(PTS, 1.0), PTS, rtol=0, atol=1e-15)


def test_rescale_group_law():
    lhs = np.asarray(run_rescale(run_rescale(PTS, 0.4), 3.0))
    np.testing.assert_allclose(lhs, run_rescale(PTS, 1.2), rtol=1e-12, atol=1e-12)


def test_generator_is_derivative_at_identity():
    h = 1e-6
    fd = (np.asarray(run_rescale(PTS, 1 + h)) - np.asarray(run_rescale(PTS, 1 - h))) / (2 * h)
    np.testing.assert_allclose(run_generator(PTS), fd, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("name", list(LOSS_DERIVS))
def test_generator_orthogonal_to_gradient(name):
    # The Noether constraint (Problem noether (a)): <∇L(θ), X(θ)> = 0 for every L = ℓ(ab).
    g = _grad(LOSS_DERIVS[name])(PTS)
    X = np.asarray(run_generator(PTS))
    dots = np.sum(g * X, axis=-1)
    scale = np.linalg.norm(g, axis=-1) * np.linalg.norm(X, axis=-1) + 1e-300
    assert np.max(np.abs(dots) / scale) < 1e-12


def test_conserved_along_flow():
    for name, dl in LOSS_DERIVS.items():
        g = _grad(dl)
        for th0 in ([1.3, 0.2], [-0.4, 1.1], [0.05, -0.9]):
            traj = solve_flow(lambda th: -g(th), th0, 8.0, n_points=81).y
            Q = np.asarray(run_conserved(traj), dtype=float)
            assert Q.shape == (81,)
            assert np.max(np.abs(Q - Q[0])) < 1e-8 * max(1.0, abs(Q[0])), name


def test_conserved_is_not_trivial():
    # Q must actually distinguish the curves a² - b² = const: its gradient is nonzero and
    # parallel to the generator (the normal of those curves).
    h = 1e-6
    for th in PTS[:20]:
        if abs(th[0]) < 0.1 and abs(th[1]) < 0.1:
            continue
        grad = np.array([(run_conserved(th + h * e) - run_conserved(th - h * e)) / (2 * h) for e in np.eye(2)])
        X = np.asarray(run_generator(th))
        assert np.linalg.norm(grad) > 1e-6
        cross = grad[0] * X[1] - grad[1] * X[0]
        assert abs(cross) < 1e-6 * np.linalg.norm(grad) * np.linalg.norm(X)


def test_generator_is_flat_direction_of_minima():
    # Differentiating <∇L, X> = 0 once more: at a minimum, H X = 0 (Problem noether (b)).
    for a in (0.3, 1.0, 2.7):
        th = np.array([a, 1.0 / a])
        Hx = anchor_hessian(th) @ np.asarray(run_generator(th))
        assert np.linalg.norm(Hx) < 1e-12 * (a * a + 1 / a ** 2)
