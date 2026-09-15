"""Test for lr_at (the warmup-then-cosine schedule).  [AI-owned; do not edit]

Checks the hand-worked example in the handout (S = 801, W = 24) at the
endpoints and at the quarter and half points of the decay, then the shape
constraints, then the warmup length of today's real 30M run (S = 9155).
"""
import pytest
from adapters import run_lr_at


def _lr(step, S):
    return run_lr_at(step, S, 6e-4, 0.03, 0.1)


def test_lr_at():
    eta, S, W = 6e-4, 801, 24
    assert _lr(0, S) == pytest.approx(eta / W, rel=1e-9)
    assert _lr(W - 1, S) == pytest.approx(eta, rel=1e-9)
    assert _lr(W, S) == pytest.approx(eta, rel=1e-9)
    assert _lr(218, S) == pytest.approx(5.209188e-4, rel=1e-5), "p = 0.25: cosine gives 5.209e-4; a straight line would give 4.65e-4"
    assert _lr(412, S) == pytest.approx(3.3e-4, rel=1e-6)
    assert _lr(S - 1, S) == pytest.approx(0.1 * eta, rel=1e-9)
    lrs = [_lr(s, S) for s in range(S)]
    assert all(lrs[s + 1] >= lrs[s] for s in range(W - 1)), "warmup must be non-decreasing"
    assert all(lrs[s + 1] <= lrs[s] for s in range(W, S - 1)), "decay must be non-increasing"
    assert min(lrs[W:]) >= 0.1 * eta * (1 - 1e-9), "never below the floor"
    S2 = 9155  # today's 30M run: W = round(274.65) = 275
    assert _lr(275, S2) == pytest.approx(eta, rel=1e-9)
    assert _lr(274, S2) == pytest.approx(eta, rel=1e-9)
    assert _lr(273, S2) < eta
    assert _lr(S2 - 1, S2) == pytest.approx(0.1 * eta, rel=1e-9)
