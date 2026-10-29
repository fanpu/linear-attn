"""Tests for classify_fixed_point.  [AI-owned; do not edit]

Problem (phase-portrait): the eigenvalue criterion of Table 1, on hand-picked 2×2
Jacobians and on the anchor model's flow Jacobian J = -∇²L at its fixed points.
"""
import numpy as np
import pytest
from adapters import run_classify_fixed_point

from tdyn.models import anchor_flow_jacobian

CASES = [
    ([[-1.0, 0.0], [0.0, -3.0]], "stable node"),        # λ = -1, -3
    ([[-2.0, 1.0], [0.0, -2.0]], "stable node"),        # repeated λ = -2
    ([[2.0, 1.0], [0.0, 3.0]], "unstable node"),        # λ = 2, 3
    ([[1.0, 2.0], [2.0, 1.0]], "saddle"),               # λ = 3, -1
    ([[-1.0, -2.0], [2.0, -1.0]], "stable spiral"),     # λ = -1 ± 2i
    ([[1.0, -2.0], [2.0, 1.0]], "unstable spiral"),     # λ = 1 ± 2i
    ([[0.0, 1.0], [-4.0, 0.0]], "center"),              # λ = ± 2i
    ([[0.0, 0.0], [0.0, -1.0]], "degenerate"),          # λ = 0, -1
    ([[1.0, 1.0], [1.0, 1.0]], "degenerate"),           # λ = 0, 2
]


@pytest.mark.parametrize("J,label", CASES)
def test_classify_fixed_point_hand_picked(J, label):
    assert run_classify_fixed_point(np.array(J)) == label


def test_classify_fixed_point_anchor_origin():
    assert run_classify_fixed_point(anchor_flow_jacobian([0.0, 0.0])) == "saddle"


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, -1.0, -3.0])
def test_classify_fixed_point_anchor_manifold(a):
    # every point of ab = 1 is a fixed point; its Jacobian has a zero eigenvalue
    assert run_classify_fixed_point(anchor_flow_jacobian([a, 1.0 / a])) == "degenerate"
