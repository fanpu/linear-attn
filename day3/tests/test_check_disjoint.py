"""Test for check_disjoint.  [AI-owned; do not edit]

Two fake training shards with two aligned copies of validation windows and one
misaligned copy planted. The aligned copies count (2); the misaligned one does not.
"""
import numpy as np
from adapters import run_check_disjoint


def test_check_disjoint():
    rng = np.random.default_rng(0)
    T = 16
    val = rng.integers(0, 50304, size=(8, T)).astype(np.int64)
    s1 = rng.integers(0, 50304, size=20 * T).astype(np.uint16)
    s2 = rng.integers(0, 50304, size=20 * T + 7).astype(np.uint16)  # a ragged tail, like a real shard
    s1[3 * T:4 * T] = val[2]                # aligned copy   -> counts
    s2[7 * T:8 * T] = val[5]                # aligned copy   -> counts
    s2[10 * T + 5:11 * T + 5] = val[6]      # misaligned     -> does not count
    assert run_check_disjoint([s1, s2], val, T) == 2
    assert run_check_disjoint([s1[:2 * T], s2[:2 * T]], val, T) == 0
    assert run_check_disjoint([s1, s2], val[:2], T) == 0
