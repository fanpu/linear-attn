"""Test for seed_stats.  [AI-owned; do not edit]

Checks the handout's hand-worked pooled example (three sizes, two seeds, nu = 3)
to four decimals, the single-size case (nu = 1), a three-seed case (nu = 4), and
that a size with one entry is skipped.
"""
import math

import pytest
from adapters import run_seed_stats


def test_seed_stats():
    r = run_seed_stats({"30M": [4.41, 4.43], "60M": [3.90, 3.88], "125M": [3.50, 3.53]})
    assert r["nu"] == 3
    assert r["s_pooled"] == pytest.approx(0.016833, abs=1e-5)
    assert r["lo"] == pytest.approx(0.011661, abs=1e-5)
    assert r["hi"] == pytest.approx(0.038139, abs=1e-5)
    assert r["mdd"] == pytest.approx(0.042081, abs=1e-5)

    r1 = run_seed_stats({"30M": [4.41, 4.43]})
    assert r1["nu"] == 1
    assert r1["s_pooled"] == pytest.approx(0.0141421, abs=1e-6)
    assert r1["lo"] == pytest.approx(0.0141421 / 1.644854, abs=1e-5)
    assert r1["hi"] == pytest.approx(0.0141421 / 0.125661, abs=1e-4)

    r3 = run_seed_stats({"a": [1.00, 1.02, 1.04], "b": [2.00, 2.00, 2.03]})
    ss = (0.02 ** 2) * 2 + (0.01 ** 2 * 2 + 0.02 ** 2)
    assert r3["nu"] == 4
    assert r3["s_pooled"] == pytest.approx(math.sqrt(ss / 4), rel=1e-6)

    r4 = run_seed_stats({"30M": [4.41, 4.43], "60M": [3.9]})
    assert r4["nu"] == 1 and r4["s_pooled"] == pytest.approx(0.0141421, abs=1e-6)
