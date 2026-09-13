"""Checks: exact integer rounding vs Fractions, vs numpy float16, torch bfloat16 (CPU), numpy float32.

    python test_minifloat.py
"""
from fractions import Fraction

import numpy as np
import torch

from minifloat import BF16, E4M3, FP16, FP32, Minifloat


def frac_round(r: Fraction, mf: Minifloat) -> Fraction:
    """Reference rounding by brute force over the value list (tiny formats only)."""
    vals = [Fraction(int(x), 1 << mf.S) for x in mf.values_X()]
    best = min(vals, key=lambda v: (abs(v - r), 0))
    ties = [v for v in vals if abs(v - r) == abs(best - r)]
    if len(ties) > 1:  # choose even integral significand
        def even(v):
            X = int(v * (1 << mf.S))
            if X < (1 << mf.M):
                return X % 2 == 0
            e = X.bit_length() - 1 - mf.S
            return (X >> (e - mf.M + mf.S)) % 2 == 0
        best = [v for v in ties if even(v)][0]
    return best


def test_fraction_small():
    for mf in (E4M3, Minifloat(3, 2), Minifloat(4, 4)):
        X = mf.values_X()
        Y = mf.logistic_X(X)
        for x, y in zip(X.tolist(), Y.tolist()):
            xf = Fraction(x, 1 << mf.S)
            t = frac_round(1 - xf, mf)
            ref = frac_round(4 * xf * t, mf)
            assert Fraction(y, 1 << mf.S) == ref, (mf.name, xf, y, ref)
        Y2 = mf.logistic_X(X.astype(object))
        assert [int(a) for a in Y2] == Y.tolist()
    print("fraction reference OK")


def test_fp16_numpy():
    X = FP16.values_X()
    x = FP16.to_float(X).astype(np.float16)
    assert np.array_equal(x.astype(np.float64), FP16.to_float(X))
    y_native = (np.float16(4) * x * (np.float16(1) - x)).astype(np.float64)
    y_exact = FP16.to_float(FP16.logistic_X(X))
    bad = np.flatnonzero(y_native != y_exact)
    print(f"float16: numpy native vs exact mismatches: {len(bad)} / {len(X)}")
    t = torch.from_numpy(x.copy())
    y_t = (4 * t * (1 - t)).double().numpy()
    print(f"float16: torch CPU native vs exact mismatches: {(y_t != y_exact).sum()}")
    if torch.cuda.is_available():
        tg = t.cuda()
        y_g = (4 * tg * (1 - tg)).double().cpu().numpy()
        print(f"float16: torch CUDA native vs exact mismatches: {(y_g != y_exact).sum()}")
    return len(bad)


def test_bf16_torch():
    X = BF16.values_X()
    xf = BF16.to_float(X)
    t = torch.tensor(xf, dtype=torch.float64).to(torch.bfloat16)
    assert np.array_equal(t.double().numpy(), xf)
    y_t = (4 * t * (1 - t)).double().numpy()
    y_exact = BF16.to_float(BF16.logistic_X(X))
    nb = int((y_t != y_exact).sum())
    print(f"bfloat16: torch CPU native vs exact mismatches: {nb} / {len(X)}")
    if torch.cuda.is_available():
        tg = t.cuda()
        y_g = (4 * tg * (1 - tg)).double().cpu().numpy()
        print(f"bfloat16: torch CUDA native vs exact mismatches: {(y_g != y_exact).sum()}")
    return nb


def test_fp32_sample():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 0x3F800000 + 1, 20000).astype(np.uint32)
    x = bits.view(np.float32)
    y_native = (np.float32(4) * x * (np.float32(1) - x)).astype(np.float64)
    S = FP32.S
    X = [int(Fraction(float(v)) * (1 << S)) for v in x]
    Y = FP32.logistic_X(np.array(X, dtype=object))
    y_exact = np.array([float(Fraction(int(v), 1 << S)) for v in Y])
    print(f"float32: numpy native vs exact mismatches on 20000 random states: {(y_native != y_exact).sum()}")


if __name__ == "__main__":
    test_fraction_small()
    test_fp16_numpy()
    test_bf16_torch()
    test_fp32_sample()
