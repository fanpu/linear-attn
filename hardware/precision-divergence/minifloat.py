"""Exact, correctly-rounded logistic map x -> 4*x*(1-x) on the non-negative values of a
binary minifloat (IEEE round-to-nearest, ties-to-even), in integer arithmetic.

A format (E exponent bits, M mantissa bits, bias = 2^(E-1)-1) has every value in [0, 1]
equal to X * 2^-S with integer X and S = bias + M - 1 (the smallest subnormal).
Evaluation order is the C/Python/torch left-to-right order of `4.0 * x * (1.0 - x)`:
    t = round(1 - x);   y = round((4 x) * t)      (4x is always exact)
"""
from __future__ import annotations

import numpy as np


class Minifloat:
    def __init__(self, E: int, M: int, name: str | None = None):
        self.E, self.M = E, M
        self.bias = (1 << (E - 1)) - 1
        self.S = self.bias + self.M - 1
        self.name = name or f"E{E}M{M}"
        self.min_normal_X = 1 << (self.M)       # 2^(1-bias) in units of 2^-S  (= 2^M)

    # ------------------------------------------------------------------ enumeration
    def values_X(self) -> np.ndarray:
        """All representable X with 0 <= X*2^-S <= 1, ascending (object ints if S large)."""
        M, S, b = self.M, self.S, self.bias
        xs = list(range(0, 1 << M))                               # zero + subnormals
        for e in range(1 - b, 0):                                 # normal exponents below 1
            base = e - M + S                                      # quantum exponent in units
            xs.extend(((1 << M) + m) << base for m in range(1 << M))
        xs.append(1 << S)                                         # 1.0
        if S <= 60:
            return np.array(xs, dtype=np.int64)
        return np.array(xs, dtype=object)

    # ------------------------------------------------------------------ rounding
    def round_scalar(self, N: int, T: int) -> int:
        """Round N / 2^T (N >= 0 int, T >= S) to the format; return X (units of 2^-S)."""
        if N == 0:
            return 0
        e = N.bit_length() - 1 - T
        qe = max(e - self.M, -self.S)
        shift = T + qe
        if shift <= 0:
            return (N << (-shift)) >> 0 if shift == 0 else N << (-shift)
        q = N >> shift
        rem = N & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        return q << (qe + self.S)

    def round_vec(self, N: np.ndarray, T: int) -> np.ndarray:
        """Vectorised int64 version (requires all intermediate < 2^62)."""
        N = N.astype(np.int64)
        out = np.zeros_like(N)
        nz = N > 0
        n = N[nz]
        # floor(log2) via float64 is safe only with correction
        bl = np.floor(np.log2(n.astype(np.float64))).astype(np.int64)
        bl = np.where((np.int64(1) << np.clip(bl, 0, 62)) > n, bl - 1, bl)
        bl = np.where((np.int64(1) << np.clip(bl + 1, 0, 62)) <= n, bl + 1, bl)
        e = bl - T
        qe = np.maximum(e - self.M, -self.S)
        shift = T + qe
        pos = shift > 0
        q = np.where(pos, n >> np.where(pos, shift, 0), n << np.where(pos, 0, -shift))
        rem = np.where(pos, n & ((np.int64(1) << np.where(pos, shift, 1)) - 1), 0)
        half = np.where(pos, np.int64(1) << np.where(pos, shift - 1, 0), 1)
        up = pos & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
        q = q + up
        out[nz] = np.where(pos, q << (qe + self.S), q)
        return out

    # ------------------------------------------------------------------ maps
    def logistic_X(self, X, order: str = "4x(1-x)"):
        S = self.S
        one = 1 << S
        vec = isinstance(X, np.ndarray) and X.dtype != object and 2 * S + 3 < 62
        if vec:
            if order == "4x(1-x)":
                t = self.round_vec(one - X, S)
                return self.round_vec(4 * X * t, 2 * S)
            if order == "4x-4xx":
                p = self.round_vec(4 * X * X, 2 * S)
                return self.round_vec(4 * X - p, S)
            raise ValueError(order)
        out = []
        for x in (X.tolist() if isinstance(X, np.ndarray) else X):
            x = int(x)
            if order == "4x(1-x)":
                t = self.round_scalar(one - x, S)
                out.append(self.round_scalar(4 * x * t, 2 * S))
            else:
                p = self.round_scalar(4 * x * x, 2 * S)
                out.append(self.round_scalar(4 * x - p, S))
        return np.array(out, dtype=object if S > 60 else np.int64)

    def to_float(self, X):
        return np.array([float(int(x)) for x in X]) * 2.0 ** (-self.S) if (isinstance(X, np.ndarray) and X.dtype == object) \
            else X.astype(np.float64) * 2.0 ** (-self.S)


FP16 = Minifloat(5, 10, "float16")
BF16 = Minifloat(8, 7, "bfloat16")
E4M3 = Minifloat(4, 3, "FP8 E4M3")   # map stays in [0,1] so E4M3FN's NaN code is never reached
E5M2 = Minifloat(5, 2, "FP8 E5M2")
FP4 = Minifloat(2, 1, "FP4 E2M1")
FP32 = Minifloat(8, 23, "float32")
