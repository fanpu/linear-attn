"""Exact decoders for the number formats in *The Ruler*.

Every format is decoded by enumerating all of its bit patterns with a generic
minifloat rule, then cross-checked against torch / numpy reinterpretation of the
same bits (see verify_formats.py).  All decoded values are dyadic rationals that
are exactly representable in float64, so float64 arrays are exact here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FloatFormat:
    name: str
    E: int          # exponent bits
    M: int          # mantissa bits
    bias: int
    specials: str   # 'ieee' (all-ones exp = inf/nan), 'fn' (only all-ones pattern = nan), 'none'

    @property
    def nbits(self) -> int:
        return 1 + self.E + self.M


FP4_E2M1 = FloatFormat("FP4 E2M1", 2, 1, 1, "none")
FP8_E4M3 = FloatFormat("FP8 E4M3FN", 4, 3, 7, "fn")
FP8_E5M2 = FloatFormat("FP8 E5M2", 5, 2, 15, "ieee")
FP16 = FloatFormat("float16", 5, 10, 15, "ieee")
BF16 = FloatFormat("bfloat16", 8, 7, 127, "ieee")
FP32 = FloatFormat("float32", 8, 23, 127, "ieee")

FORMATS = [FP4_E2M1, FP8_E4M3, FP8_E5M2, FP16, BF16]


def decode_all(fmt: FloatFormat):
    """Decode every bit pattern. Returns dict of arrays indexed by bit pattern."""
    n = 1 << fmt.nbits
    bits = np.arange(n, dtype=np.int64)
    sign = (bits >> (fmt.E + fmt.M)) & 1
    e = (bits >> fmt.M) & ((1 << fmt.E) - 1)
    m = bits & ((1 << fmt.M) - 1)
    emax_field = (1 << fmt.E) - 1
    mmax_field = (1 << fmt.M) - 1
    frac = m.astype(np.float64) / (1 << fmt.M)
    sub = e == 0
    val = np.where(sub, frac * 2.0 ** (1 - fmt.bias),
                   (1.0 + frac) * np.power(2.0, (e - fmt.bias).astype(np.float64)))
    cls = np.where(sub, np.where(m == 0, 0, 1), 2)  # 0 zero, 1 subnormal, 2 normal
    if fmt.specials == "ieee":
        top = e == emax_field
        val = np.where(top & (m == 0), np.inf, val)
        val = np.where(top & (m != 0), np.nan, val)
        cls = np.where(top, np.where(m == 0, 3, 4), cls)
    elif fmt.specials == "fn":
        nan = (e == emax_field) & (m == mmax_field)
        val = np.where(nan, np.nan, val)
        cls = np.where(nan, 4, cls)
    val = np.where(sign == 1, -val, val)
    return dict(bits=bits, sign=sign, e=e, m=m, value=val, cls=cls)


def e8m0_all():
    """OCP MX E8M0 scale: unsigned 8-bit exponent, bias 127, 0xFF = NaN, no zero, no inf."""
    b = np.arange(256)
    v = np.power(2.0, (b - 127).astype(np.float64))
    v[255] = np.nan
    return b, v


def int_all(nbits: int):
    b = np.arange(1 << nbits)
    v = np.where(b >= (1 << (nbits - 1)), b - (1 << nbits), b).astype(np.float64)
    return b, v


def positive_finite(fmt: FloatFormat):
    d = decode_all(fmt)
    v = d["value"]
    ok = (d["sign"] == 0) & (d["cls"] <= 2) & (v > 0)
    order = np.argsort(v[ok])
    return {k: d[k][ok][order] for k in ("value", "e", "m", "cls")}


def round_to_format(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round-to-nearest, ties to the value with even mantissa LSB, onto a sorted grid of
    non-negative representable values (grid includes 0).  Saturates at grid max.
    Used only for FP4/E4M3 block quantisation illustrations."""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    a = np.abs(x)
    idx = np.searchsorted(grid, a)
    idx = np.clip(idx, 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    dlo, dhi = a - lo, hi - a
    pick_hi = dhi < dlo
    tie = dhi == dlo
    # tie -> even index in the grid (grid index parity == mantissa LSB parity for these formats)
    pick_hi = np.where(tie, (idx % 2) == 0, pick_hi)
    q = np.where(pick_hi, hi, lo)
    q = np.where(a >= grid[-1], grid[-1], q)
    return s * q
