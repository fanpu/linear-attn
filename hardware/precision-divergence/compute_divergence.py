"""Measure when finite-precision logistic-map orbits (r=4) leave the true orbit.

    python compute_divergence.py      # -> cache/divergence.npz, cache/divergence_stats.json

Formats (native arithmetic, verified correctly rounded by test_minifloat.py):
    bfloat16 (torch CPU tensors), float16 (numpy), float32 (numpy), float64 (numpy)
plus an ideal binary float with p-bit significand for p = 3..64 (mpmath, round-to-nearest, unbounded exponent).
Reference: mpmath at 1024 bits (checked against 2048 bits).

Seeds: K random doubles in (0.02, 0.98).  Protocol A: each format rounds the double seed on input.
Protocol B: seed is a random bfloat16 value (exactly representable in every format), isolating
arithmetic rounding.
Divergence step n*(delta) = first n with |x_n - ref_n| > delta.
"""
import json
from pathlib import Path

import mpmath
import numpy as np
import torch

HERE = Path(__file__).parent
CACHE = HERE / "cache"
K = 2000
STEPS = 120
DELTAS = (0.01, 0.1, 0.5)
MANT = {"bfloat16": 7, "float16": 10, "float32": 23, "float64": 52}


def ref_orbits(seeds, prec, steps):
    mpmath.mp.prec = prec
    out = np.empty((len(seeds), steps + 1))
    four, one = mpmath.mpf(4), mpmath.mpf(1)
    for i, s in enumerate(seeds):
        x = mpmath.mpf(float(s))
        out[i, 0] = float(x)
        for n in range(1, steps + 1):
            x = four * x * (one - x)
            out[i, n] = float(x)
    return out


def ideal_float_orbits(seeds, p, steps):
    """mpmath at precision p: each op rounded to nearest p-bit significand (no exponent limits)."""
    mpmath.mp.prec = p
    out = np.empty((len(seeds), steps + 1))
    for i, s in enumerate(seeds):
        x = mpmath.mpf(float(s))  # rounds seed to p bits
        out[i, 0] = float(x)
        for n in range(1, steps + 1):
            x = 4 * x * (1 - x)   # 4*x exact; (1-x) rounded; product rounded
            out[i, n] = float(x)
    return out


def native_orbits(seeds, fmt, steps):
    if fmt == "bfloat16":
        x = torch.tensor(seeds, dtype=torch.float64).to(torch.bfloat16)
        out = [x.double().numpy()]
        for _ in range(steps):
            x = 4 * x * (1 - x)
            assert x.dtype == torch.bfloat16
            out.append(x.double().numpy())
        return np.stack(out, 1)
    dt = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[fmt]
    x = np.asarray(seeds, dtype=np.float64).astype(dt)
    four, one = dt(4), dt(1)
    out = [x.astype(np.float64)]
    for _ in range(steps):
        x = four * x * (one - x)
        assert x.dtype == dt
        out.append(x.astype(np.float64))
    return np.stack(out, 1)


def div_step(orb, ref, delta):
    bad = np.abs(orb - ref) > delta
    first = np.where(bad.any(1), bad.argmax(1), STEPS + 1)
    return first


def main():
    rng = np.random.default_rng(2026)
    seedsA = rng.uniform(0.02, 0.98, K)
    bf = torch.tensor(rng.uniform(0.02, 0.98, K), dtype=torch.float64).to(torch.bfloat16).double().numpy()
    seedsB = bf
    res = {}
    arrays = dict(seedsA=seedsA, seedsB=seedsB)
    for proto, seeds in (("A", seedsA), ("B", seedsB)):
        ref = ref_orbits(seeds, 1024, STEPS)
        chk = ref_orbits(seeds[:50], 2048, STEPS)
        assert np.max(np.abs(chk - ref[:50])) < 1e-12, "reference not converged"
        arrays[f"ref{proto}"] = ref
        for fmt in MANT:
            orb = native_orbits(seeds, fmt, STEPS)
            arrays[f"{fmt}{proto}"] = orb
            for d in DELTAS:
                st = div_step(orb, ref, d)
                arrays[f"nstar_{fmt}_{proto}_{d}"] = st
                res[f"{fmt}|{proto}|{d}"] = dict(mantissa_bits=MANT[fmt], median=float(np.median(st)),
                                                 mean=float(st.mean()), p10=float(np.percentile(st, 10)),
                                                 p90=float(np.percentile(st, 90)),
                                                 frac_never=float(np.mean(st > STEPS)))
            print(proto, fmt, {d: res[f"{fmt}|{proto}|{d}"]["median"] for d in DELTAS})
        if proto == "B":
            continue
        # ideal p-bit floats: slope of n* vs p
        for p in list(range(3, 33)) + [40, 48, 53, 56, 64]:
            orb = ideal_float_orbits(seeds[:500], p, STEPS)
            for d in DELTAS:
                st = div_step(orb, ref[:500], d)
                arrays[f"nstar_ideal{p}_{proto}_{d}"] = st
                res[f"ideal{p}|{proto}|{d}"] = dict(mantissa_bits=p, median=float(np.median(st)), mean=float(st.mean()),
                                                   p10=float(np.percentile(st, 10)), p90=float(np.percentile(st, 90)))
            print("ideal p", p, res[f"ideal{p}|{proto}|0.1"]["median"])
    # consistency: mpmath p=53 vs numpy float64, and p=32 check on native float32 is not comparable (exponent same, ok)
    o53 = ideal_float_orbits(seedsA[:200], 53, STEPS)
    res["mpmath53_equals_numpy_float64_first200"] = bool(np.array_equal(o53, arrays["float64A"][:200]))
    o24 = ideal_float_orbits(seedsA[:200], 24, STEPS)
    res["mpmath24_equals_numpy_float32_first200"] = bool(np.array_equal(o24, arrays["float32A"][:200]))
    print({k: v for k, v in res.items() if k.startswith("mpmath")})
    np.savez_compressed(CACHE / "divergence.npz", **arrays)
    (CACHE / "divergence_stats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
