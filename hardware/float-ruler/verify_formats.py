"""Decode every format by bit enumeration, cross-check against torch/numpy, write cache.

    python verify_formats.py      # writes cache/formats.npz and cache/verification.json
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from formats import BF16, FP4_E2M1, FP8_E4M3, FP8_E5M2, FP16, FP32, FORMATS, decode_all, e8m0_all, int_all

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)


def same(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


report = {}
out = {}

# --- torch / numpy reinterpretation of raw bits -------------------------------------------
u8 = torch.arange(256, dtype=torch.int32).to(torch.uint8)
u16 = np.arange(65536, dtype=np.uint16)
checks = {
    FP8_E4M3.name: u8.view(torch.float8_e4m3fn).double().numpy(),
    FP8_E5M2.name: u8.view(torch.float8_e5m2).double().numpy(),
    FP16.name: u16.view(np.float16).astype(np.float64),
    BF16.name: torch.from_numpy(u16.astype(np.int32)).to(torch.int16).view(torch.bfloat16).double().numpy()
    if False else torch.from_numpy(u16.view(np.int16).copy()).view(torch.bfloat16).double().numpy(),
}
checks_fp16_torch = torch.from_numpy(u16.view(np.int16).copy()).view(torch.float16).double().numpy()

for fmt in FORMATS:
    d = decode_all(fmt)
    v = d["value"]
    fin = np.isfinite(v)
    pos = v[(v > 0) & fin]
    distinct = np.unique(v[fin])
    r = dict(
        bits=fmt.nbits, E=fmt.E, M=fmt.M, bias=fmt.bias,
        patterns=int(len(v)),
        nan_patterns=int(np.isnan(v).sum()),
        inf_patterns=int(np.isinf(v).sum()),
        distinct_finite_values=int(len(distinct)),
        max=float(pos.max()),
        min_normal=float(v[(d["cls"] == 2) & (v > 0)].min()),
        min_subnormal=float(pos.min()),
        eps_spacing_at_1=float(np.min(pos[pos > 1]) - 1.0),
        normal_values_per_octave=1 << fmt.M,
        positive_values_in_0_1_inclusive=int(((v >= 0) & (v <= 1) & (d["sign"] == 0)).sum()),
    )
    if fmt.name in checks:
        r["matches_torch_or_numpy_bit_view"] = same(v, checks[fmt.name])
    if fmt is FP16:
        r["matches_torch_float16_view"] = same(v, checks_fp16_torch)
        tf = torch.finfo(torch.float16)
        r["torch_finfo"] = dict(max=tf.max, smallest_normal=tf.smallest_normal, eps=tf.eps)
    if fmt is BF16:
        tf = torch.finfo(torch.bfloat16)
        r["torch_finfo"] = dict(max=tf.max, smallest_normal=tf.smallest_normal, eps=tf.eps)
    if fmt is FP8_E4M3:
        tf = torch.finfo(torch.float8_e4m3fn)
        r["torch_finfo"] = dict(max=tf.max, smallest_normal=tf.smallest_normal, eps=tf.eps)
    if fmt is FP8_E5M2:
        tf = torch.finfo(torch.float8_e5m2)
        r["torch_finfo"] = dict(max=tf.max, smallest_normal=tf.smallest_normal, eps=tf.eps)
    if fmt is FP4_E2M1:
        spec = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        r["matches_ocp_e2m1_table"] = sorted(set(np.abs(v).tolist())) == spec
    report[fmt.name] = r
    key = fmt.name.split()[-1].lower()
    for k in ("value", "e", "m", "cls", "sign"):
        out[f"{key}_{k}"] = d[k]

# fp32: spot check 4M random bit patterns + all edge patterns against numpy
rng = np.random.default_rng(0)
pats = np.concatenate([rng.integers(0, 2**32, 4_000_000, dtype=np.uint64),
                       np.array([0, 1, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0x3F800000], np.uint64)])
fmt = FP32
e = (pats >> np.uint64(23)) & np.uint64(255)
m = pats & np.uint64((1 << 23) - 1)
s = pats >> np.uint64(31)
e = e.astype(np.int64)
m = m.astype(np.float64)
val = np.where(e == 0, m / 2**23 * 2.0 ** (-126), (1 + m / 2**23) * np.power(2.0, (e - 127).astype(float)))
val = np.where(e == 255, np.where(m == 0, np.inf, np.nan), val)
val = np.where(s == 1, -val, val)
report["float32 (4M random patterns)"] = dict(matches_numpy=same(val, pats.astype(np.uint32).view(np.float32).astype(np.float64)))

# E8M0 scale vs torch.float8_e8m0fnu
b, v = e8m0_all()
report["E8M0 (MX scale)"] = dict(bias=127, min=float(v[0]), max=float(v[254]), nan_code=255,
                                 matches_torch_float8_e8m0fnu=same(v, u8.view(torch.float8_e8m0fnu).double().numpy()))
out["e8m0_value"] = v
for nb in (4, 8):
    b, v = int_all(nb)
    out[f"int{nb}_value"] = v
    report[f"int{nb}"] = dict(min=float(v.min()), max=float(v.max()), distinct=int(len(np.unique(v))))

# exact self-similarity statement for E4M3: v(e+1, m) == 2 v(e, m) for all normal e < top
d = decode_all(FP8_E4M3)
ok = True
for e_ in range(1, 14):
    a = d["value"][(d["sign"] == 0) & (d["e"] == e_)]
    bb = d["value"][(d["sign"] == 0) & (d["e"] == e_ + 1)]
    ok &= bool(np.array_equal(2 * a, bb))
report["E4M3 octave self-similarity v(e+1,m)=2v(e,m), e=1..13"] = ok
top = d["value"][(d["sign"] == 0) & (d["e"] == 15)]
report["E4M3 top octave finite values"] = [float(x) for x in top if np.isfinite(x)]

# Exact rational check of a few values (guards against float64 decode mistakes)
report["exact_fraction_checks"] = {
    "E4M3 min subnormal": str(Fraction(float(np.min(d['value'][(d['value'] > 0)])))),
    "bf16 min subnormal": str(Fraction(report["bfloat16"]["min_subnormal"])),
}

np.savez_compressed(CACHE / "formats.npz", **out)
(CACHE / "verification.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
