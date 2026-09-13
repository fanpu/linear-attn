"""Measured conditional moments of the total error on the actual 30 s chirp (same seed/dither as the plates is not
required: moments are statistics). Bins samples by input position within an LSB, frac(x/Delta) in [-1/2, 1/2).
Saves cache/chirp_moments.npz. Exact theory curves come from verify.py (cache/moments.npz)."""
import numpy as np
import dsp

x, _ = dsp.log_chirp(30.0, 20.0, 24000.0)
rng = np.random.default_rng(7)
NB = 48
out = {}
for b in (3, 4):
    d = dsp.int_step(b)
    pos = np.mod(x / d + 0.5, 1.0) - 0.5
    idx = np.clip(((pos + 0.5) * NB).astype(int), 0, NB - 1)
    cnt = np.bincount(idx, minlength=NB)
    for kind, sub in [(None, False), ("rpdf", False), ("tpdf", False), ("rpdf", True)]:
        e = (dsp.quant_int(x, b, dither=kind, rng=rng, subtractive=sub) - x) / d
        nm = f"b{b}_{'sub' if sub else (kind or 'none')}"
        out[nm + "_m1"] = np.bincount(idx, e, NB) / cnt
        out[nm + "_m2"] = np.bincount(idx, e ** 2, NB) / cnt
out["centers"] = (np.arange(NB) + 0.5) / NB - 0.5
np.savez("cache/chirp_moments.npz", **out)
print({k: (float(v.min()), float(v.max())) for k, v in out.items() if k.startswith("b3")})
