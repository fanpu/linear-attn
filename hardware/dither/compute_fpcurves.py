"""SINAD of a 997.3 Hz sine vs amplitude (0 to -84 dB re format max, 0.25 dB steps) for int4/int8/FP4/FP8.
Uniform grids lose ~6 dB per 6 dB of level; FP grids hold SINAD roughly constant with an octave-periodic ripple
(exact scale invariance Q(x/2)=Q(x)/2 in the normal range) until the subnormal/zero region is reached."""
import numpy as np
import dsp

xs = np.sin(2 * np.pi * 997.3 * np.arange(2 ** 15) / dsp.FS)
adb = np.arange(0, -84.01, -0.25)
fp4, fp8 = dsp.make_fp_quantizer("fp4"), dsp.make_fp_quantizer("fp8")
out = {k: [] for k in ("int4", "int8", "fp4", "fp8")}
for a_db in adb:
    s = 10 ** (a_db / 20) * xs
    for k, y in (("int4", dsp.quant_int(s, 4)), ("int8", dsp.quant_int(s, 8)), ("fp4", fp4(s)), ("fp8", fp8(s))):
        err = np.mean((y - s) ** 2)
        out[k].append(10 * np.log10(np.mean(s ** 2) / err) if err > 0 else np.inf)
np.savez("cache/fp_sinad_curves.npz", amp_db=adb, **{k: np.array(v) for k, v in out.items()})
for k, v in out.items():
    v = np.array(v); print(k, "SINAD at 0/-12/-24/-48 dB:", [round(float(v[int(-d / 0.25)]), 1) for d in (0, -12, -24, -48)])
