"""Numerical verification of every claim captioned in the Dither README.

Writes cache/verify.json and cache/moments.npz (the moment curves are also rendered).
Run:  python verify.py
"""
import json
import numpy as np
from scipy.special import jv
from scipy.signal import get_window
from scipy import stats
import dsp

rng = np.random.default_rng(12345)
out = {}
FS = dsp.FS


def psd(x, win="blackmanharris"):
    w = get_window(win, len(x), fftbins=True)
    return np.abs(np.fft.rfft(x * w)) ** 2


# ------------------------------------------------------------------ V1: undithered sine -> harmonics, aliasing
N = 2 ** 16
f0 = 3001.3
t = np.arange(N) / FS
x = np.sin(2 * np.pi * f0 * t)
bins = np.fft.rfftfreq(N, 1 / FS)
res = {}
for kind in [None, "rpdf", "tpdf"]:
    y = dsp.quant_int(x, 4, dither=kind, rng=rng)
    e = y - x
    P = psd(e)
    for K in (15, 51, 201):
        mask = np.zeros_like(P, bool)
        for k in range(1, K + 1):
            b = int(round(dsp.fold(k * f0) / (FS / N)))
            mask[max(b - 4, 0):b + 5] = True
        res[f"{kind}_K{K}"] = dict(power_frac=float(P[mask].sum() / P.sum()), bin_coverage=float(mask.mean()))
out["V1_harmonic_power_fraction_4bit_sine_3001.3Hz"] = res

# identify the 40 strongest spectral peaks of the undithered error
y = dsp.quant_int(x, 4)
e = y - x
P = psd(e)
Pdb = 10 * np.log10(P / P.max())
pk = [i for i in range(2, len(P) - 2) if P[i] == P[i - 2:i + 3].max()]
pk = sorted(pk, key=lambda i: -P[i])[:40]
ks = np.arange(1, 2000)
fk = dsp.fold(ks * f0)
matched, aliased, odd = 0, 0, 0
table = []
for i in pk:
    j = np.argmin(np.abs(fk - bins[i]))
    ok = abs(fk[j] - bins[i]) <= 1.5 * FS / N
    matched += ok
    aliased += ok and ks[j] * f0 > FS / 2
    odd += ok and ks[j] % 2 == 1
    table.append(dict(freq=float(bins[i]), dB=float(Pdb[i]), k=int(ks[j]), kf0=float(ks[j] * f0), matched=bool(ok)))
out["V1_top40_error_peaks"] = dict(matched_to_folded_harmonic=int(matched), of_which_aliased_past_nyquist=int(aliased),
                                   of_which_odd_k=int(odd), top10=table[:10])
# even-harmonic check: power at even harmonic bins vs odd (exact half-wave symmetry -> only odd k)
ev = np.mean([P[int(round(dsp.fold(k * f0) / (FS / N)))] for k in range(2, 60, 2)])
od = np.mean([P[int(round(dsp.fold(k * f0) / (FS / N)))] for k in range(1, 60, 2)])
out["V1_odd_vs_even_harmonic_mean_power_dB"] = float(10 * np.log10(od / ev))

# the same peak-identification test at several bit depths (a chirp is locally such a tone; its error is exactly
# g(phi(t)) for a periodic g, so its content is the harmonics k*phi(t) folded by sampling -- see the key plate)
bd = {}
for bits in (2, 3, 4, 6, 8, 12):
    e = dsp.quant_int(x, bits) - x
    P = psd(e)
    pk = [i for i in range(2, len(P) - 2) if P[i] == P[i - 2:i + 3].max()]
    pk = sorted(pk, key=lambda i: -P[i])[:40]
    fk = dsp.fold(ks * f0)
    m = a_ = 0
    kk = []
    for i in pk:
        j = np.argmin(np.abs(fk - bins[i]))
        if abs(fk[j] - bins[i]) <= 1.5 * FS / N:
            m += 1; a_ += ks[j] * f0 > FS / 2; kk.append(int(ks[j]))
    bd[bits] = dict(top40_matched=m, aliased=int(a_), median_k=float(np.median(kk)) if kk else None,
                    peak_level_A_over_LSB=round(1 / dsp.int_step(bits), 1), two_pi_A_over_LSB=round(2 * np.pi / dsp.int_step(bits), 1))
out["V1_top40_peaks_by_bitdepth"] = bd


# ------------------------------------------------------------------ V2: conditional moments (exact, by integration)
def tri_cdf(v):  # TPDF on [-1,1]
    v = np.clip(v, -1, 1)
    return np.where(v < 0, 0.5 * (1 + v) ** 2, 1 - 0.5 * (1 - v) ** 2)


def rect_cdf(v):  # RPDF on [-1/2,1/2]
    return np.clip(v + 0.5, 0, 1)


xs = np.linspace(-0.5, 0.5, 1001)
levels = np.arange(-4, 5)[:, None]
mom = {}
for name, F in [("rpdf", rect_cdf), ("tpdf", tri_cdf)]:
    p = F(levels + 0.5 - xs[None]) - F(levels - 0.5 - xs[None])  # P(round(x+nu)=n)
    err = levels - xs[None]
    m = [np.sum(p * err ** r, axis=0) for r in (1, 2, 3, 4)]
    mom[name] = np.array(m)
mom["none"] = np.array([(np.round(xs) - xs) ** r for r in (1, 2, 3, 4)])
mom["subtractive_rpdf"] = np.array([np.full_like(xs, v) for v in (0, 1 / 12, 0, 1 / 80)])
np.savez("cache/moments.npz", xs=xs, **mom)
summ = {}
for k, m in mom.items():
    summ[k] = dict(mean_range=[float(m[0].min()), float(m[0].max())], m2_range=[float(m[1].min()), float(m[1].max())],
                   m3_range=[float(m[2].min()), float(m[2].max())], m4_range=[float(m[3].min()), float(m[3].max())])
out["V2_exact_conditional_moments_LSB_units"] = summ

# Monte-Carlo cross-check including subtractive (error = Q(x+nu) - nu - x)
mc = {}
for name, kind, sub in [("rpdf", "rpdf", False), ("tpdf", "tpdf", False), ("subtractive_rpdf", "rpdf", True)]:
    means, m2s, ks_p = [], [], []
    for xv in np.linspace(-0.5, 0.5, 41):
        nu = dsp.make_dither(kind, 400_000, rng)
        q = np.round(xv + nu)
        e = (q - nu - xv) if sub else (q - xv)
        means.append(e.mean())
        m2s.append((e ** 2).mean())
        if sub:
            ks_p.append(stats.kstest(e[:20000], "uniform", args=(-0.5, 1.0)).pvalue)
    mc[name] = dict(mean_range=[float(min(means)), float(max(means))], m2_range=[float(min(m2s)), float(max(m2s))])
    if sub:
        mc[name]["ks_uniform_min_pvalue_over_41_inputs"] = float(min(ks_p))
out["V2_montecarlo_4e5_per_input"] = mc

# ------------------------------------------------------------------ V3: noise modulation measured on the piece's own chirp
xc4, _ = dsp.log_chirp(20.0, 20.0, 24000.0)
d4 = dsp.int_step(4)
nm = {}
for kind in [None, "rpdf", "tpdf"]:
    e = (dsp.quant_int(xc4, 4, dither=kind, rng=rng) - xc4) / d4
    ph = np.mod(xc4 / d4 + 0.5, 1.0) - 0.5
    b = np.clip(((ph + 0.5) * 20).astype(int), 0, 19)
    v = np.array([np.mean(e[b == i] ** 2) for i in range(20)])
    mu = np.array([np.mean(e[b == i]) for i in range(20)])
    nm[str(kind)] = dict(err_power_by_input_position_min=float(v.min()), max=float(v.max()),
                         cond_mean_absmax=float(np.abs(mu).max()), overall_power=float(np.mean(e ** 2)))
out["V3_chirp_4bit_error_power_vs_position_within_LSB"] = nm

# decaying tone: short-time error RMS in the silent tail (RPDF gates off, TPDF does not)
tt6 = np.arange(int(6 * FS)) / FS
xd = np.sin(2 * np.pi * 220 * tt6) * np.exp(-tt6 / 0.8)
dec = {}
for kind in [None, "rpdf", "tpdf"]:
    e = (dsp.quant_int(xd, 6, dither=kind, rng=rng) - xd) / dsp.int_step(6)
    seg = lambda a, b: float(np.sqrt(np.mean(e[int(a * FS):int(b * FS)] ** 2)))
    dec[str(kind)] = dict(rms_err_0_0p5s=seg(0, .5), rms_err_4_6s=seg(4, 6))
out["V3_decaying_220Hz_6bit_rms_error_LSB"] = dec

# ------------------------------------------------------------------ V4: spectral whiteness of dithered error (sine)
wh = {}
for kind in [None, "rpdf", "tpdf"]:
    e = dsp.quant_int(x, 4, dither=kind, rng=rng) - x
    P = psd(e)[50:-50]
    Ps = np.convolve(P, np.ones(64) / 64, mode="valid")
    wh[str(kind)] = dict(spectral_flatness_smoothed=float(np.exp(np.mean(np.log(Ps))) / np.mean(Ps)),
                         max_bin_over_median_dB=float(10 * np.log10(P.max() / np.median(P))))
out["V4_error_spectrum_whiteness_4bit_sine"] = wh

# ------------------------------------------------------------------ V5: FP formats, amplitude dependence
def sinad(y, x):
    return 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))


xs5 = np.sin(2 * np.pi * 997.3 * np.arange(2 ** 15) / FS)
amps_db = np.arange(0, -61, -6)
fp4, fp8 = dsp.make_fp_quantizer("fp4"), dsp.make_fp_quantizer("fp8")
sn = {"int4": [], "int8": [], "fp4": [], "fp8": []}
for adb in amps_db:
    a = 10 ** (adb / 20)
    s = a * xs5
    sn["int4"].append(sinad(dsp.quant_int(s, 4), s))
    sn["int8"].append(sinad(dsp.quant_int(s, 8), s))
    sn["fp4"].append(sinad(fp4(s), s))
    sn["fp8"].append(sinad(fp8(s), s))
out["V5_SINAD_dB_vs_amplitude_dBFS"] = dict(amps_dBFS=amps_db.tolist(), **{k: [round(v, 2) for v in vals] for k, vals in sn.items()})
# octave invariance: Q(x/2^j) == Q(x)/2^j for samples that stay in the normal (non-subnormal) range
inv = {}
for name, q, nmin in [("fp8", fp8, 2.0 ** -6), ("fp4", fp4, 1.0)]:
    base = q(xs5)
    r = []
    for j in range(1, 9):
        xj = xs5 * 2.0 ** -j
        normal = np.abs(xj * q.vmax) >= nmin
        eq = q(xj) == base * 2.0 ** -j
        r.append(dict(j=j, frac_equal_all=round(float(eq.mean()), 4), frac_equal_on_normal=float(eq[normal].mean()),
                      frac_normal=round(float(normal.mean()), 4)))
    inv[name] = r
out["V5_octave_scale_invariance"] = inv

# ------------------------------------------------------------------ V6: sigma-delta slopes and idle tones
Nsd = 2 ** 16
tsd = np.arange(Nsd)
fin = 37 / Nsd * 7  # low-frequency sine, OSR-ish test
sl = {}
for order in (1, 2):
    u = 0.5 * np.sin(2 * np.pi * (1021 / Nsd) * tsd)
    ysd = dsp.sigma_delta(u[None], order=order)[0].astype(float)
    P = psd(ysd)
    fr = np.arange(len(P)) / Nsd
    band = lambda a, b: np.median(P[(fr > a) & (fr < b)])  # median: robust to idle tones / harmonics
    slope = 10 * np.log10(band(0.02, 0.03) / band(0.002, 0.003))
    sl[f"order{order}"] = dict(noise_rise_dB_per_decade_0p0025_to_0p025fs=float(slope), expected=20 * order)
out["V6_sigma_delta_noise_shaping"] = sl
us = np.linspace(-0.9, 0.9, 37)
ysd = dsp.sigma_delta(np.repeat(us[:, None], 2 ** 14, axis=1) + 1e-9, order=1).astype(float)
ok = 0
for i, u0 in enumerate(us):
    P = psd(ysd[i, 2000:] - ysd[i, 2000:].mean(), win="hann")
    fr = np.arange(len(P)) / (len(ysd[i, 2000:]))
    fpk = fr[np.argmax(P[5:]) + 5]
    rho = (1 + u0) / 2
    ok += abs(fpk - min(rho, 1 - rho)) < 2 / len(ysd[i, 2000:])
out["V6_first_order_idle_tone_at_fold(rho)_rho=(1+u)/2"] = f"{ok}/{len(us)} DC inputs"

# ------------------------------------------------------------------ V7: Bessel series for harmonic amplitudes vs amplitude
M = 2 ** 18
th = 2 * np.pi * np.arange(M) / M
bes = []
nn = np.arange(1, 20001)
for A in (1.3, 3.7, 7.25):
    e = np.round(A * np.sin(th)) - A * np.sin(th)
    c = np.fft.rfft(e) / (M / 2)
    for k in (1, 3, 5, 11, 21):
        bk_fft = -c[k].imag
        bk_bes = np.sum(2 * (-1.0) ** nn / (np.pi * nn) * jv(k, 2 * np.pi * nn * A))
        bes.append(dict(A_LSB=A, k=k, fft=float(bk_fft), bessel_series_n20000=float(bk_bes)))
out["V7_harmonic_coeff_fft_vs_bessel"] = bes

json.dump(out, open("cache/verify.json", "w"), indent=1)
print(json.dumps(out, indent=1))
