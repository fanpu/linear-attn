"""Zone-plate and product error fields.

ZONE PLATE  f(x, y) = cos(k r^2), r^2 = x^2 + y^2 in pixel units, 4096^2 grid centred at 0 (x, y in [-2048, 2048)).
Local radial frequency 2kr rad/px reaches Nyquist (pi) at R_N = pi / (2k). Two regimes:
  "ws" (well sampled): R_N = 2897 (beyond the image corner): the signal itself never aliases.
  "al" (aliased):      R_N = 1024: beyond r = 1024 the signal aliases; ghost zone plates appear centred on multiples of
                       2 R_N... precisely at pixel offsets that are multiples of 1024 (odd multiples carry a (-1)^u checkerboard).
Every field here is a function of r only, so the "continuous" references are exact 1-D series:
  quantization error of the continuous signal:  e(phi) = Q(cos phi) - cos phi = sum_m c_m cos(m phi),  phi = k r^2
  band-limited (what the pixel grid can hold without aliasing): keep harmonic m only where its local frequency
  2 m k r < pi, with a raised-cosine taper over the last 10% (declared); same for the signal itself (m = 1).
Decomposition written to cache (float32, 4096^2 unless noted):
  f_point    cos(k r^2) at pixel centres                      (signal as sampled)
  f_bl       band-limited signal                             A_f = f_point - f_bl   (SAMPLING ALIASING of the signal)
  e_point    Q(f_point) - f_point                            (quantization error of the samples: what an int/fp tensor holds)
  e_bl       band-limited continuous quantization error      (QUANTIZATION alone)
  A_e = e_point - e_bl                                        (harmonics of the quantization error folded by sampling)
PRODUCT (FP8 E4M3): relative error of multiplication on (a) a linear grid x, y in (0, 8], 4096^2, 512 px per unit, and
  (b) a log-log grid x, y in [2^-4, 2^4], 512 px per octave (exact octave alignment: x[j+512] = 2 x[j] bit-exactly).
  r_out = (Q(xy) - xy)/xy, r_in = (Q(x)Q(y) - xy)/xy, r_full = (Q(Q(x)Q(y)) - xy)/xy.
Run: OMP_NUM_THREADS=4 python compute_fields.py [zone] [product]
"""
import sys
import json
import numpy as np
import common as c

which = sys.argv[1:] or ["zone", "product"]
N = 4096
KS = {"ws": np.pi / (2 * 2897.0), "al": np.pi / (2 * 1024.0)}
MCAP = 2048


def taper(z):
    """1 for z < 0.9, raised cosine to 0 at z = 1 (z = local frequency / Nyquist)."""
    return np.where(z < 0.9, 1.0, np.where(z < 1.0, 0.5 * (1 + np.cos(np.pi * (z - 0.9) / 0.1)), 0.0))


def harmonic_coeffs(fmt, M=2 ** 18):
    phi = 2 * np.pi * np.arange(M) / M
    s = np.cos(phi)
    g = c.quantize(s, fmt) - s
    C = np.fft.rfft(g) / (M / 2)
    C[0] /= 2
    return C.real[: MCAP + 1]


def radial_table(coef, k, rmax, n=120_000):
    r = np.linspace(0, rmax, n)
    out = np.full(n, coef[0])
    phase = k * r ** 2
    for m in range(1, len(coef)):
        w = taper(2 * m * k * r / np.pi)
        sel = w > 0
        if not sel.any():
            break
        out[sel] += coef[m] * w[sel] * np.cos(m * phase[sel])
    return r, out


if "zone" in which:
    x = (np.arange(N) - N / 2).astype(np.float64)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    rr = np.sqrt(r2)
    rmax = float(rr.max()) + 1
    stats = {}
    coefs = {fmt: harmonic_coeffs(fmt) for fmt in c.FORMATS}
    # check: the series reproduces g(phi) (with all MCAP harmonics, no band limit)
    ph = np.linspace(0, 2 * np.pi, 5001)
    for fmt in c.FORMATS:
        g = c.quantize(np.cos(ph), fmt) - np.cos(ph)
        rec = coefs[fmt][0] + sum(coefs[fmt][m] * np.cos(m * ph) for m in range(1, MCAP + 1))
        stats[f"series_rms_residual_{fmt}"] = float(np.sqrt(np.mean((rec - g) ** 2)) / np.sqrt(np.mean(g ** 2)))
    for reg, k in KS.items():
        f_point = np.cos(k * r2)
        f_bl = np.cos(k * r2) * taper(2 * k * rr / np.pi)
        np.save(f"cache/zone_{reg}_f_point.npy", f_point.astype(np.float32))
        np.save(f"cache/zone_{reg}_f_bl.npy", f_bl.astype(np.float32))
        stats[f"{reg}_k"] = k
        stats[f"{reg}_R_N"] = float(np.pi / (2 * k))
        stats[f"{reg}_signal_alias_rel_rms"] = float(np.sqrt(np.mean((f_point - f_bl) ** 2) / np.mean(f_point ** 2)))
        for fmt in c.FORMATS:
            e_point = c.quantize(f_point, fmt) - f_point
            rt, et = radial_table(coefs[fmt], k, rmax)
            e_bl = np.interp(rr, rt, et)
            np.save(f"cache/zone_{reg}_{fmt}_e_point.npy", e_point.astype(np.float32))
            np.save(f"cache/zone_{reg}_{fmt}_e_bl.npy", e_bl.astype(np.float32))
            A = e_point - e_bl
            stats[f"{reg}_{fmt}_err_rms"] = float(np.sqrt(np.mean(e_point ** 2)))
            stats[f"{reg}_{fmt}_alias_part_rel_rms"] = float(np.sqrt(np.mean(A ** 2) / np.mean(e_point ** 2)))
            # inside r < R_N/3 the 3rd harmonic is not yet aliased; report the aliased share by radius band
            for lo_, hi_ in ((0, 256), (256, 1024), (1024, 2048), (2048, 2897)):
                m = (rr >= lo_) & (rr < hi_)
                stats[f"{reg}_{fmt}_alias_share_r{lo_}-{hi_}"] = float(np.sqrt(np.mean(A[m] ** 2) / np.mean(e_point[m] ** 2)))
            print(reg, fmt, stats[f"{reg}_{fmt}_alias_part_rel_rms"], flush=True)
    # ghost check (aliased regime): near an image-edge ghost centre (x0, 0) with x0 = 2048 - 1024 = 1024 px from centre
    k = KS["al"]
    for x0 in (1024, 2048 - 1):  # odd multiple (checkerboard) and the edge (even multiple, 2048 falls outside the grid)
        u = np.arange(-40, 41)
        X = x0 + u
        Y = np.arange(-40, 41)
        fp = np.cos(k * (X[None, :] ** 2 + Y[:, None] ** 2))
        ghost = np.cos(k * (u[None, :] ** 2 + Y[:, None] ** 2) + k * x0 ** 2) * ((-1.0) ** (u[None, :] * (x0 // 1024 % 2)))
        # for x0 = 2047 the phase slope is 2k*2047 = pi*2047/1024 ~ 2 pi - small: compare against the exact expansion
        if x0 == 1024:
            stats["ghost_at_x1024_max_abs_diff_vs_checkered_zone_plate"] = float(np.abs(fp - ghost).max())
    json.dump(stats, open("cache/zone_stats.json", "w"), indent=1)

if "product" in which:
    st = {}
    # (a) linear grid
    n = 4096
    v = (np.arange(n) + 0.5) / 512.0                     # (0, 8]
    X, Y = np.meshgrid(v, v)
    P = X * Y
    qx = c.q_fp(v, "fp8", peak=448.0)
    QX, QY = np.meshgrid(qx, qx)
    r_out = (c.q_fp(P, "fp8", peak=448.0) - P) / P
    r_in = (QX * QY - P) / P
    r_full = (c.q_fp(QX * QY, "fp8", peak=448.0) - P) / P
    np.savez("cache/product_linear.npz", v=v, r_out=r_out.astype(np.float32), r_in=r_in.astype(np.float32), r_full=r_full.astype(np.float32))
    st["linear_absmax"] = [float(np.abs(a).max()) for a in (r_out, r_in, r_full)]
    # (b) log-log grid, exact octave alignment
    base = 2.0 ** (np.arange(512) / 512.0)
    lv = np.concatenate([base * 2.0 ** o for o in range(-4, 4)])   # 4096 values, [2^-4, 2^4)
    X, Y = np.meshgrid(lv, lv)
    P = X * Y
    qx = c.q_fp(lv, "fp8", peak=448.0)
    QX, QY = np.meshgrid(qx, qx)
    l_out = (c.q_fp(P, "fp8", peak=448.0) - P) / P
    l_in = (QX * QY - P) / P
    l_full = (c.q_fp(QX * QY, "fp8", peak=448.0) - P) / P
    np.savez("cache/product_loglog.npz", v=lv, r_out=l_out.astype(np.float32), r_in=l_in.astype(np.float32), r_full=l_full.astype(np.float32))
    assert np.array_equal(lv[512:], 2 * lv[:-512])
    # exact octave self-similarity test: shift by one octave in x; require all inputs/outputs normal and unsaturated
    normal_in = lv >= 2.0 ** -6
    for name, R in (("out", l_out), ("in", l_in), ("full", l_full)):
        A_, B_ = R[:, 512:], R[:, :-512]
        QP = QX * QY
        prod_ok = (P[:, :-512] >= 2.0 ** -6) & (P[:, 512:] <= 448) & (QP[:, :-512] >= 2.0 ** -6)  # rounded product also normal
        ok = prod_ok & normal_in[None, :-512]
        st[f"loglog_{name}_octave_shift_exact_equal_frac_on_normal"] = float(np.mean(A_[ok] == B_[ok]))
        st[f"loglog_{name}_octave_shift_exact_equal_frac_all"] = float(np.mean(A_ == B_))
        st[f"loglog_{name}_normal_pixels_frac"] = float(ok.mean())
    json.dump(st, open("cache/product_stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))
