"""Verification for the ouroboros gallery (reads caches only; see verify_compute.py for the measurements).

usage: python verify.py            -> verify_results.txt + gallery/verify_boundary.png

1. CRN determinism: rows of the phase map with equal n_r = floor(lambda n) must be the same chain.
2. Collapse metrics over seeds 0-4 at the film settings + sliced-W2 noise floor of true samples.
3. Escape-set boundary of the replace phase map (tau = 0.25 above generation 0 within G = 80):
   box counting on the boundary, against (a) the smoothed field alone and (b) a null model: smoothed field
   plus phase-randomised residual (same power spectrum, random phases), thresholded identically.
4. Resolution check: native-resolution window (every integer n in [32, 96], every distinct n_r) vs the
   phase-map cells in the same window, rasterised on a common 256 x 256 grid in (log n, lambda); seed 0 vs seed 1.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

TAU, OUT = 0.25, []


def say(s=""):
    print(s); OUT.append(s)


def boundary(mask):
    b = np.zeros_like(mask, bool)
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]; b[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    b[:-1] |= mask[:-1] != mask[1:]; b[1:] |= mask[:-1] != mask[1:]
    return b


def boxcount(b, eps):
    out = []
    for e in eps:
        H, W = (b.shape[0] // e) * e, (b.shape[1] // e) * e
        out.append(b[:H, :W].reshape(H // e, e, W // e, e).any((1, 3)).sum())
    return np.array(out, float)


def slope(eps, N, lo, hi):
    k = (np.array(eps) >= lo) & (np.array(eps) <= hi) & (N > 0)
    if k.sum() < 2:
        return np.nan
    return -np.polyfit(np.log(np.array(eps)[k]), np.log(N[k]), 1)[0]


# ------------------------------------------------------------------ 1. CRN determinism
d = np.load("cache/phase_ring_gmm_replace.npz")
lams, ns, sw = d["lams"], d["ns"], d["sw2"][0].astype(float)  # [L, N, G+1]
G = sw.shape[-1] - 1
worst, groups = 0.0, 0
for j, n in enumerate(ns):
    nr = np.floor(lams * n + 1e-9).astype(int)
    for v in np.unique(nr):
        rows = np.nonzero(nr == v)[0]
        if len(rows) > 1:
            groups += 1
            worst = max(worst, float(np.abs(sw[rows, j] - sw[rows[0], j]).max()))
distinct = np.array([len(np.unique(np.floor(lams * n + 1e-9))) for n in np.unique(ns)])
say("== 1. common random numbers")
say(f"rows sharing n_r = floor(lambda n): {groups} groups, max |sliced W2 difference| over all generations = {worst:.2e}")
say(f"distinct chains per column: {distinct.min()} (n = {np.unique(ns)[0]}) to {distinct.max()} of {len(lams)} rows; "
    f"lambda resolution floor = 1/n (0.125 at n = 8, 0.0104 = grid spacing at n = 96)")

# ------------------------------------------------------------------ 2. collapse metrics
say("\n== 2. collapse over seeds (ring, GMM K = 8, n = 128, G = 200)")
if os.path.exists("cache/verify_floor.npz"):
    f = np.load("cache/verify_floor.npz")
    say(f"noise floor, true samples (64 draws of 2048): sliced W2 {f['sw2'].mean():.3f} +- {f['sw2'].std():.3f}, "
        f"modes {f['modes'].mean():.2f}, var_ratio {f['var_ratio'].mean():.3f}")
if os.path.exists("cache/verify_seeds.npz"):
    s = np.load("cache/verify_seeds.npz")
    say(f"{'regime':<11} {'sW2 g0':>13} {'sW2 g50':>13} {'sW2 g200':>13} {'modes g200':>12} {'var_ratio g200':>15}")
    for reg in ("replace", "anchored", "accumulate"):
        w, m, v = s[f"{reg}/sw2"], s[f"{reg}/modes"], s[f"{reg}/var_ratio"]
        ms = lambda x: f"{x.mean():.3f}+-{x.std():.3f}"  # noqa: E731
        say(f"{reg:<11} {ms(w[:, 0]):>13} {ms(w[:, 50]):>13} {ms(w[:, -1]):>13} "
            f"{' '.join(str(int(q)) for q in m[:, -1]):>12} {ms(v[:, -1]):>15}")

# ------------------------------------------------------------------ 3. box counting + null model
say(f"\n== 3. escape-set boundary, replace phase map ({len(lams)} x {len(ns)} cells, tau = {TAU}, G = {G})")
F = (sw - sw[..., :1]).max(-1) - TAU  # > 0: escaped
esc = F > 0
b = boundary(esc)
eps = [1, 2, 3, 4, 6, 8, 12, 16]
Nb = boxcount(b, eps)
D_real = slope(eps, Nb, 1, 8)
say(f"escaped cells {esc.mean():.1%}; boundary cells {b.sum()}; box counts eps={eps}: {Nb.astype(int).tolist()}")
say(f"box-counting slope D over eps 1-8 cells (0.9 decades): {D_real:.2f}")
Fs = gaussian_filter(F, 3.0)
res = F - Fs
Ns = boxcount(boundary(Fs > 0), eps)
D_smooth = slope(eps, Ns, 1, 8)
rng = np.random.default_rng(0)
Dn, Nn, Da, Na = [], [], [], []
amp = np.abs(np.fft.fft2(res))
for _ in range(40):
    ph = np.angle(np.fft.fft2(rng.standard_normal(res.shape)))
    r = np.real(np.fft.ifft2(amp * np.exp(1j * ph)))
    r *= res.std() / r.std()
    bn = boundary(Fs + r > 0)
    Nn.append(boxcount(bn, eps)); Dn.append(slope(eps, Nn[-1], 1, 8))
    # amplitude-adjusted variant: same ranks as r, values = the measured residual distribution (heavy tails kept)
    ra = np.empty(res.size); ra[np.argsort(r.ravel())] = np.sort(res.ravel())
    Na.append(boxcount(boundary(Fs + ra.reshape(res.shape) > 0), eps)); Da.append(slope(eps, Na[-1], 1, 8))
Dn, Nn, Da, Na = np.array(Dn), np.array(Nn), np.array(Da), np.array(Na)
say(f"smoothed field alone (Gaussian sigma = 3 cells): D = {D_smooth:.2f}")
say(f"null model, smooth + phase-randomised residual (40 surrogates): D = {Dn.mean():.2f} +- {Dn.std():.2f}; "
    f"real D is at the {np.mean(Dn < D_real):.0%} quantile; boundary cells {np.mean([boundary(Fs + 0 > 0).sum()]):.0f} (smooth) "
    f"vs {Nn[:, 0].mean():.0f} +- {Nn[:, 0].std():.0f} (null) vs {int(Nb[0])} (real)")
say(f"amplitude-adjusted null (residual's own heavy-tailed distribution): D = {Da.mean():.2f} +- {Da.std():.2f}; real at the "
    f"{np.mean(Da < D_real):.0%} quantile; boundary cells {Na[:, 0].mean():.0f} +- {Na[:, 0].std():.0f}")
loc = -np.diff(np.log(Nb)) / np.diff(np.log(eps))
say("local slopes between successive eps (a fractal would hold one value): " + ", ".join(f"{x:.2f}" for x in loc))

# ------------------------------------------------------------------ 4. resolution check
say("\n== 4. resolution check: native window n in [32, 96], lambda in [0.03, 0.35]")
parts = sorted(glob.glob("cache/parts/verify_native_n*.npz"))
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
ax[0].loglog(eps, Nn.mean(0), color="0.6", lw=6, alpha=0.5, label=f"null: smooth + phase-randomised noise (D={Dn.mean():.2f})")
ax[0].loglog(eps, Na.mean(0), color="0.3", lw=1, ls="-.", label=f"null, amplitude-adjusted (D={Da.mean():.2f})")
ax[0].loglog(eps, Ns, "--", color="#3288bd", label=f"smoothed field only (D={D_smooth:.2f})")
ax[0].loglog(eps, Nb, "o-", color="#d53e4f", label=f"measured escape boundary (D={D_real:.2f})")
ax[0].set_xlabel("box size ε (phase-map cells)"); ax[0].set_ylabel("boxes containing boundary"); ax[0].legend(fontsize=8)
ax[0].set_title("97 × 65 replace phase map")
if parts:
    rows = [dict(np.load(p)) for p in parts]
    nat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    have = set(int(x) for x in np.unique(nat["n"]))
    n_hi = 32
    while n_hi + 1 in have and n_hi < 96:
        n_hi += 1  # contiguous range from 32
    esc_nat = ((nat["sw2"] - nat["sw2"][:, :1]).max(1) - TAU) > 0
    lut = {(int(n), int(r), int(sd)): e for n, r, sd, e in zip(nat["n"], nat["nr"], nat["seed"], esc_nat)}
    R = 256
    logn = np.linspace(np.log(32), np.log(n_hi + 0.5), R, endpoint=False)
    lam_ax = np.linspace(0.05, 0.33, R, endpoint=False)
    nn = np.clip(np.round(np.exp(logn)).astype(int), 32, n_hi)

    def native(seed):
        M = np.zeros((R, R), bool)
        for i, lv in enumerate(lam_ax):
            for j, n in enumerate(nn):
                M[i, j] = lut[(n, int(np.floor(lv * n + 1e-9)), seed)]
        return M
    M0, M1 = native(0), native(1)
    # phase-map cells in the same window: nearest lambda row and nearest log-n column
    li = np.abs(lams[None] - lam_ax[:, None]).argmin(1)
    ci = np.abs(np.log(ns)[None] - logn[:, None]).argmin(1)
    Mb = esc[li][:, ci]
    # consistency: base map cell vs native seed-0 chain at the same (n, n_r) must agree exactly
    chk = [(esc[i, j], lut[(int(ns[j]), int(np.floor(lams[i] * ns[j] + 1e-9)), 0)])
           for i in range(len(lams)) for j in range(len(ns))
           if 32 <= ns[j] <= n_hi and 0.05 <= lams[i] < 0.33 and (int(ns[j]), int(np.floor(lams[i] * ns[j] + 1e-9)), 0) in lut]
    agree = np.mean([x == y for x, y in chk])
    say(f"columns done: n = 32..{n_hi}; base-map cells re-measured by the native run agree {agree:.1%} ({len(chk)} cells)")
    epsR = [1, 2, 4, 8, 16, 32, 64]
    Nb_r, N0, N1 = boxcount(boundary(Mb), epsR), boxcount(boundary(M0), epsR), boxcount(boundary(M1), epsR)
    for lab_, N_ in (("phase-map cells", Nb_r), ("native seed 0", N0), ("native seed 1", N1)):
        say(f"{lab_:<16} 256^2 raster box counts eps={epsR}: {N_.astype(int).tolist()}; "
            f"D(1-4 px) = {slope(epsR, N_, 1, 4):.2f}, D(4-32 px) = {slope(epsR, N_, 4, 32):.2f}")
    dis = np.mean(M0 != M1)
    bz = gaussian_filter((boundary(M0) | boundary(M1)).astype(float), 6) > 0.01
    say(f"seed 0 vs seed 1 escape labels disagree on {dis:.1%} of the window, {np.mean((M0 != M1)[bz]):.1%} within 6 px of either boundary")
    # per-column transition lambda* (first n_r above which chains survive) for both seeds
    dl = []
    for n in range(32, n_hi + 1):
        t = []
        for sd in (0, 1):
            rr = sorted(r for (m, r, s_) in lut if m == n and s_ == sd)
            e = np.array([lut[(n, r, sd)] for r in rr])
            surv = np.nonzero(~e)[0]
            t.append(rr[surv[0]] / n if len(surv) else np.nan)
        dl.append(abs(t[0] - t[1]) * n)
    dl = np.array(dl)
    say(f"first surviving n_r per column, |seed 0 - seed 1| in units of 1/n: median {np.nanmedian(dl):.1f}, mean {np.nanmean(dl):.1f}")
    ext = [0, R, 0, R]
    ax[1].imshow(np.flipud(M0.astype(float) - M1.astype(float) * 0.5), cmap="Spectral_r", extent=ext, interpolation="nearest")
    ax[1].set_title(f"native window, seed 0 (±) vs seed 1: disagree {dis:.0%}")
    ax[1].set_xticks([0, R]); ax[1].set_xticklabels(["n=32", f"n={n_hi}"]); ax[1].set_yticks([0, R]); ax[1].set_yticklabels(["λ=0.05", "λ=0.33"])
    ax[2].loglog(epsR, Nb_r, "o-", color="#d53e4f", label="phase-map cells (nearest)")
    ax[2].loglog(epsR, N0, "s-", color="#3288bd", label="native, seed 0")
    ax[2].loglog(epsR, N1, "^-", color="#66c2a5", label="native, seed 1")
    ax[2].loglog(epsR, N0[0] / np.array(epsR), ":", color="0.5", label="slope −1 (smooth curve)")
    ax[2].loglog(epsR, N0[0] / np.array(epsR) ** 2, ":", color="0.8", label="slope −2 (area-filling)")
    ax[2].set_xlabel("box size ε (px of 256² raster)"); ax[2].legend(fontsize=8); ax[2].set_title("resolution check")
else:
    say("native window not computed yet")
fig.tight_layout(); fig.savefig("gallery/verify_boundary.png"); plt.close(fig)

# ------------------------------------------------------------------ 5. accumulate
if os.path.exists("cache/phase_ring_gmm_accumulate.npz"):
    a = np.load("cache/phase_ring_gmm_accumulate.npz")
    swa = a["sw2"][0].astype(float)
    ea = ((swa - swa[..., :1]).max(-1) - TAU) > 0
    say(f"\n== 5. accumulate phase map ({len(a['lams'])} x {len(a['ns'])}): escaped cells {ea.mean():.1%} vs replace {esc.mean():.1%}; "
        f"at lambda = 0: accumulate escaped {ea[0].mean():.0%} of n columns, replace {esc[0].mean():.0%}")
    ratio = swa[..., -1] / swa[..., 0]
    say(f"accumulate sliced W2(G)/W2(0): median {np.median(ratio):.2f}, max {ratio.max():.2f}; replace median "
        f"{np.median(sw[..., -1] / sw[..., 0]):.2f}, max {(sw[..., -1] / sw[..., 0]).max():.2f}")

open("verify_results.txt", "w").write("\n".join(OUT) + "\n")
