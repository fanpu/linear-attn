#!/usr/bin/env python3
"""
strassen_forensics.py

Question: does the GPU library behind torch `@` (cuBLAS) ever run Strassen's algorithm?

Rounding error answers it. Our own Strassen, built on cuBLAS, is the positive control:
a test that doesn't flag it is worthless.

  fig1_fingerprint.png     accurate digits in every entry of C = AB. Strassen's rounding error
                           follows its recursion tree; a classical algorithm's is uniform.
  fig2_error_vs_scale.png  shrink half of each input by s: Strassen loses 2 accurate digits in
                           C22 per 10x smaller s, a classical algorithm loses none.
  fig3_scorecard.png       that test at one scale plus three bitwise tests, across dtypes and sizes.

Why the library wouldn't use it anyway:

  fig4_time_breakdown.png  GPU kernel time split into matmul and additions
  fig5_speed.png           cuBLAS throughput and Strassen / cuBLAS time vs n

"Accurate digits" of an entry = -log10(|C_hat - C| / (|A||B|)), with C computed in fp64 from the
same low-precision inputs. Miller (1975): any polynomial matmul algorithm meeting the componentwise
bound |C_hat - C| <= c_n u |A||B| must perform at least n^3 multiplications, so passing the scaling
test rules out every subcubic scheme of this kind (Winograd, AlphaTensor-style, ...), not just Strassen.

Usage
  python strassen_forensics.py                        # run everything, write results + figures
  python strassen_forensics.py --big                  # timing grid up to n = 65536
  python strassen_forensics.py --replot               # redraw figures from saved results
  python strassen_forensics.py --device cpu --smoke   # logic check, no GPU needed
"""
import argparse, json, math, os, subprocess, tempfile, textwrap, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

DEV = "cuda"
DIGIT_CAP = 12   # an exactly rounded entry has infinitely many accurate digits; cap before averaging

# ----------------------------------------------------------------------------- style
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAD = "#d03b3b"
COL = {"cublas": "#2a78d6", "s1": "#eb6834", "scut": "#1baf7a"}
LBL = {"cublas": "torch @ (cuBLAS)", "s1": "Strassen, 1 level", "s2": "Strassen, 2 levels",
       "scut": "Strassen, recursive"}
KIND_COL = {"matmul": "#a9a8a1", "add": "#4a3aa7", "other": GRID}
KIND_LBL = {"matmul": "matmul (GEMM) kernels", "add": "elementwise add / subtract kernels", "other": "other kernels"}
MODE_LBL = {"fp32": "fp32", "tf32": "fp32 + TF32", "bf16": "bf16"}
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
})


# ----------------------------------------------------------------------------- algorithms
def strassen(A, B, levels):
    """Strassen (1969) on square matrices, `levels` recursion levels, library matmul at the leaves.
    Per level: 7 half-size products, 18 half-size additions (10 on inputs, 8 on outputs)."""
    if levels == 0:
        return A @ B
    n = A.shape[0]
    h = n // 2
    A11, A12, A21, A22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    B11, B12, B21, B22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    r = lambda X, Y: strassen(X, Y, levels - 1)
    M1 = r(A11 + A22, B11 + B22)
    M2 = r(A21 + A22, B11)
    M3 = r(A11, B12 - B22)
    M4 = r(A22, B21 - B11)
    M5 = r(A11 + A12, B22)
    M6 = r(A21 - A11, B11 + B12)
    M7 = r(A12 - A22, B21 + B22)
    C = torch.empty((n, n), device=A.device, dtype=A.dtype)
    C11, C12, C21, C22 = C[:h, :h], C[:h, h:], C[h:, :h], C[h:, h:]
    torch.add(M1, M4, out=C11); C11.sub_(M5); C11.add_(M7)    # C11 = M1 + M4 - M5 + M7
    torch.add(M3, M5, out=C12)                                  # C12 = M3 + M5
    torch.add(M2, M4, out=C21)                                  # C21 = M2 + M4
    torch.sub(M1, M2, out=C22); C22.add_(M3); C22.add_(M6)    # C22 = M1 - M2 + M3 + M6
    return C


def cutoff_levels(n, n0):
    L, m = 0, n
    while m > n0 and m % 2 == 0:
        m //= 2
        L += 1
    return L


def matmul(alg, A, B, mode):
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    if alg == "cublas":
        return A @ B
    return strassen(A, B, {"s1": 1, "s2": 2}[alg])


# ----------------------------------------------------------------------------- numerics
def make_pair(n, mode, s=None, graded=None, seed=0):
    """A = diag(r) X, B = Y diag(c) with Gaussian X, Y. s: the second half of r and c set to s, so
    C22 is s^2 smaller than C11. graded: smooth scaling r_i = c_i = graded^(i/n)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = torch.randn(n, n, device=DEV, dtype=torch.float64, generator=g)
    Y = torch.randn(n, n, device=DEV, dtype=torch.float64, generator=g)
    r = torch.ones(n, device=DEV, dtype=torch.float64); c = torch.ones_like(r)
    if s is not None:
        r[n // 2:] = s; c[n // 2:] = s
    if graded is not None:
        idx = torch.arange(n, device=DEV, dtype=torch.float64) / n
        r = graded ** idx; c = graded ** idx
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    return (X * r[:, None]).to(dt), (Y * c[None, :]).to(dt)


def accurate_digits(C_hat, A, B):
    """-log10(|C_hat - C| / (|A||B|)) per entry, reference C in fp64 from the exact low-precision inputs."""
    A64, B64 = A.double(), B.double()
    err = (C_hat.double() - A64 @ B64).abs()
    return (-torch.log10(err / (A64.abs() @ B64.abs()))).clamp(max=DIGIT_CAP)


def fingerprint(n, graded):
    block = max(1, n // 256)
    maps, stats = {}, {}
    for alg in ("cublas", "s1", "s2"):
        A, B = make_pair(n, "fp32", graded=graded)
        D = accurate_digits(matmul(alg, A, B, "fp32"), A, B).float()
        stats[alg] = dict(median=float(D.median()), worst=float(D.min()),
                          p001=float(torch.quantile(D.flatten()[::max(1, D.numel() // 2 ** 20)], 0.001)))
        maps[alg] = F.avg_pool2d(D[None, None], block)[0, 0].cpu().numpy()
        print(f"  {alg:6s} median {stats[alg]['median']:5.2f} digits, worst entry {stats[alg]['worst']:5.2f}")
    return maps, dict(n=n, block=block, graded=graded, stats=stats)


def c22_digits(n, mode, alg, s, seed=0):
    A, B = make_pair(n, mode, s=s, seed=seed)
    h = n // 2
    return float(accurate_digits(matmul(alg, A, B, mode), A, B)[h:, h:].median())


def error_vs_scale(n):
    ss = 10.0 ** -np.arange(0, 4.01, 0.5)
    res = dict(n=n, s=ss.tolist())
    for alg in ("cublas", "s1", "s2"):
        res[alg] = [c22_digits(n, "fp32", alg, s) for s in ss]
        print(f"  {alg:6s} " + " ".join(f"{d:5.2f}" for d in res[alg]))
    return res


def bitwise_tests(n, mode, alg, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    A, B = make_pair(n, mode, seed=seed)
    C1 = matmul(alg, A, B, mode); sync()
    C1b = matmul(alg, A, B, mode)
    e = torch.randint(-8, 9, (n,), device=DEV, generator=g)
    d = (2.0 ** e.double()).to(A.dtype)                      # powers of two: scaling is exact
    C2 = matmul(alg, A * d[None, :], B / d[:, None], mode)   # (A D)(D^-1 B) = A B exactly
    I = torch.eye(n, device=DEV, dtype=A.dtype)
    return dict(reproducible=bool(torch.equal(C1, C1b)),
                inner_scaling_invariant=bool(torch.equal(C1, C2)),
                identity_exact=bool(torch.equal(matmul(alg, I, B, mode), B)))


def scorecard(ns, modes, s):
    rows = []
    for mode in modes:
        for n in ns:
            for alg in ("cublas", "s1"):
                if alg == "s1" and n % 2:
                    continue
                balanced, lopsided = c22_digits(n, mode, alg, 1.0), c22_digits(n, mode, alg, s)
                row = dict(mode=mode, n=n, alg=alg, digits_balanced=balanced, digits_lopsided=lopsided,
                           digits_lost=balanced - lopsided, **bitwise_tests(n, mode, alg))
                rows.append(row)
                print(f"  {mode:5s} n={n:5d} {alg:6s} digits {balanced:5.2f} -> {lopsided:5.2f}  "
                      f"repro={row['reproducible']!s:5s} rescale={row['inner_scaling_invariant']!s:5s} "
                      f"identity={row['identity_exact']}")
    return rows


# ----------------------------------------------------------------------------- performance
def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def bench(fn, budget_s=1.0, warmup=2):
    fn(); sync()
    t0 = time.perf_counter(); fn(); sync(); est = time.perf_counter() - t0
    iters = int(min(50, max(3, budget_s / max(est, 1e-6))))
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        if DEV == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e) / 1e3)
        else:
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))   # median: robust to the odd throttled iteration


def other_gpu_processes():
    """Other compute processes on the GPU; their load would contaminate timings."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [l.strip() for l in out.splitlines() if l.strip() and int(l.split(",")[0]) != os.getpid()]


def add_bandwidth(numel):
    x = torch.randn(numel, device=DEV, dtype=torch.bfloat16); y = torch.randn_like(x); z = torch.empty_like(x)
    t = bench(lambda: torch.add(x, y, out=z))
    return 3 * numel * 2 / t        # read x, read y, write z


def perf_sweep(ns, n0):
    rows = []
    for n in ns:
        A = torch.randn(n, n, device=DEV, dtype=torch.bfloat16); B = torch.randn_like(A)
        tl = bench(lambda: A @ B)
        t1 = bench(lambda: strassen(A, B, 1))
        Lc = cutoff_levels(n, n0)
        tc = bench(lambda: strassen(A, B, Lc)) if Lc > 0 else tl
        del A, B
        rows.append(dict(n=n, t_lib=tl, t_s1=t1, t_scut=tc, L_cut=Lc))
        print(f"  n={n:6d}  cuBLAS {tl*1e3:9.2f} ms ({2*n**3/tl/1e12:6.1f} TFLOPS)   "
              f"S1 {t1*1e3:9.2f} ms   S-recursive(L={Lc}) {tc*1e3:9.2f} ms")
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return rows


KINDS = ("matmul", "add", "other")
def kernel_kind(name):
    l = name.lower()
    if any(k in l for k in ("gemm", "nvjet", "cutlass", "xmma", "matmul")):
        return "matmul"
    return "add" if "elementwise" in l else "other"


def kernel_breakdown(n):
    A = torch.randn(n, n, device=DEV, dtype=torch.bfloat16); B = torch.randn_like(A)
    variants = {"cublas": lambda: A @ B, "s1": lambda: strassen(A, B, 1), "s2": lambda: strassen(A, B, 2)}
    res = dict(n=n)
    for name, fn in variants.items():
        fn(); sync()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as p:
            fn(); sync()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.json")
            p.export_chrome_trace(path)
            ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"]
        kinds = [kernel_kind(e["name"]) for e in ev]
        res[name] = dict(ms={k: sum(e["dur"] for e, kk in zip(ev, kinds) if kk == k) / 1e3 for k in KINDS},
                         count={k: kinds.count(k) for k in KINDS},
                         matmul_kernels=sorted({e["name"] for e, kk in zip(ev, kinds) if kk == "matmul"}))
        print(f"  {name:6s} {res[name]['count']}  ms {({k: round(v, 1) for k, v in res[name]['ms'].items()})}")
    return res


# ----------------------------------------------------------------------------- figures
def titled_figure(w, h, title, subtitle):
    """A figure whose top band carries the finding (bold) and what exactly is plotted (muted)."""
    fig = plt.figure(figsize=(w, h))
    t = textwrap.fill(title, int(w * 72 / (0.6 * 12.5)))      # bold glyphs run wider
    s = textwrap.fill(subtitle, int(w * 72 / (0.5 * 9)))
    t_pt = 12.5 * 1.25 * (t.count("\n") + 1)
    s_pt = 9 * 1.3 * (s.count("\n") + 1)
    fig.text(0.0, 1, t, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.0, 1 - (t_pt + 6) / 72 / h, s, fontsize=9, color=INK2, ha="left", va="top", linespacing=1.3)
    fig.set_layout_engine("constrained", rect=(0, 0, 1, 1 - (t_pt + s_pt + 10) / 72 / h))
    return fig


def sup(k):
    """Exponent as Unicode superscript, for plain (non-mathtext) strings: sup(-3) -> '⁻³'."""
    return str(k).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def series(ax, x, y, key, **kw):
    return ax.plot(x, y, marker="o", ms=6, color=COL[key], mec=SURF, mew=1.5, label=LBL[key], **kw)[0]


def key_handle(key):
    """Legend key without the marker's surface ring, which would read as a dashed line."""
    return Line2D([], [], marker="o", ms=6, color=COL[key], mew=0, label=LBL[key])


def fig_fingerprint(fp, maps, out):
    n, b, st = fp["n"], fp["block"], fp["stats"]
    uniform = st["cublas"]["p001"] > st["cublas"]["median"] - 2
    fig = titled_figure(11, 4.9,
        "Strassen's rounding error follows its recursion tree; cuBLAS's is "
        + ("uniform" if uniform else "not uniform either"),
        f"Accurate digits in each entry of C = AB, fp32, n = {n:,}"
        + (f" (each pixel averages a {b}×{b} block)" if b > 1 else "")
        + f". The inputs are graded so that entries of C shrink smoothly by "
        f"10{sup(round(-math.log10(fp['graded'])))} from top-left to bottom-right. "
        "Accurate digits = −log₁₀(|error| / (|A||B|)), with the error measured against fp64.")
    axs = fig.subplots(1, 3)
    cmap = LinearSegmentedColormap.from_list("digits", ["#1a1a19", "#f4f3ef"])
    for ax, alg in zip(axs, ("cublas", "s1", "s2")):
        m = maps[alg]; h = m.shape[0] // 2; pad = 0.03 * m.shape[0]
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
        ax.axhline(h - 0.5, color=AXIS, lw=0.6); ax.axvline(h - 0.5, color=AXIS, lw=0.6)
        for (i, j), name in {(0, 0): "C₁₁", (0, 1): "C₁₂", (1, 0): "C₂₁", (1, 1): "C₂₂"}.items():
            under = m[i * h:i * h + h // 4, j * h:j * h + h // 4].mean()
            ax.text(j * h + pad, i * h + pad, name, fontsize=8.5, ha="left", va="top",
                    color=SURF if under < 4.5 else INK2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AXIS)
        worst = f"{st[alg]['worst']:.1f}".replace("-", "−")
        ax.set_title(f"{LBL[alg]}\nmedian {st[alg]['median']:.1f} digits · worst entry {worst}",
                     loc="left", fontsize=9.5, color=INK, linespacing=1.4)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, aspect=25, ticks=[0, 3, 6, 9])
    cb.ax.set_yticklabels(["0 (none)", "3", "6", "9+"])
    cb.set_label("accurate digits", color=INK2); cb.outline.set_visible(False)
    save(fig, out, "fig1_fingerprint.png")


def fig_error_vs_scale(sl, out):
    s = np.array(sl["s"]); d = {k: np.array(sl[k]) for k in ("cublas", "s1", "s2")}
    tail = s <= 0.1
    slope = np.polyfit(np.log10(s[tail]), d["s1"][tail], 1)[0]
    lib_loss = max(0.0, d["cublas"][0] - d["cublas"].min())
    fig = titled_figure(7.5, 5.2,
        f"Shrinking half of each input costs Strassen {slope:.1f} accurate digits per 10×; cuBLAS loses {lib_loss:.1f}",
        f"Median accurate digits in C₂₂ of fp32 C = AB, n = {sl['n']:,}. The bottom half of A's rows and the right "
        "half of B's columns are multiplied by s, so C₂₂ is s² smaller than C₁₁. Strassen forms "
        "C₂₂ = M₁ − M₂ + M₃ + M₆ by cancelling unscaled terms like A₁₁B₁₁, so rounding error of size 1 lands on a "
        f"result of size s². 2-level Strassen (not drawn) stays within {np.abs(d['s2'] - d['s1']).max():.1f} digits "
        "of 1 level.")
    ax = fig.subplots()
    for k in ("cublas", "s1"):
        series(ax, s, d[k], k)
    ax.annotate(LBL["cublas"], (s[-1], d["cublas"][-1]), xytext=(0, 9), textcoords="offset points",
                ha="right", va="bottom", color=INK2)
    i = len(s) // 2
    ax.annotate(f"{LBL['s1']}\n−{slope:.1f} digits per 10× smaller s", (s[i], d["s1"][i]), xytext=(12, 8),
                textcoords="offset points", ha="left", va="bottom", color=INK2, linespacing=1.4)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "1" if abs(v - 1) < 1e-9 else f"$10^{{{int(round(math.log10(v)))}}}$"))
    ax.set_ylim(0, max(v.max() for v in d.values()) + 1.5)
    ax.set_xlabel("s, the scale applied to half of each input")
    ax.set_ylabel("accurate digits in C₂₂")
    ax.text(0.01, 0.02, "0 = no accurate digits left", transform=ax.transAxes, color=MUTED, fontsize=8.5)
    ax.legend(handles=[key_handle("cublas"), key_handle("s1")], loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, borderaxespad=0)
    save(fig, out, "fig2_error_vs_scale.png")


def fig_scorecard(rows, s_det, out):
    get = {(r["mode"], r["n"], r["alg"]): r for r in rows}
    modes = list(dict.fromkeys(r["mode"] for r in rows)); ns = list(dict.fromkeys(r["n"] for r in rows))
    lib = [r for r in rows if r["alg"] == "cublas"]; st = [r for r in rows if r["alg"] == "s1"]
    lib_max = max(abs(r["digits_lost"]) for r in lib)
    st_lo, st_hi = min(r["digits_lost"] for r in st), max(r["digits_lost"] for r in st)
    title = (f"In all {len(lib)} dtype × size configurations, cuBLAS loses {lib_max:.1f} digits where "
             f"Strassen loses {st_lo:.1f}–{st_hi:.1f}")
    if all(r["inner_scaling_invariant"] for r in lib) and not any(r["inner_scaling_invariant"] for r in st):
        title += ", and only cuBLAS survives exact rescaling"
    k = int(round(-math.log10(s_det)))
    sub = (f"Left: the fig 2 test at one scale, the drop in median accurate digits of C₂₂ from s = 1 to "
           f"s = 10{sup(-k)}. Right: bitwise tests. D is a diagonal of random powers of two, so (AD)(D⁻¹B) = AB exactly "
           "in floating point; a classical algorithm computes the same bits, while Strassen's sums like "
           "A₁₁ + A₂₂ mix differently scaled entries and round differently.")
    tf32_id = [r for r in lib if not r["identity_exact"]]
    if tf32_id and all(r["mode"] == "tf32" and r["inner_scaling_invariant"] for r in tf32_id):
        sub += " TF32 multiplies at reduced precision, so I·B ≠ B bitwise there even for a classical algorithm."

    y, ys, heads = 0.0, {}, []
    for mode in modes:
        heads.append((mode, y)); y += 1.0
        for n in ns:
            if (mode, n, "cublas") in get:
                ys[(mode, n)] = y; y += 1
        y += 0.4
    fig = titled_figure(10.5, 0.27 * y + 2.6, title, sub)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2.2])
    ax = fig.add_subplot(gs[0]); tb = fig.add_subplot(gs[1], sharey=ax)

    theory = 2 * k
    for (mode, n), yy in ys.items():
        a, b = get[(mode, n, "cublas")], get.get((mode, n, "s1"))
        if b:
            ax.plot([a["digits_lost"], b["digits_lost"]], [yy, yy], color=AXIS, lw=1.2, zorder=1)
            ax.plot(b["digits_lost"], yy, "o", ms=7, color=COL["s1"], mec=SURF, mew=1.5, zorder=3)
        ax.plot(a["digits_lost"], yy, "o", ms=7, color=COL["cublas"], mec=SURF, mew=1.5, zorder=3)
    for mode, yy in heads:
        ax.text(0, yy + 0.35, MODE_LBL[mode], transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontweight="bold", color=INK)
    ax.axvline(theory, color=INK2, lw=0.8, zorder=0)
    ax.annotate(f"s⁻² prediction: {theory}", (theory, 0), xycoords=("data", "axes fraction"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set_yticks(list(ys.values()), [f"n = {n:,}" for (_, n) in ys])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.6, theory + 0.8); ax.set_ylim(y - 0.2, -2.0)
    ax.set_xlabel(f"accurate digits lost in C₂₂ when half of each input is scaled by $s = 10^{{-{k}}}$")
    ax.legend(handles=[key_handle("cublas"), key_handle("s1")], loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, borderaxespad=0)

    tb.set_xlim(0, 9); tb.axis("off")
    tests = [("reproducible", "same bits\non a rerun"), ("inner_scaling_invariant", "unchanged by\n(AD)(D⁻¹B)"),
             ("identity_exact", "I·B = B\nexactly")]
    for t, (key, head) in enumerate(tests):
        cx = 3 * t + 1.5
        tb.text(cx, -1.25, head, ha="center", va="bottom", fontsize=8.5, color=INK2, linespacing=1.3)
        for dx, alg in ((-0.55, "cublas"), (0.55, "s1")):
            tb.plot(cx + dx, -0.75, "o", ms=5, color=COL[alg], mew=0)
            for (mode, n), yy in ys.items():
                r = get.get((mode, n, alg))
                if r:
                    tb.text(cx + dx, yy, "✓" if r[key] else "✗", ha="center", va="center", fontsize=11,
                            color=INK2 if r[key] else BAD)
    save(fig, out, "fig3_scorecard.png")


def fig_time_breakdown(kb, out):
    n, algs = kb["n"], ("cublas", "s1", "s2")
    tot = {k: sum(kb[k]["ms"].values()) for k in algs}
    pct = 100 * kb["s2"]["ms"]["add"] / tot["s2"]
    fig = titled_figure(9.5, 3.6,
        f"At n = {n:,}, 2-level Strassen spends {pct:.0f}% of its GPU time on additions and takes "
        f"{tot['s2'] / tot['cublas']:.1f}× as long as cuBLAS",
        "Summed GPU kernel time for one bf16 multiply, from torch.profiler. Each Strassen level replaces one matmul "
        "with 7 half-size matmuls plus 18 half-size additions, so 2 levels run 7² = 49 matmuls and "
        "18 + 7·18 = 144 additions.")
    ax = fig.subplots()
    used = [kind for kind in KINDS if any(kb[k]["ms"][kind] > 0 for k in algs)]
    for y, k in enumerate(algs):
        left = 0.0
        for kind in used:
            w = kb[k]["ms"][kind]
            if w > 0:
                ax.barh(y, w, left=left, height=0.5, color=KIND_COL[kind], edgecolor=SURF, linewidth=1.5)
                left += w
        c = kb[k]["count"]
        parts = [f"{c[kind]} {'add' if kind == 'add' else kind}" for kind in KINDS if c[kind]]
        ax.text(left + 0.012 * max(tot.values()), y,
                f"{left:.1f} ms  ·  " + " + ".join(parts) + (" kernel" if sum(c.values()) == 1 else " kernels"),
                va="center", ha="left", color=INK2)
    ax.set_yticks(range(len(algs)), [LBL[k] for k in algs]); ax.invert_yaxis()
    ax.set_xlim(0, 1.5 * max(tot.values())); ax.set_xlabel("GPU kernel time (ms)")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=KIND_COL[kind], label=KIND_LBL[kind]) for kind in used],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(used), borderaxespad=0)
    save(fig, out, "fig4_time_breakdown.png")


def fig_speed(res, out):
    rows = res["timing"]
    n = np.array([r["n"] for r in rows], float)
    tf = 2 * n ** 3 / np.array([r["t_lib"] for r in rows]) / 1e12
    r1 = np.array([r["t_s1"] / r["t_lib"] for r in rows])
    rc = np.array([r["t_scut"] / r["t_lib"] for r in rows])
    rec = np.array([r["L_cut"] > 1 for r in rows])   # at 1 level it is the same algorithm as the orange line
    ip = int(tf.argmax())
    wins = [i for i in range(len(n)) if r1[i] < 1]
    if wins:
        title = "1-level Strassen beats cuBLAS only at n = " + ", ".join(f"{int(n[i]):,}" for i in wins)
        drop = tf[ip] / tf[wins[0]]
        if drop > 1.5:
            title += f", where cuBLAS throughput has fallen {drop:.1f}× below its peak"
    else:
        title = f"Strassen is slower than cuBLAS at every size tested, up to n = {int(n[-1]):,}"
    sub = (f"bf16 on {res['device']}, median of repeated timings. Top: cuBLAS throughput, counting 2n³ FLOPs per "
           f"multiply. Bottom: time relative to cuBLAS; the recursive variant halves until blocks are "
           f"≤ {res['cutoff']:,}, and is drawn only where that takes 2+ levels. A cost model that assumes cuBLAS always runs at its peak throughput puts the "
           f"1-level break-even at n ≈ {res['predicted_crossover_n']:,.0f}.")
    if res.get("timing_contention"):
        sub += " WARNING: other processes were using the GPU during timing, so these numbers are unreliable."
    fig = titled_figure(8, 6.6, title, sub)
    ax0, ax1 = fig.subplots(2, 1, sharex=True, height_ratios=[1, 1.3])

    series(ax0, n, tf, "cublas")
    ax0.set_ylim(0, tf.max() * 1.25); ax0.set_ylabel("cuBLAS throughput\n(trillion FLOPs / s)")
    ax0.annotate(f"peak {tf[ip]:.0f}", (n[ip], tf[ip]), xytext=(0, 8), textcoords="offset points",
                 ha="center", va="bottom", color=INK2)
    if ip != len(n) - 1 and tf[-1] < tf[ip] / 1.5:
        ax0.annotate(f"{tf[-1]:.0f}, {tf[ip] / tf[-1]:.1f}× below peak", (n[-1], tf[-1]), xytext=(0, 9),
                     textcoords="offset points", ha="right", va="bottom", color=INK2)
    ax0.legend(handles=[key_handle(k) for k in ("cublas", "s1", "scut")], loc="lower left",
               bbox_to_anchor=(0, 1.02), ncol=3, borderaxespad=0)

    series(ax1, n, r1, "s1")
    if rec.any():
        series(ax1, n[rec], rc[rec], "scut")
    ax1.axhline(1, color=INK2, lw=1, zorder=1)
    for txt, off, va in (("↑ Strassen slower", 4, "bottom"), ("↓ Strassen faster", -4, "top")):
        ax1.annotate(txt, (0.01, 1), xycoords=("axes fraction", "data"), xytext=(0, off), textcoords="offset points",
                     va=va, ha="left", color=MUTED, fontsize=8.5)
    ends = [(r1[-1], "s1")] + ([(rc[rec][-1], "scut")] if rec.any() else [])
    for rank, (v, k) in enumerate(sorted(ends)):
        low = rank == 0 and len(ends) > 1
        ax1.annotate(LBL[k], (n[-1], v), xytext=(0, -10 if low else 9), textcoords="offset points",
                     ha="right", va="top" if low else "bottom", color=INK2)
    vals = np.concatenate([r1, rc[rec], [1.0]])
    lo, hi = vals.min() / 1.6, vals.max() * 1.6
    ax1.set_yscale("log", base=2); ax1.set_ylim(lo, hi)
    ax1.yaxis.set_major_locator(FixedLocator([2.0 ** e for e in range(math.floor(math.log2(lo)),
                                                                        math.ceil(math.log2(hi)) + 1)]))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×")); ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_ylabel("time ÷ cuBLAS time")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_locator(FixedLocator([v for v in n if int(v) & (int(v) - 1) == 0]))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlabel("matrix size n")
    save(fig, out, "fig5_speed.png")


def plot_all(res, maps, out):
    fig_fingerprint(res["fingerprint"], maps, out)
    fig_error_vs_scale(res["error_vs_scale"], out)
    fig_scorecard(res["scorecard"], res["s_det"], out)
    if res.get("kernels"):
        fig_time_breakdown(res["kernels"], out)
    fig_speed(res, out)


# ----------------------------------------------------------------------------- main
def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--big", action="store_true", help="extend the timing grid to n = 65536")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes; checks logic, not performance")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved results, no compute")
    ap.add_argument("--cutoff", type=int, default=2048)
    ap.add_argument("--out", default="strassen_forensics_out")
    a = ap.parse_args()
    DEV = a.device
    os.makedirs(a.out, exist_ok=True)
    res_path, maps_path = os.path.join(a.out, "results.json"), os.path.join(a.out, "fingerprint.npz")
    if a.replot:
        plot_all(json.load(open(res_path)), dict(np.load(maps_path)), a.out)
        print(f"redrew figures -> {a.out}/")
        return

    if a.smoke:
        ns = [256, 512, 768, 1024]; fp_n = 256; det_ns = [256, 300]; bw_numel = 2 ** 22; kb_n = None
    else:
        ns = [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
        if a.big:
            ns += [24576, 32768, 40960, 49152, 57344, 65536]
        fp_n = 1024; det_ns = [512, 1000, 2048, 4096]; bw_numel = 2 ** 28; kb_n = 8192
    res = dict(device=torch.cuda.get_device_name() if DEV == "cuda" else "cpu", torch=torch.__version__,
               cutoff=a.cutoff, s_det=1e-3)

    print("[1/5] fingerprint: accurate digits per entry, graded inputs")
    maps, res["fingerprint"] = fingerprint(fp_n, graded=1e-6)
    if res["fingerprint"]["stats"]["cublas"]["median"] < 6:   # fp32 should give ~8; cf. Day 1 TF32 surprise
        print("  WARNING: cuBLAS fp32 keeps < 6 digits. TF32 may be active despite 'highest'.")
    print("[2/5] accurate digits in C22 vs input scale s")
    res["error_vs_scale"] = error_vs_scale(fp_n)
    print("[3/5] scorecard: scaling test + bitwise tests")
    res["scorecard"] = scorecard(det_ns, ["fp32", "tf32", "bf16"], res["s_det"])
    if DEV == "cuda" and kb_n:
        print("[4/5] kernel time breakdown")
        res["kernels"] = kernel_breakdown(kb_n)
    print("[5/5] timing")
    res["timing_contention"] = other_gpu_processes() if DEV == "cuda" else []
    if res["timing_contention"]:
        print("  WARNING: other GPU processes are running; timings will be unreliable:", res["timing_contention"])
    bw = add_bandwidth(bw_numel)
    res["timing"] = perf_sweep(ns, a.cutoff)
    P = max(2 * r["n"] ** 3 / r["t_lib"] for r in res["timing"])
    # 1 level saves 1/8 of the matmul, 2n^3/(8P) seconds at peak throughput P, and costs 18 additions of
    # (n/2)^2 bf16 entries touching 3 arrays each, 27n^2/BW seconds: break-even at n* = 108 P / BW.
    res.update(add_GBps=bw / 1e9, peak_TFLOPS=P / 1e12, predicted_crossover_n=108 * P / bw)
    print(f"  add bandwidth {bw/1e9:.1f} GB/s, peak {P/1e12:.1f} TFLOPS, model break-even n* = {108*P/bw:,.0f}")

    json.dump(res, open(res_path, "w"), indent=2, default=str)
    np.savez_compressed(maps_path, **maps)
    plot_all(res, maps, a.out)
    print(f"done -> {a.out}/")


if __name__ == "__main__":
    main()
