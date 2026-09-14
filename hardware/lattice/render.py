"""render.py - every Lattice plate from cache/ (no GPU). Run pinned to efficiency cores during sweeps:
    OMP_NUM_THREADS=1 taskset -c 0-4 python render.py [--only NAME ...]
"""
import argparse, collections, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from PIL import Image
from scipy.ndimage import binary_dilation
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P
from common import *

INK = "#1b1b1b"; PAPER = "#f3eee3"; NIGHT = "#0b0b0e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6,
                     "xtick.major.width": .5, "ytick.major.width": .5})


def exists(kind, tag):
    return os.path.exists(f"{CACHE}/{kind}_{tag}.npz")


def trend_residual(d, iters=30):
    """ln t minus a robust (IRLS least-absolute-deviation, i.e. median) fit of t ~ a + b m + c n + e mn
    in relative error (overhead + linear compute). Declared detrending: the seam r = 0 means 'as fast
    as the smooth size trend predicts'; half the shapes fall on each side by construction."""
    m, n, t = d["m"].astype(float), d["n"].astype(float), d["t_med"]
    F = np.stack([np.ones_like(m), m, n, m * n], 1)
    w = np.ones_like(t)
    for _ in range(iters):
        X = F / t[:, None] * w[:, None]
        coef, *_ = np.linalg.lstsq(X, w, rcond=None)
        rel = np.abs(F @ coef / t - 1)
        w = 1 / np.sqrt(np.maximum(rel, 1e-3))
    fit = F @ coef
    return (np.log(t) - np.log(np.maximum(fit, 1e-9))).reshape(d["G"], d["G"]), coef


def short_label(lab):
    for a, b in (("cutlass:", ""), ("tensorop_", "to_"), ("bf16_", ""), ("f16_", ""), ("_nn_", "_"),
                 ("tmaAB_", ""), ("_NNNN", ""), (" +splitK", " +sK"), ("s161616gemm", "s161616"),
                 ("s16816gemm", "s16816"), ("s1688gemm", "s1688"), ("simt_sgemm_", "sgemm_")):
        lab = lab.replace(a, b)
    return lab


def save_rgb(img, name, scale=1):
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)[::-1]          # origin lower-left
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    Image.fromarray(a).save(f"{GAL}/{name}.png", optimize=True)
    print("wrote", name, a.shape)


FAMILY = [  # (substring, base hue colours light->dark) : declared categorical scheme by kernel family
    ("s1688gemm", ["#f6c28b", "#e8914f", "#c5602b", "#8f3b1b"]),            # cutlass_75 tensorop, align1
    ("s16816gemm", ["#a9cfe8", "#5e9fd0", "#2f6aa8", "#1b3f73"]),           # cutlass_80 tensorop, align2/8
    ("wmma", ["#bfe3b0", "#7fbf73", "#43904a", "#23602f", "#d9ef8b"]),      # cutlass wmma
    ("simt_sgemm", ["#f2b6c6", "#d9738f", "#a8425f", "#6e2340"]),           # fp32 cutlass simt
    ("nvjet", ["#d7c3ef", "#b08fdb", "#8a60c4", "#6440a0", "#e7b0e6", "#c070b9", "#91408f", "#4b2a78",
               "#cfb0ff", "#7a5bd6", "#a36bd8", "#5a3190"]),                   # nvjet (cublasLt JIT-ish)
    ("gemmSN", ["#f7e08a", "#d6b23c"]),                                     # small-n gemm
    ("gemvx", ["#d9d9d9", "#b0b0b0", "#8a8a8a", "#666666", "#c8c0b0"]),     # vector cases
    ("dot_kernel", ["#fafafa"]), ("", ["#555555", "#777777", "#999999"]),
]


def kernel_colors(vocab):
    used = collections.defaultdict(int); cols = []
    for lab, _ in vocab:
        for key, pal in FAMILY:
            if key in lab:
                cols.append(pal[used[key] % len(pal)]); used[key] += 1; break
    return cols


def mosaic_rgb(ids, cols, edges=None, edge_col="#000000"):
    lut = np.array([to_rgb(c) for c in cols])
    img = lut[ids]
    if edges is not None:
        img[edges] = to_rgb(edge_col)
    return img


def boundaries(ids):
    e = np.zeros(ids.shape, bool)
    e[1:, :] |= ids[1:, :] != ids[:-1, :]; e[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    return e


def upsample_edges(ids, s):
    """Boundaries drawn as 1-px hairlines on an s-times upsampled grid."""
    big = np.repeat(np.repeat(ids, s, 0), s, 1)
    return boundaries(big)


def label_fp_by_kernel(T, K):
    """Fingerprint classes coloured by their majority kernel family, lightness by class."""
    labs = [kernel_label(s) for s in K["names"]]
    fl = fp_labels(T["fp"]); fid, fv = categorize(fl, T["G"])
    maj = collections.defaultdict(collections.Counter)
    for l, f in zip(labs, fl):
        maj[f][l] += 1
    vocab = [(maj[f].most_common(1)[0][0], c) for f, c in fv]
    return fid, fv, kernel_colors(vocab)


# ------------------------------------------------------------------------------------------ plates
def plate_spectral(tag, T, scale=8):
    r, coef = trend_residual(T)
    img = P.render_split(r, "sd_spectral", near_boundary="small")
    save_rgb(img, f"lattice_spectral_{tag}", scale)
    for pair in ("aurora_ember", "cyanotype_vandyke"):
        save_rgb(P.render_split(r, pair, near_boundary="small"), f"lattice_{pair}_{tag}", scale)
    return r


def plate_dark(tag, T, scale=8):
    g = np.log10(gflops(T))
    u = P.rank_normalize(g)
    save_rgb(plt.get_cmap("cmc.lajolla_r" if "cmc.lajolla_r" in plt.colormaps() else "magma")(u)[..., :3],
             f"throughput_dark_{tag}", scale)


def plate_mosaic(tag, T, K, scale=8):
    labs = [kernel_label(s) for s in K["names"]]
    kid, kv = categorize(labs, T["G"])
    cols = kernel_colors(kv)
    big = np.repeat(np.repeat(kid, scale, 0), scale, 1)
    save_rgb(mosaic_rgb(big, cols), f"dispatch_mosaic_{tag}")
    ed = boundaries(big)
    line = np.ones(big.shape + (3,)) * to_rgb(PAPER); line[ed] = to_rgb(INK)
    save_rgb(line, f"dispatch_lines_{tag}")
    fid, fv, fcols = label_fp_by_kernel(T, K)
    save_rgb(mosaic_rgb(np.repeat(np.repeat(fid, scale, 0), scale, 1), fcols), f"fingerprint_mosaic_{tag}")
    return kid, kv, cols, fid, fv


def plate_riso(tag, T, K, lo=1, hi=128, scale=16, shift=(3, -2)):
    """Two-ink riso: blue coverage = slower-than-trend (residual > 0, rank-normalised), pink = kernel
    boundaries (from profiler names), printed 1-bit with Bayer dithering and a declared misregistration."""
    r, _ = trend_residual(T)
    kid, _ = categorize([kernel_label(s) for s in K["names"]], T["G"])
    sl = slice(lo - 1, hi)
    r, kid = r[sl, sl], kid[sl, sl]
    cov = np.clip(P.rank_normalize(r), 0, 1)
    covb = np.repeat(np.repeat(cov, scale, 0), scale, 1)
    b = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32
    H, W = covb.shape
    ink1 = (covb > np.tile(b, (H // 4 + 1, W // 4 + 1))[:H, :W]).astype(float)
    ed = binary_dilation(upsample_edges(kid, scale), iterations=1).astype(float)
    ed = np.roll(ed, shift, (0, 1))
    img = P.overprint([ink1 * .92, ed * .95], ["#0078bf", "#ff48b0"], paper=PAPER)
    save_rgb(img, f"riso_closeup_{tag}_{lo}-{hi}")


def plate_science(tag, T, K, name=None):
    G, v = T["G"], T["grid"]
    labs = [kernel_label(s) for s in K["names"]]
    kid, kv = categorize(labs, G); cols = kernel_colors(kv)
    r, coef = trend_residual(T)
    ext = [v[0] - .5 * (v[1] - v[0]), v[-1] + .5 * (v[1] - v[0])] * 2
    fig = plt.figure(figsize=(16, 13.2), facecolor=PAPER)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .7], hspace=.16, wspace=.28, left=.05, right=.98, top=.9, bottom=.13)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for a in ax:
        a.set_facecolor(PAPER); a.set_xlabel("n  (columns of C)"); a.set_ylabel("m  (rows of C)")
    im = ax[0].imshow(T["t_med"].reshape(G, G) * 1e6, origin="lower", extent=ext, cmap="cmc.batlow", norm=matplotlib.colors.LogNorm(), interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[0], fraction=.046, pad=.02); cb.set_label("microseconds per call", labelpad=2)
    gf = gflops(T)
    ax[0].set_title(f"a  time per call (GFLOPS=2mnk/t: {gf.min():.2g} to {gf.max():.3g})", loc="left")
    im = ax[1].imshow(r, origin="lower", extent=ext, cmap="RdBu_r", vmin=-.4, vmax=.4, interpolation="nearest")
    fig.colorbar(im, ax=ax[1], fraction=.046, pad=.02).set_label("ln t - ln(a + bm + cn + d mn)", labelpad=2)
    ax[1].set_title("b  residual vs robust overhead+linear fit (red = slower)", loc="left")
    ax[2].imshow(mosaic_rgb(kid, cols), origin="lower", extent=ext, interpolation="nearest")
    ax[2].set_title(f"c  kernel that ran (torch.profiler), {len(kv)} labels", loc="left")
    fid, fv, fcols = label_fp_by_kernel(T, K)
    ax[3].imshow(mosaic_rgb(fid, fcols), origin="lower", extent=ext, interpolation="nearest")
    ax[3].set_title(f"d  bitwise output fingerprint, {len(fv)} classes", loc="left")
    lax = fig.add_subplot(gs[:, 2]); lax.axis("off")
    for i, (lab, c) in enumerate(kv[:40]):
        y = 1 - i * .024
        lax.add_patch(plt.Rectangle((0, y - .012), .06, .018, color=cols[i], transform=lax.transAxes, clip_on=False))
        lax.text(.08, y - .003, f"{c:6d}  {short_label(lab)}", transform=lax.transAxes, fontsize=6.4,
                 va="center", family="DejaVu Sans Mono")
    # drift inset
    ref = T["ref"]
    iax = fig.add_axes([.73, .15, .24, .07]); iax.set_facecolor(PAPER)
    iax.plot(ref[:, 1] / 60, ref[:, 2] / np.median(ref[:, 2]), lw=.7, color=INK, label=f"ref {T['info']['args']['ref']}")
    iax.plot(ref[:, 1] / 60, ref[:, 3] / np.median(ref[:, 3]), lw=.7, color="#c5602b", label="1x1 overhead")
    iax.set_xlabel("minutes", fontsize=7); iax.set_ylabel("t / median", fontsize=7); iax.tick_params(labelsize=6)
    iax.legend(fontsize=6, frameon=False, ncol=2)
    info = T["info"]; sb, sa = info["smi_before"], info["smi_after"]
    fig.suptitle(f"Lattice: torch.mm(A, B), A in R^(m x {info['stack']['k']}), B in R^({info['stack']['k']} x n)  |  "
                 f"{T['info']['stack']['dtype']}{' + TF32' if info['stack']['allow_tf32'] else ''}", x=.05, ha="left", fontsize=15)
    fig.text(.05, .925, "A portrait of GB10 + this cuBLAS version, not of 'the GB10'. " + stack_line(info), fontsize=8.5)
    noise = np.median(T["iqr"]) * 100
    fig.text(.05, .06, f"{G}x{G} shapes, randomized order, warmup 3, per-call time = median of 5 windows of >=1.5 ms; "
             f"median IQR/median {noise:.2f}%, p99 {np.percentile(T['iqr'], 99)*100:.1f}%.  Reference shape re-timed every "
             f"{info['args']['ref_every']} shapes: std {100*ref[:,2].std()/np.median(ref[:,2]):.1f}%.\n"
             f"nvidia-smi before: util {sb['util']}%, {sb['temp']} C, {sb['power']} W;  after: util {sa['util']}%, {sa['temp']} C, "
             f"{sa['power']} W; no other compute processes. Wall {info['wall_s']/60:.1f} min. Fingerprint = bits of C entries for two "
             f"cancellation probes (A rows identical, B columns identical).\nColours: (a) batlow log scale; (b) diverging RdBu; (c,d) declared "
             f"categorical scheme by kernel family (orange cutlass_75 align1, blue cutlass_80 align2/8, green wmma, violet nvjet, grey gemv).",
             fontsize=7.5, va="top")
    fig.savefig(f"{GAL}/{name or 'plate_' + tag}.png", dpi=150, facecolor=PAPER)
    plt.close(fig); print("wrote plate", tag)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    tags = a.tags or ["bf16_g256_k4096", "fp16_g256_k4096", "fp32_g256_k4096", "fp32_tf32_g256_k4096",
                      "bf16_s2048_k4096", "fp16_s2048_k4096", "fp32_s2048_k4096"]
    os.makedirs(GAL, exist_ok=True)
    want = lambda x: a.only is None or x in a.only
    for tag in tags:
        if not (exists("time", tag) and exists("kernels", tag)):
            continue
        T, K = load("time", tag), load("kernels", tag)
        sc = 8 if T["G"] == 256 else 13
        if want("spectral"): plate_spectral(tag, T, sc)
        if want("dark"): plate_dark(tag, T, sc)
        if want("mosaic"): plate_mosaic(tag, T, K, sc)
        if want("science"): plate_science(tag, T, K)
        if want("riso") and T["G"] == 256: plate_riso(tag, T, K)


if __name__ == "__main__":
    main()
