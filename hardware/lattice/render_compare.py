"""render_compare.py - cross-run plates and statistics (no GPU):
dtype triptych, allow_tf32 diptych, test-retest noise, alignment close-up, alignment statistics -> cache/stats.json.
    OMP_NUM_THREADS=1 taskset -c 0-4 python render_compare.py
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render import *

TAGS = dict(bf16="bf16_g256_k4096", fp16="fp16_g256_k4096", fp32="fp32_g256_k4096", fp32_tf32="fp32_tf32_g256_k4096")


def align_stats(T, lo=33):
    """Median over m of t(n) ratios by residue of n (n, m >= lo), relative to all n in the same row."""
    G = T["G"]; t = T["t_med"].reshape(G, G); v = T["grid"]
    sel = v >= lo
    tt = t[np.ix_(sel, sel)]; nv = v[sel]
    rel = tt / np.median(tt, axis=1, keepdims=True)
    out = {}
    for mod in (2, 4, 8, 16, 32, 64):
        z = nv % mod == 0
        out[f"n%{mod}==0 vs rest"] = float(np.median(rel[:, z]) / np.median(rel[:, ~z]))
    mv = v[sel]; relm = tt / np.median(tt, axis=0, keepdims=True)
    for mod in (2, 8, 16):
        z = mv % mod == 0
        out[f"m%{mod}==0 vs rest"] = float(np.median(relm[z]) / np.median(relm[~z]))
    out["n odd / n even"] = float(np.median(rel[:, nv % 2 == 1]) / np.median(rel[:, nv % 2 == 0]))
    return out


def triptych(runs, stats):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), facecolor=NIGHT)
    for ax, (dt, (T, K)) in zip(axs, runs.items()):
        r, _ = trend_residual(T)
        ax.imshow(P.render_split(r, "sd_spectral"), origin="lower", extent=[.5, 256.5, .5, 256.5], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        nk = len(set(kernel_label(s) for s in K["names"])); nf = len(set(map(tuple, T["fp"])))
        ax.set_title(f"{dt}   {nk} kernels  {nf} fingerprints   odd/even-n time x{stats[dt]['n odd / n even']:.2f}",
                     color="#ddd", fontsize=10, loc="left")
    fig.text(.01, .02, "Residual ln t vs robust overhead+linear fit, per-side rank-normalised Sohl-Dickstein Spectral split at 0 "
             "(purple = faster than trend, red = slower; declared). m up, n right, [1,256]^2, k = 4096.   "
             + stack_line(runs["bf16"][0]["info"]).split(" | bf16")[0] + " | " + runs["bf16"][0]["info"]["stack"]["date"],
             color="#aaa", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.06, wspace=.03)
    fig.savefig(f"{GAL}/dtype_triptych_spectral.png", dpi=130, facecolor=NIGHT); plt.close(fig)
    print("wrote triptych")


def tf32_diptych(off, on):
    T0, K0 = off; T1, K1 = on; G = T0["G"]
    lr = np.log(T0["t_med"] / T1["t_med"]).reshape(G, G)        # > 0: TF32 faster
    save_rgb(P.render_split(-lr, "sd_spectral"), "tf32_ratio_spectral", 8)
    l0 = [kernel_label(s) for s in K0["names"]]; l1 = [kernel_label(s) for s in K1["names"]]
    same = np.array([a == b for a, b in zip(l0, l1)]).reshape(G, G)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), facecolor=PAPER)
    for ax, (T, K, ttl) in zip(axs[:2], [(T0, K0, "allow_tf32 = False"), (T1, K1, "allow_tf32 = True")]):
        kid, kv = categorize([kernel_label(s) for s in K["names"]], G)
        ax.imshow(mosaic_rgb(kid, kernel_colors(kv)), origin="lower", extent=[.5, 256.5, .5, 256.5], interpolation="nearest")
        ax.set_title(f"{ttl}: {len(kv)} kernel labels; top: {short_label(kv[0][0])[:38]}", loc="left", fontsize=9)
    im = axs[2].imshow(np.exp(lr), origin="lower", extent=[.5, 256.5, .5, 256.5], cmap="PuOr_r",
                       norm=matplotlib.colors.LogNorm(vmin=1 / 3, vmax=3), interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[2], fraction=.046, pad=.02); cb.set_label("t(off) / t(on)   (>1: TF32 faster)")
    cb.set_ticks([1 / 3, .5, 1, 2, 3]); cb.set_ticklabels(["1/3", "1/2", "1", "2", "3"]); cb.minorticks_off()
    axs[2].set_title(f"speed-up; same kernel at {100*same.mean():.1f}% of shapes; median x{np.median(np.exp(lr)):.2f}", loc="left", fontsize=9)
    for ax in axs:
        ax.set_xlabel("n"); ax.set_ylabel("m"); ax.set_facecolor(PAPER)
    fig.suptitle("fp32 matmul with and without TF32 - GB10 + cuBLAS " + T0["info"]["stack"]["cublas"], x=.01, ha="left")
    fig.text(.01, .01, stack_line(T0["info"]), fontsize=7.5)
    fig.subplots_adjust(left=.04, right=.97, top=.88, bottom=.1, wspace=.22)
    fig.savefig(f"{GAL}/tf32_diptych_plate.png", dpi=130, facecolor=PAPER); plt.close(fig)
    return dict(tf32_same_kernel_frac=float(same.mean()), tf32_median_speedup=float(np.median(np.exp(lr))),
                tf32_max_speedup=float(np.exp(lr).max()), tf32_min_speedup=float(np.exp(lr).min()))


def retest(T256, R):
    G = R["G"]; a = T256["t_med"].reshape(256, 256)[:G, :G]; b = R["t_med"].reshape(G, G)
    lr = np.log(b / a)
    fa = T256["fp"].reshape(256, 256, -1)[:G, :G].reshape(G * G, -1)
    st = dict(retest_median_abs_log_ratio=float(np.median(np.abs(lr))), retest_frac_gt_5pct=float((np.abs(lr) > np.log(1.05)).mean()),
              retest_fp_identical_frac=float((fa == R["fp"]).all(1).mean()), retest_iqr_median=float(np.median(R["iqr"])),
              retest_ref_std=float(R["ref"][:, 2].std() / np.median(R["ref"][:, 2])))
    return st, lr


def closeup_plate(R, K, lr):
    G = R["G"]; r, _ = trend_residual(R)
    kid, kv = categorize([kernel_label(s) for s in K["names"]], 256)
    kid = kid[:G, :G]
    fig = plt.figure(figsize=(15, 8.4), facecolor=PAPER)
    ax = fig.add_axes([.04, .1, .5, .8]); ax2 = fig.add_axes([.6, .55, .36, .35]); ax3 = fig.add_axes([.6, .1, .36, .33])
    ax.imshow(P.render_split(r, "sd_spectral"), origin="lower", extent=[.5, G + .5, .5, G + .5], interpolation="nearest")
    ed = boundaries(kid)
    yy, xx = np.nonzero(ed)
    for g in range(8, G + 1, 8):
        ax.axvline(g + .5, color=INK, lw=.35 if g % 32 else .9, alpha=.55); ax.axhline(g + .5, color=INK, lw=.35 if g % 32 else .9, alpha=.55)
    ax.set_xticks(range(0, G + 1, 16)); ax.set_yticks(range(0, G + 1, 16)); ax.set_xlabel("n"); ax.set_ylabel("m")
    ax.set_title("[1,128]^2 re-measured (seed 1, 9 windows of 3 ms): residual, Spectral split at 0; hairlines every 8, bold every 32", loc="left", fontsize=9)
    v = R["grid"]; t = R["t_med"].reshape(G, G); sel = v >= 17
    rel = t[np.ix_(sel, sel)] / np.median(t[np.ix_(sel, sel)], axis=1, keepdims=True)
    res = [np.median(rel[:, (v[sel] % 8) == q]) for q in range(8)]
    ax2.bar(range(8), res, color=["#5e4fa2" if q == 0 else ("#3288bd" if q % 2 == 0 else "#9e0142") for q in range(8)])
    ax2.set_ylim(min(res) * .95, max(res) * 1.03); ax2.set_xlabel("n mod 8"); ax2.set_ylabel("t / row median  (m, n >= 17)")
    ax2.set_facecolor(PAPER)
    im = ax3.imshow(lr, origin="lower", cmap="RdBu_r", vmin=-.1, vmax=.1, extent=[.5, G + .5, .5, G + .5], interpolation="nearest")
    fig.colorbar(im, ax=ax3, fraction=.046, pad=.02).set_label("ln(t retest / t first run)")
    ax3.set_title("test-retest (independent random order)", loc="left", fontsize=9); ax3.set_facecolor(PAPER)
    fig.text(.04, .02, "Alignment close-up, bf16. " + stack_line(R["info"]), fontsize=7.5)
    fig.savefig(f"{GAL}/alignment_closeup_plate.png", dpi=150, facecolor=PAPER); plt.close(fig)
    save_rgb(P.render_split(r, "sd_spectral"), "alignment_closeup_spectral", 16)


def main():
    stats = {}
    runs = {dt: (load("time", t), load("kernels", t)) for dt, t in TAGS.items() if exists("time", t) and exists("kernels", t)}
    for dt, (T, K) in runs.items():
        stats[dt] = align_stats(T)
        stats[dt].update(n_kernels=len(set(kernel_label(s) for s in K["names"])), n_fp=len(set(map(tuple, T["fp"]))),
                         iqr_median=float(np.median(T["iqr"])), iqr_p99=float(np.percentile(T["iqr"], 99)),
                         ref_std=float(T["ref"][:, 2].std() / np.median(T["ref"][:, 2])),
                         one_range_us=[float(T["ref"][:, 3].min() * 1e6), float(T["ref"][:, 3].max() * 1e6)],
                         t_us=[float(T["t_med"].min() * 1e6), float(T["t_med"].max() * 1e6)],
                         gflops_max=float(gflops(T).max()), smi_before=T["info"]["smi_before"], smi_after=T["info"]["smi_after"],
                         wall_min=T["info"]["wall_s"] / 60)
    if all(d in runs for d in ("bf16", "fp16", "fp32")):
        triptych({d: runs[d] for d in ("bf16", "fp16", "fp32")}, stats)
    if "fp32" in runs and "fp32_tf32" in runs:
        stats["tf32"] = tf32_diptych(runs["fp32"], runs["fp32_tf32"])
    if exists("time", "bf16_g128_k4096_retest") and "bf16" in runs:
        R = load("time", "bf16_g128_k4096_retest")
        st, lr = retest(runs["bf16"][0], R); stats["retest"] = st
        closeup_plate(R, runs["bf16"][1], lr)
    json.dump(stats, open(f"{CACHE}/stats.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "smi" not in kk} for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
