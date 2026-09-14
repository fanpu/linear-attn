"""render_large.py - stride-13 [13,2041]^2 analysis + plates (no GPU). Writes cache/stats_large.json and
gallery/{cliffs_*, large_triptych_spectral, profiles_plate_*}.  Run after render.py (which makes the per-run s2048 plates):
    OMP_NUM_THREADS=1 taskset -c 0-4 nice python render_large.py
"""
import json
import numpy as np
from scipy.stats import spearmanr
from render import *
from render_compare import align_stats

DTS = ("bf16", "fp16", "fp32")
TAG = "{}_s2048_k4096"


def jumps(r, kid):
    """|delta residual| between neighbours in m (same n) and in n at the same parity (n vs n+26), and whether the
    kernel label changes across that edge. Returns two lists of (J, boundary, i, j, axis)."""
    out = []
    Jm = np.abs(np.diff(r, axis=0)); Bm = kid[1:] != kid[:-1]
    Jn = np.abs(r[:, 2:] - r[:, :-2]); Bn = kid[:, 2:] != kid[:, :-2]
    return (Jm, Bm), (Jn, Bn)


def cliff_stats(T, K, r, kid, kv):
    (Jm, Bm), (Jn, Bn) = jumps(r, kid)
    J = np.concatenate([Jm.ravel(), Jn.ravel()]); B = np.concatenate([Bm.ravel(), Bn.ravel()])
    v = T["grid"]; labs = [l for l, _ in kv]
    top = []
    for arr, bb, ax in ((Jm, Bm, "m"), (Jn, Bn, "n")):
        for flat in np.argsort(np.where(bb, arr, 0).ravel())[::-1][:6]:
            i, j = np.unravel_index(flat, arr.shape)
            i2, j2 = (i + 1, j) if ax == "m" else (i, j + 2)
            top.append(dict(axis=ax, frm=[int(v[i]), int(v[j])], to=[int(v[i2]), int(v[j2])], ratio=float(np.exp(r[i2, j2] - r[i, j])),
                            k_from=short_label(labs[kid[i, j]]), k_to=short_label(labs[kid[i2, j2]])))
    top.sort(key=lambda d: -abs(np.log(d["ratio"])))
    thr = np.log(1.2)
    return dict(edges=int(J.size), boundary_edges=int(B.sum()),
                median_jump_boundary=float(np.exp(np.median(J[B])) - 1), median_jump_interior=float(np.exp(np.median(J[~B])) - 1),
                p90_jump_boundary=float(np.exp(np.percentile(J[B], 90)) - 1), p90_jump_interior=float(np.exp(np.percentile(J[~B], 90)) - 1),
                frac_jumps_gt20pct_on_boundary=float(B[J > thr].mean()) if (J > thr).any() else None,
                n_jumps_gt20pct=int((J > thr).sum()), top_cliffs=top[:8])


def drift_stats(T, r):
    ref = T["ref"]; rt = ref[:, 2] / np.median(ref[:, 2])
    half = len(rt) // 2
    rho, _ = spearmanr(T["pos"], r.ravel())
    return dict(ref_range_pct=float(100 * (rt.max() - rt.min())), ref_std_pct=float(100 * rt.std()),
                ref_first_half_over_second=float(np.median(rt[:half]) / np.median(rt[half:])),
                spearman_residual_vs_order=float(rho),
                iqr_median_pct=float(100 * np.median(T["iqr"])), iqr_p99_pct=float(100 * np.percentile(T["iqr"], 99)),
                t_us=[float(T["t_med"].min() * 1e6), float(np.median(T["t_med"]) * 1e6), float(T["t_med"].max() * 1e6)],
                gflops_max=float(gflops(T).max()), wall_min=T["info"]["wall_s"] / 60,
                smi_before=T["info"]["smi_before"], smi_after=T["info"]["smi_after"])


def cliff_plate(tag, T, r, kid, scale=13):
    """Dark ground: faint log GFLOPS (oslo), kernel boundaries (parity-aware) lit by the residual jump across them (cet_fire)."""
    import colorcet  # noqa: F401
    G = T["G"]; g = P.rank_normalize(np.log(gflops(T)))
    base = plt.get_cmap("cmc.oslo")(.03 + .22 * g)[..., :3]
    big = np.repeat(np.repeat(base, scale, 0), scale, 1)
    (Jm, Bm), (Jn, Bn) = jumps(r, kid)
    lit = np.zeros((G * scale, G * scale))
    for i, j in zip(*np.nonzero(Bm)):                      # edge between rows i and i+1 -> horizontal line
        y = (i + 1) * scale; lit[y - 1:y + 1, j * scale:(j + 1) * scale] = np.maximum(lit[y - 1:y + 1, j * scale:(j + 1) * scale], Jm[i, j])
    for i, j in zip(*np.nonzero(Bn)):                      # same-parity neighbours (j, j+2) -> vertical line at j+1 centre
        x = (j + 1) * scale + scale // 2; lit[i * scale:(i + 1) * scale, x - 1:x + 1] = np.maximum(lit[i * scale:(i + 1) * scale, x - 1:x + 1], Jn[i, j])
    on = lit > 0
    u = np.clip(np.log1p(lit / .02) / np.log1p(.5 / .02), 0, 1)   # 2 % .. 65 % jump -> dim .. white-hot (declared log scale)
    col = plt.get_cmap("cet_fire")(.25 + .75 * u)[..., :3]
    big[on] = col[on]
    save_rgb(big, f"cliffs_fire_{tag}")


def profiles_plate(tag, T, K, kid, kv, cst):
    """Survey sheet: GFLOPS along m for four n (odd, even, n % 8 == 0), step changes marked where the kernel switches."""
    v = T["grid"]; G = T["G"]; g = gflops(T)
    picks = [(1027, "#9e0142"), (1040, "#5e4fa2"), (1014, "#3288bd"), (1521, "#f46d43")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=PAPER, gridspec_kw=dict(width_ratios=[1.35, 1]))
    for n0, c in picks:
        j = int(np.argmin(np.abs(v - n0))); n = int(v[j])
        ax.plot(v, g[:, j], color=c, lw=1.0, drawstyle="steps-mid", label=f"n = {n}  ({'odd' if n % 2 else ('n % 8 = 0' if n % 8 == 0 else 'even')})")
        sw = np.nonzero(kid[1:, j] != kid[:-1, j])[0]
        ax.plot(v[sw + 1], g[sw + 1, j], "|", color=c, ms=9, mew=.8)
    ax.set_xlabel("m"); ax.set_ylabel("GFLOPS  (2 m k n / t)"); ax.set_facecolor(PAPER); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Throughput along m at fixed n; ticks = profiler kernel switch", loc="left", fontsize=9)
    rel = T["t_med"].reshape(G, G) / np.median(T["t_med"].reshape(G, G), axis=1, keepdims=True)
    for q, c, lab in ((1, "#9e0142", "odd n"), (0, "#3288bd", "even n, n % 8 != 0"), (8, "#5e4fa2", "n % 8 == 0")):
        sel = (v % 2 == 1) if q == 1 else ((v % 2 == 0) & (v % 8 != 0) if q == 0 else (v % 8 == 0))
        bx.plot(v, np.median(rel[:, sel], axis=1), color=c, lw=1, label=lab)
    bx.set_xlabel("m"); bx.set_ylabel("t / row median"); bx.set_facecolor(PAPER); bx.legend(frameon=False, fontsize=8)
    bx.set_title("Alignment penalty by n class, per row m", loc="left", fontsize=9)
    fig.text(.05, .015, f"Lattice, stride-13 grid [13, 2041]^2. Jumps across kernel boundaries: median {100*cst['median_jump_boundary']:.1f} %, "
             f"interior {100*cst['median_jump_interior']:.1f} %.  " + stack_line(T["info"]), fontsize=7.5)
    fig.subplots_adjust(left=.05, right=.98, top=.93, bottom=.14, wspace=.18)
    fig.savefig(f"{GAL}/profiles_plate_{tag}.png", dpi=150, facecolor=PAPER); plt.close(fig)


def triptych_large(runs, stats):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), facecolor=NIGHT)
    for ax, (dt, (T, K, r, kid, kv)) in zip(axs, runs.items()):
        ax.imshow(P.render_split(r, "sd_spectral"), origin="lower", extent=[6.5, 2047.5, 6.5, 2047.5], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{dt}   {len(kv)} kernels   odd/even-n time x{stats[dt]['n odd / n even']:.2f}   n%8 x{stats[dt]['n%8==0 vs rest']:.2f}",
                     color="#ddd", fontsize=10, loc="left")
    fig.text(.01, .02, "Stride-13 grid m, n in [13, 2041], k = 4096. Residual ln t vs robust overhead+linear fit, Sohl-Dickstein Spectral split at 0 "
             "(purple = faster than trend; declared).   " + stack_line(runs["bf16"][0]["info"]).split(" | bf16")[0], color="#aaa", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.06, wspace=.03)
    fig.savefig(f"{GAL}/large_triptych_spectral.png", dpi=130, facecolor=NIGHT); plt.close(fig)


def spectral_split(tag, T, r, scale=13):
    """Residual Spectral plate with odd-n and even-n sub-lattices as separate panels (columns doubled), so the parity
    stripes stop dominating and each sub-lattice's cells read."""
    G = T["G"]; W = G * scale; gap = 16
    img = np.ones((W, 2 * W + gap, 3)) * to_rgb(NIGHT)
    for k, off in enumerate((0, 1)):
        sub = P.render_split(r[:, off::2], "sd_spectral", near_boundary="small")
        big = np.repeat(np.repeat(sub, scale, 0), 2 * scale, 1)[:, :W]
        x0 = k * (W + gap); img[:, x0:x0 + big.shape[1]] = big
    save_rgb(img, f"lattice_spectral_split_{tag}")


def main():
    stats, runs = {}, {}
    for dt in DTS:
        tag = TAG.format(dt)
        if not (exists("time", tag) and exists("kernels", tag)):
            continue
        T, K = load("time", tag), load("kernels", tag)
        r, _ = trend_residual(T)
        labs = [kernel_label(s) for s in K["names"]]; kid, kv = categorize(labs, T["G"])
        st = align_stats(T, lo=0)
        st.update(n_kernels=len(kv), n_fp=len(set(map(tuple, T["fp"]))), kernels_top=[(short_label(l), c) for l, c in kv[:8]])
        st.update(drift_stats(T, r)); st["cliffs"] = cliff_stats(T, K, r, kid, kv)
        stats[dt] = st; runs[dt] = (T, K, r, kid, kv)
        cliff_plate(tag, T, r, kid); spectral_split(tag, T, r); profiles_plate(tag, T, K, kid, kv, st["cliffs"])
    if len(runs) == 3:
        triptych_large(runs, stats)
    json.dump(stats, open(f"{CACHE}/stats_large.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "smi" not in kk} for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
