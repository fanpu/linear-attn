"""Finite-width frontier renders from zoom chains: single plates, the four-panel magnification
sequence (zoom factor + measured local box-counting slope under each panel), in several styles.

Signed field for the split styles (declared):
  chaotic side  (L_avg > tau): magnitude = log10(L_avg / tau)      (small = close to the frontier)
  ordered side  (L_avg <= tau): magnitude = (D + 1) - t_hit          (t_hit = first depth with L < 1e-10;
                                                                     late convergence = close to the frontier)
Each panel is rank-normalised on its own (as in Sohl-Dickstein's figures).

  python render_zoom_plates.py --chain cache/zoom_B_N100_D1000_s0_f64_r1024.npz --report cache/fractal_report_B_N100.json --panels 0 2 4 6
"""
import argparse, json, os
import numpy as np
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--chain", required=True)
ap.add_argument("--report", default=None)
ap.add_argument("--panels", type=int, nargs="+", default=[0, 2, 4, 6])
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--prefix", default="frontier")
ap.add_argument("--styles", default="spectral,aurora,magma,ink,riso")
args = ap.parse_args()
Z = np.load(os.path.join(HERE, args.chain))
N, D = int(Z["N"]), int(Z["D"])
rep = json.load(open(os.path.join(HERE, args.report))) if args.report else None
wins = Z["windows"]


def signed(k):
    L = Z["L_avg"][k]; t = Z["t_hit"][k].astype(float)
    ch = L > args.tau
    mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / args.tau), (D + 1) - t)
    return ch, mag


def style_rgb(k, style):
    ch, mag = signed(k)
    L = Z["L_avg"][k]
    if style == "spectral":
        return split_rgb(ch, mag, "sd_spectral", near_boundary="small")
    if style == "aurora":
        return split_rgb(ch, mag, "aurora_ember", near_boundary="small")
    if style == "magma":
        v = np.log10(np.maximum(L, 1e-30))
        lo, hi = -16, 0.5
        return plt.get_cmap("magma")(np.clip((v - lo) / (hi - lo), 0, 1))[..., :3]
    if style == "ink":
        # single ink on paper: boundary cells (ordered/chaotic 2x2 mixing) in ink, chaotic side a light
        # tint whose density is the within-side rank (declared)
        from sp_core import boundary_mask
        e = boundary_mask(ch)
        v = P.signed_rank_normalize(np.where(ch, 1, -1) * np.maximum(mag, 1e-300), near_boundary="small", pastel=1.0)
        tint = np.where(ch, 0.10 + 0.25 * (1 - np.abs(v)), 0.0)
        cov = np.maximum(tint, e * 0.95)
        return P.overprint([cov], [INK])
    if style == "riso":
        v = P.signed_rank_normalize(np.where(ch, 1, -1) * np.maximum(mag, 1e-300), near_boundary="small", pastel=1.0)
        cov_o = np.where(v < 0, 1 - np.abs(v), 0) ** 1.6
        cov_c = np.where(v >= 0, 1 - np.abs(v), 0) ** 1.6
        cov_c = np.roll(cov_c, (3, -4), axis=(0, 1))  # declared misregistration
        return P.overprint([cov_o * 0.9, cov_c * 0.9], [P.RISO["teal"], P.RISO["fluo_orange"]])
    raise ValueError(style)


BG = {"spectral": "#111014", "aurora": "#07080b", "magma": "#07060a", "ink": PAPER, "riso": P.RISO_PAPER}
FG = {"spectral": "#e9e4da", "aurora": "#e9e4da", "magma": "#e9e4da", "ink": INK, "riso": INK}
STYLE_NOTE = {
    "spectral": "Sohl-Dickstein Spectral split (declared): purple half = ordered (pairs of inputs merge), red half = chaotic; shade = per-side rank of closeness to the frontier.",
    "aurora": "Aurora/ember split palette (declared), same signed field as the Spectral version.",
    "magma": "log10 of the measured output distance L (magma, -16 ... 0.5); black = the two inputs merged to float64 precision.",
    "ink": "Single ink: frontier cells (2x2 neighbourhoods containing both outcomes) inked; chaotic side lightly tinted by rank (declared).",
    "riso": "Two-drum risograph study: teal = ordered, fluorescent orange = chaotic, density = rank of closeness to the frontier; 3/4 px misregistration is declared.",
}


def zoom_label(k):
    side = wins[k][1] - wins[k][0]
    return 4.0 / side


def slope_of(k):
    if rep is None:
        return None
    for d in rep["levels"]:
        if d["level"] == k:
            return d["slope"]
    return None


styles = args.styles.split(",")
R = Z["L_avg"].shape[1]
for style in styles:
    # ---- single native-resolution plates (every level), raw PNG
    for k in range(Z["L_avg"].shape[0]):
        if k in args.panels or k == 0:
            save_rgb(style_rgb(k, style)[::-1], f"{args.prefix}_N{N}_level{k}_{style}_raw.png")
    # ---- four-panel magnification sequence
    ks = args.panels
    pw = R  # panels are shown 1:1 with the native grid (no resampling)
    s_ = pw / 1024
    gap, top, bot, side_m = 60, 230, 360, 110
    W = side_m * 2 + len(ks) * pw + (len(ks) - 1) * gap
    H = top + pw + bot
    fig = fig_px(W, H, bg=BG[style], dpi=int(200 * s_))
    fg = FG[style]
    fig.text(side_m / W, 1 - 90 / H, f"Finite Width: the order/chaos frontier of a random erf network, N = {N}, depth {D}",
             color=fg, fontsize=30, va="center")
    fig.text(side_m / W, 1 - 150 / H, "Four magnifications of one fixed random network (common random numbers: the same standard-normal "
             "weights at every pixel, scaled by sigma_w and sigma_b)", color=fg, fontsize=14, va="center", alpha=0.85)
    for i, k in enumerate(ks):
        x0 = side_m + i * (pw + gap)
        ax = img_axes(fig, x0, top, pw, pw, W, H)
        w = wins[k]
        ax.imshow(style_rgb(k, style), origin="lower", extent=[w[0], w[1], w[2], w[3]], interpolation="nearest")
        # mark the next panel's window on this panel
        if i + 1 < len(ks):
            wn = wins[ks[i + 1]]
            ax.add_patch(plt.Rectangle((wn[0], wn[2]), wn[1] - wn[0], wn[3] - wn[2], fill=False, ec=fg, lw=1.0))
        ax.set_xlim(w[0], w[1]); ax.set_ylim(w[2], w[3])
        sl = slope_of(k)
        txt = f"x{zoom_label(k):,.0f}" + (f"     box-count dim {sl:.2f}" if sl is not None and np.isfinite(sl) else "")
        fig.text((x0 + pw / 2) / W, 1 - (top + pw + 45) / H, txt, color=fg, fontsize=17, ha="center", va="center")
        fig.text((x0 + pw / 2) / W, 1 - (top + pw + 85) / H,
                 f"sigma_w {0.5*(w[0]+w[1]):.9f}   sigma_b {0.5*(w[2]+w[3]):.9f}   side {w[1]-w[0]:.2e}",
                 color=fg, fontsize=10.5, ha="center", va="center", alpha=0.8, family="DejaVu Sans Mono")
    fig.text(side_m / W, 1 - (top + pw + 150) / H,
             f"Measured: L = |x1 - x2|^2 after {D} layers (mean of the last 20) for two independent inputs; frontier at L = {args.tau:g}. "
             f"Each panel computed natively at {R} x {R} in float64 (no upsampling); each zoom centred on the sub-window with the most ordered/chaotic mixing.\n"
             + STYLE_NOTE[style] + "\nDimension = local box-counting slope inside that panel (box sizes 2 to " + str(R // 8) +
             " px). Width matters: at infinite width the frontier is the smooth mean-field curve (null model, same pipeline: dimension "
             + (f"{np.nanmean([d['slope'] for d in rep['null']]):.2f}" if rep else "~1") + ").",
             color=fg, fontsize=12, va="top", linespacing=1.6, alpha=0.9)
    savefig(fig, f"{args.prefix}_N{N}_magnification_{style}.png", dpi=int(200 * s_))
    print("wrote", style)
