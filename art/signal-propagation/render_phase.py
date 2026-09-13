"""Part A renders: the mean-field phase diagram of a deep tanh MLP, analytic and measured.

  python render_phase.py [--analytic cache/meanfield_tanh.npz] [--empirical cache/empirical_tanh.npz]
"""
import argparse, os
import numpy as np
from render_common import *
from matplotlib.colors import LogNorm
from scipy.ndimage import shift as ndshift

ap = argparse.ArgumentParser()
ap.add_argument("--analytic", default="cache/meanfield_tanh.npz")
ap.add_argument("--empirical", default="cache/empirical_tanh.npz")
ap.add_argument("--train", default="cache/trainability.npz")
ap.add_argument("--only", default="")
args = ap.parse_args()
A = np.load(os.path.join(HERE, args.analytic))
E = np.load(os.path.join(HERE, args.empirical)) if os.path.exists(os.path.join(HERE, args.empirical)) else None
ext = [A["sw2"][0] - (A["sw2"][1] - A["sw2"][0]) / 2, A["sw2"][-1] + (A["sw2"][1] - A["sw2"][0]) / 2,
       A["sb2"][0] - (A["sb2"][1] - A["sb2"][0]) / 2, A["sb2"][-1] + (A["sb2"][1] - A["sb2"][0]) / 2]
want = lambda k: (not args.only) or k in args.only.split(",")


def empirical_sign(E):
    """chaotic = measured chi_1 > 1 (infinitesimal-pair growth); where chi_1 could not be fitted,
    fall back to the measured correlation fixed point c*_emp < 1 - 1e-3."""
    ch = E["chi1"] > 1
    nan = ~np.isfinite(E["chi1"])
    ch[nan] = E["cstar"][nan] < 1 - 1e-3
    return ch


def plate(rgb, name, title, caption, extent, bg=PAPER, fg=INK, W=3000, line=None, img_interp="nearest"):
    H_img = rgb.shape[0] / rgb.shape[1]
    iw = W - 520
    ih = int(iw * H_img)
    H = ih + 520
    fig = fig_px(W, H, bg=bg)
    ax = fig.add_axes([300 / W, 1 - (190 + ih) / H, iw / W, ih / H])
    ax.imshow(rgb, origin="lower", extent=extent, aspect="auto", interpolation=img_interp)
    if line is not None:
        ax.plot(line[0], line[1], color=line[2], lw=line[3], ls=line[4] if len(line) > 4 else "-")
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    for s in ax.spines.values():
        s.set_color(fg); s.set_linewidth(0.8)
    ax.tick_params(colors=fg, labelsize=11, length=4, width=0.6)
    ax.set_xlabel(r"weight variance  $\sigma_w^2$", color=fg, fontsize=14)
    ax.set_ylabel(r"bias variance  $\sigma_b^2$", color=fg, fontsize=14)
    fig.text(300 / W, 1 - 95 / H, title, color=fg, fontsize=26, va="center")
    import textwrap
    caption = "\n".join(textwrap.fill(par, 185) for par in caption.replace("\n", " ").split("  ")) if len(caption) > 0 else caption
    fig.text(300 / W, 1 - (ih + 330) / H, caption, color=fg, fontsize=11.5, va="top", linespacing=1.5, alpha=0.9)
    return savefig(fig, name)


LINE = (A["line_sw2"], A["line_sb2"])
xi_a = A["xi_c"]; ch_a = A["chi1"] > 1

# ---------------------------------------------------------------- A1 Spectral split (analytic, hero)
if want("spectral"):
    rgb = split_rgb(ch_a, xi_a, "sd_spectral", near_boundary="large")
    plate(rgb, "phase_spectral_analytic.png", "Order and Chaos: the tanh phase diagram (infinite width, analytic)",
          "Each pixel: mean-field theory by Gauss-Hermite quadrature (2400 x 1200 grid). Side = sign of chi_1 - 1 "
          "(purple half: ordered, chi_1 < 1; red half: chaotic, chi_1 > 1).\nShade = correlation depth scale xi_c, "
          "rank-normalised separately on each side (declared Sohl-Dickstein Spectral split); the dark seam is where "
          "xi_c diverges.\nNo line is drawn: the seam is the data.", ext)
    save_rgb(rgb[::-1], "phase_spectral_analytic_raw.png")
    if E is not None:
        ch_e = empirical_sign(E)
        rgb = split_rgb(ch_e, E["xi_c"], "sd_spectral", near_boundary="large")
        plate(rgb, "phase_spectral_measured.png", "Order and Chaos, measured (width 1000, 6 random nets per pixel)",
              f"Each pixel: {int(E['K'])} random tanh MLPs of width N = {int(E['N'])}, depth {int(E['D'])} (common random numbers across pixels). "
              "\nSide = measured growth rate of an infinitesimal input perturbation (chi_1 > 1: red). Shade = measured "
              "correlation depth scale xi_c (exponential fit of |c^l - c*| vs depth),\nrank-normalised per side "
              "(declared Spectral split). 480 x 240 pixels, nearest-neighbour enlargement. No line drawn; black = fit failed.",
              ext)
        for pair in ("aurora_ember", "indigo_madder"):
            rgb = split_rgb(ch_e, E["xi_c"], pair, near_boundary="large")
            plate(rgb, f"phase_split_{pair}_measured.png", f"Order and Chaos, measured ({P.PAIRINGS[pair]['title']})",
                  f"Same measurement as the Spectral plate (N = {int(E['N'])}, depth {int(E['D'])}, {int(E['K'])} nets/pixel); declared split palette "
                  f"'{pair}' from color-research/palettes.py.\nSide = measured chi_1 > 1; shade = per-side rank of measured xi_c.", ext)

# ---------------------------------------------------------------- A2 dark ridge (measured, no line)
if want("ridge") and E is not None:
    import matplotlib.cm as cm
    xi = E["xi_c"]
    norm = LogNorm(vmin=0.5, vmax=np.nanpercentile(xi, 99.7))
    cmap = plt.get_cmap("magma")
    rgb = cmap(norm(np.nan_to_num(xi, nan=0.5)))[..., :3]
    rgb[~np.isfinite(xi)] = 0
    cap = (f"Measured correlation depth scale xi_c (log colour, magma) of random tanh MLPs, width N = {int(E['N'])}, depth {int(E['D'])}, "
           f"{int(E['K'])} nets per pixel.\nThe bright ridge is where signals take longest to forget their input correlation: it emerges from "
           "the fits, nothing is drawn. Black = fit failed.")
    plate(rgb, "phase_ridge_magma_measured.png", "The ridge of the edge of chaos (measured)", cap, ext, bg="#0b0a0c", fg="#e8e2d6")
    plate(rgb, "phase_ridge_magma_measured_with_analytic_line.png", "Measured ridge + analytic critical line (declared overlay)",
          cap.replace("nothing is drawn", "the dashed cyan line is the separately computed mean-field chi_1 = 1 line, overlaid") ,
          ext, bg="#0b0a0c", fg="#e8e2d6", line=(LINE[0], LINE[1], "#6fe3ff", 1.2, (0, (4, 3))))

# ---------------------------------------------------------------- A3 paper contours (analytic)
if want("contour"):
    W, H = 3000, 1880
    fig = fig_px(W, H, bg=PAPER)
    ax = fig.add_axes([300 / W, 1 - 1500 / H, 2480 / W, 1240 / H])
    ax.set_facecolor(PAPER)
    X, Y = np.meshgrid(A["sw2"], A["sb2"])
    lv = np.geomspace(0.75, 400, 34)
    ax.contour(X, Y, np.log(xi_a), levels=np.log(lv), colors=INK, linewidths=np.where(np.arange(34) % 5 == 0, 0.9, 0.35), negative_linestyles="solid")
    ax.contour(X, Y, np.where(ch_a, A["cstar"], np.nan), levels=np.linspace(0.05, 0.95, 19), colors="#b8473a",
               linewidths=0.35, linestyles="dashed")
    for s in ax.spines.values():
        s.set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)
    ax.set_xlabel(r"$\sigma_w^2$", color=INK, fontsize=15); ax.set_ylabel(r"$\sigma_b^2$", color=INK, fontsize=15)
    fig.text(300 / W, 1 - 95 / H, "Survey sheet: isolines of the depth scale", color=INK, fontsize=26, va="center")
    fig.text(300 / W, 1 - 1640 / H, "Black ink: level lines of the analytic correlation depth scale xi_c (34 geometric levels from 0.75 "
             "to 400 layers; every fifth line heavier).\nThey crowd toward the critical line where xi_c diverges. "
             "Red dashed ink: level lines of the correlation fixed point c* (0.05 ... 0.95) in the chaotic phase.\n"
             "Tanh MLP, infinite width, Gauss-Hermite quadrature. Line weights and inks are declared choices.",
             color=INK, fontsize=11.5, va="top", linespacing=1.5)
    savefig(fig, "phase_contours_paper.png")

# ---------------------------------------------------------------- A4 two-ink riso (measured)
if want("riso") and E is not None:
    ch_e = empirical_sign(E)
    v = P.signed_rank_normalize(np.where(ch_e, 1, -1) * np.nan_to_num(E["xi_c"], nan=1e-9), near_boundary="large", pastel=1.0)
    k = 6
    up = lambda a: np.kron(a, np.ones((k, k)))
    cov_o = up(np.where(v < 0, 1 - np.abs(v), 0)) ** 1.4   # ink density highest at the seam
    cov_c = up(np.where(v >= 0, 1 - np.abs(v), 0)) ** 1.4
    rng = np.random.default_rng(3)
    grain = lambda c: np.clip(c + 0.10 * rng.standard_normal(c.shape), 0, 1)
    cov_c = ndshift(grain(cov_c), (5, -7), order=0, mode="nearest")   # declared misregistration
    rgb = P.overprint([grain(cov_o) * 0.92, cov_c * 0.92], [P.RISO["medium_blue"], P.RISO["fluo_pink"]])
    plate(rgb, "phase_riso_measured.png", "Order / Chaos, two-drum risograph study",
          f"Measured (N = {int(E['N'])}, depth {int(E['D'])}, {int(E['K'])} nets/pixel). Medium-blue ink: ordered side; fluorescent-pink ink: chaotic side. "
          "Ink density = within-side rank of the measured xi_c, heaviest at the edge of chaos.\nDeclared: simulated overprint, "
          "grain noise and a 5/7-pixel misregistration of the pink drum are aesthetic, not data.", ext, img_interp="nearest")

# ---------------------------------------------------------------- A5 atlas: analytic vs measured
if want("atlas") and E is not None:
    rows = [("q*", "qstar", "cmc.oslo", None), ("chi_1", "chi1", "cmc.vik", (0.2, 1.8)), ("c*", "cstar", "cmc.lajolla_r", (0, 1)),
            ("xi_q (layers)", "xi_q", "cmc.batlow", None), ("xi_c (layers, log)", "xi_c", "magma", "log")]
    W, H = 3000, 3300
    fig = fig_px(W, H, bg=PAPER)
    fig.text(0.05, 0.975, "Atlas: mean-field theory (left) against measurement at width 1000 (right)", fontsize=24, color=INK, va="top")
    import matplotlib.colors as mcolors
    for i, (lab, key, cmap, rng_) in enumerate(rows):
        a, e = A[key], E[key]
        if rng_ == "log":
            norm = LogNorm(vmin=0.5, vmax=300)
        elif rng_ is None:
            norm = mcolors.Normalize(vmin=np.nanpercentile(a, 0.5), vmax=np.nanpercentile(a, 99.5))
        else:
            norm = mcolors.Normalize(*rng_)
        for j, (f, t) in enumerate(((a, "analytic"), (e, "measured"))):
            ax = fig.add_axes([0.07 + j * 0.45, 0.955 - (i + 1) * 0.184, 0.40, 0.150])
            im = ax.imshow(f, origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
            ax.tick_params(labelsize=8, colors=INK)
            ax.set_title(f"{lab}, {t}", fontsize=12, color=INK, loc="left")
        cax = fig.add_axes([0.93, 0.955 - (i + 1) * 0.184 + 0.02, 0.012, 0.125])
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8)
    fig.text(0.05, 0.012, "Axes: sigma_w^2 (horizontal, 0.5-4.5), sigma_b^2 (vertical, 0-2). Measured: random tanh MLPs, "
             f"N = {int(E['N'])}, depth {int(E['D'])}, {int(E['K'])} nets per pixel;\nq*, c* are tail means, xi_q, xi_c exponential fits, "
             "chi_1 the growth rate of an infinitesimal perturbation. White = not measurable (perturbation fell below float32 floor / "
             "transient shorter than one layer).", fontsize=11, color=INK)
    savefig(fig, "phase_atlas_analytic_vs_measured.png")

# ---------------------------------------------------------------- A6 trainability overlay
if want("train") and os.path.exists(os.path.join(HERE, args.train)):
    T = np.load(os.path.join(HERE, args.train))
    acc, sw2s, depths = T["acc"], T["sw2"], T["depths"]
    sb2 = float(T["sb2"])
    row = np.argmin(np.abs(A["sb2"] - sb2))
    W, H = 2400, 1600
    fig = fig_px(W, H, bg=PAPER)
    ax = fig.add_axes([0.1, 0.2, 0.84, 0.7])
    ax.set_facecolor(PAPER)
    xe = np.r_[sw2s[0] - (sw2s[1] - sw2s[0]) / 2, (sw2s[1:] + sw2s[:-1]) / 2, sw2s[-1] + (sw2s[-1] - sw2s[-2]) / 2]
    de = np.r_[0, (depths[1:] + depths[:-1]) / 2, depths[-1] + (depths[-1] - depths[-2]) / 2]
    m = ax.pcolormesh(xe, de, acc, cmap="cmc.lajolla_r", vmin=0.1, vmax=1.0, edgecolors=PAPER, linewidth=2)
    for i, d in enumerate(depths):
        for j, s in enumerate(sw2s):
            ax.text(s, d, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK if acc[i, j] < 0.6 else "white")
    ax.plot(A["sw2"], 6 * A["xi_c"][row], color=INK, lw=1.6)
    ax.set_ylim(de[0], de[-1]); ax.set_xlim(xe[0], xe[-1])
    ax.set_xlabel(r"$\sigma_w^2$", fontsize=14); ax.set_ylabel("depth L", fontsize=14)
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.01).set_label("training accuracy", fontsize=11)
    fig.text(0.1, 0.95, f"Does 6 xi_c predict trainability? (sigma_b^2 = {sb2})", fontsize=22, color=INK)
    fig.text(0.1, 0.08, f"Cells: training accuracy on 10k MNIST training images after {int(T['steps'])} SGD steps (momentum 0.9, lr 1e-3, batch 128), "
             f"\ntanh MLP width {int(T['width'])}, one seed per cell. Line: 6 x analytic xi_c at sigma_b^2 = {sb2} (Schoenholz et al. 2017 predict "
             "networks deeper than ~6 xi_c are untrainable).\nCells are centred on the trained sigma_w^2 values; their widths are not to scale.", fontsize=11, color=INK, va="top")
    savefig(fig, "trainability_check.png")
print("ok")
