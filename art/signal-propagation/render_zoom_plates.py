"""Finite-width frontier renders from zoom chains: single plates, the four-panel magnification
sequence (zoom factor + measured local box-counting slope under each panel), in several styles.

Signed field for the split styles (declared):
  chaotic side  (L_avg > tau): magnitude = log10(L_avg / tau)      (small = close to the frontier)
  ordered side  (L_avg <= tau): magnitude = (D + 1) - t_hit          (t_hit = first depth with L < 1e-10;
                                                                     late convergence = close to the frontier)
Each panel is rank-normalised on its own (as in Sohl-Dickstein's figures).

  python render_zoom_plates.py --chain cache/zoom_B_N100_D1000_s0_f64_r1024.npz --report cache/fractal_report_B_N100.json --panels 0 2 4 6
"""
import argparse, json, os, textwrap
import numpy as np
from render_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--chain", nargs="+", required=True, help="path or path:level (repeatable; mixed chains allowed)")
ap.add_argument("--report", default=None)
ap.add_argument("--panels", type=int, nargs="+", default=None, help="levels, when a single chain is given")
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--label", default=None, help="tau or sync (default: stored in the chain)")
ap.add_argument("--pw", type=int, default=1024, help="panel size on the sheet (px)")
ap.add_argument("--prefix", default="frontier")
ap.add_argument("--styles", default="spectral,aurora,magma,ink,riso")
args = ap.parse_args()
rep = json.load(open(os.path.join(HERE, args.report))) if args.report else None
PAN = []
_cache = {}
for spec in args.chain:
    path, _, lv = spec.partition(":")
    if path not in _cache:
        _cache[path] = np.load(os.path.join(HERE, path))
    Zc = _cache[path]
    levs = [int(lv)] if lv else (args.panels or list(range(Zc["L_avg"].shape[0])))
    for k in levs:
        lab_ = args.label or (str(Zc["label"]) if "label" in Zc else "tau")
        PAN.append(dict(L=Zc["L_avg"][k], t=Zc["t_hit"][k].astype(float), win=Zc["windows"][k], R=Zc["L_avg"].shape[1],
                        dtype=str(Zc["dtype"]), N=int(Zc["N"]), D=int(Zc["D"]), label=lab_, name=os.path.basename(path), level=k))
N, D, LABEL = PAN[0]["N"], PAN[0]["D"], PAN[0]["label"]


def signed(p):
    L, t = p["L"], p["t"]
    if p["label"] == "sync":
        ch = t > p["D"]
        mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / 1e-10), (p["D"] + 1) - t)
    else:
        ch = L > args.tau
        mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / args.tau), (p["D"] + 1) - t)
    return ch, mag


def style_rgb(p, style):
    ch, mag = signed(p)
    L = p["L"]
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
    "spectral": "Sohl-Dickstein Spectral split (declared): purple half = ordered (the pair merges), red half = chaotic; shade = per-side rank of closeness to the frontier (late merging / barely separated = dark).",
    "aurora": "Aurora/ember split palette (declared), same signed field as the Spectral version.",
    "magma": "log10 of the measured output distance L (magma, -16 ... 0.5); black = the two inputs merged to float64 precision.",
    "ink": "Single ink: frontier cells (2x2 neighbourhoods containing both outcomes) inked; chaotic side lightly tinted by rank (declared).",
    "riso": "Two-drum risograph study: teal = ordered, fluorescent orange = chaotic, density = rank of closeness to the frontier; 3/4 px misregistration is declared.",
}


def zoom_label(p):
    return 4.0 / (p["win"][1] - p["win"][0])


def slope_of(p):
    # slope measured on the 256^2 float64 chain at the same zoom factor (report), if available
    if rep is None:
        return None
    for d in rep["levels"]:
        if abs(d["zoom"] - zoom_label(p)) / zoom_label(p) < 1e-6:
            return d["slope"]
    return None


def fit_panel(rgb, pw):
    """Show a native grid in a pw x pw panel: identity if R == pw, area-average if R is a multiple-free larger
    grid (downsampling only; never invent detail), nearest-neighbour integer upscaling otherwise (declared)."""
    R = rgb.shape[0]
    if R == pw:
        return rgb
    from PIL import Image
    im = Image.fromarray(to_uint8(rgb))
    return np.asarray(im.resize((pw, pw), Image.BOX if R > pw else Image.NEAREST)) / 255.0


styles = args.styles.split(",")
for style in styles:
    for p in PAN:
        save_rgb(style_rgb(p, style)[::-1], f"{args.prefix}_N{N}_x{zoom_label(p):.0f}_{p['dtype']}_r{p['R']}_{style}_raw.png")
    pw = args.pw
    s_ = pw / 1024
    gap, top, bot, side_m = int(60 * s_), int(230 * s_), int(380 * s_), int(110 * s_)
    W = side_m * 2 + len(PAN) * pw + (len(PAN) - 1) * gap
    H = top + pw + bot
    fig = fig_px(W, H, bg=BG[style], dpi=int(200 * s_))
    fg = FG[style]
    fig.text(side_m / W, 1 - 90 * s_ / H, f"Finite Width: the order/chaos frontier of one random erf network, N = {N}, depth {D}",
             color=fg, fontsize=30, va="center")
    fig.text(side_m / W, 1 - 150 * s_ / H, "Four magnifications of one fixed random network (common random numbers: the same standard-normal "
             "weights at every pixel, scaled by sigma_w and sigma_b)", color=fg, fontsize=14, va="center", alpha=0.85)
    for i, p in enumerate(PAN):
        x0 = side_m + i * (pw + gap)
        ax = img_axes(fig, x0, top, pw, pw, W, H)
        w = p["win"]
        ax.imshow(fit_panel(style_rgb(p, style), pw), origin="lower", extent=[w[0], w[1], w[2], w[3]], interpolation="nearest")
        if i + 1 < len(PAN):
            wn = PAN[i + 1]["win"]
            side = wn[1] - wn[0]; minside = 0.012 * (w[1] - w[0])
            if side < minside:  # too small to see: draw a marker ring around it
                cx, cy = (wn[0] + wn[1]) / 2, (wn[2] + wn[3]) / 2
                ax.add_patch(plt.Rectangle((cx - minside, cy - minside), 2 * minside, 2 * minside, fill=False, ec=fg, lw=1.2))
            else:
                ax.add_patch(plt.Rectangle((wn[0], wn[2]), side, wn[3] - wn[2], fill=False, ec=fg, lw=1.2))
        ax.set_xlim(w[0], w[1]); ax.set_ylim(w[2], w[3])
        sl = slope_of(p)
        txt = f"x{zoom_label(p):,.0f}" + (f"     box-count slope {sl:.2f}" if sl is not None and np.isfinite(sl) else "")
        fig.text((x0 + pw / 2) / W, 1 - (top + pw + 45 * s_) / H, txt, color=fg, fontsize=17, ha="center", va="center")
        fig.text((x0 + pw / 2) / W, 1 - (top + pw + 85 * s_) / H,
                 f"sigma_w {0.5*(w[0]+w[1]):.9f}   sigma_b {0.5*(w[2]+w[3]):.9f}   side {w[1]-w[0]:.2e}",
                 color=fg, fontsize=10.5, ha="center", va="center", alpha=0.8, family="DejaVu Sans Mono")
        fig.text((x0 + pw / 2) / W, 1 - (top + pw + 115 * s_) / H, f"computed at {p['R']} x {p['R']}, {'float64' if p['dtype']=='f64' else 'float32'}",
                 color=fg, fontsize=10.5, ha="center", va="center", alpha=0.7, family="DejaVu Sans Mono")
    labtxt = (f"Measured: two independent inputs through {D} layers; a pixel is chaotic (red half) if the pair never came within L = |x1 - x2|^2 < 1e-10, "
              "ordered (purple half) if it did." if LABEL == "sync" else
              f"Measured: L = |x1 - x2|^2 after {D} layers (mean of the last 20) for two independent inputs; frontier at L = {args.tau:g}.")
    fig.text(side_m / W, 1 - (top + pw + 170 * s_) / H,
             "\n".join(textwrap.wrap(labtxt + " Every panel computed natively on its own grid (larger grids shown area-averaged to the panel); each zoom centred on the sub-window with the most ordered/chaotic mixing.", int((W - 2 * side_m) / (18 * s_))))
             + "\n" + STYLE_NOTE[style] + "\n" + "\n".join(textwrap.wrap("Slope = local box-counting slope of the frontier at that zoom (256 x 256 float64 chain, box sizes 2 to 32 px). "
             "It rises from ~1 to a peak near 1.87 and falls again: no single fractal dimension. At infinite width the frontier is the smooth mean-field curve (null model slope "
             + (f"{np.nanmean([d['slope'] for d in rep['null']]):.2f}" if rep else "~1") + ").", int((W - 2 * side_m) / (18 * s_)))),
             color=fg, fontsize=12, va="top", linespacing=1.6, alpha=0.9, wrap=False)
    savefig(fig, f"{args.prefix}_N{N}_magnification_{style}.png", dpi=int(200 * s_))
    if len(PAN) == 4:
        # 2x2 poster: panels read left-to-right, top-to-bottom; minimal captions (PIL, pixel-exact)
        from PIL import Image, ImageDraw, ImageFont
        g, m, capH = 24, 70, 170
        Wp = 2 * m + 2 * pw + g
        img = Image.new("RGB", (Wp, m + 2 * pw + g + capH), BG[style])
        dr = ImageDraw.Draw(img)
        fB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 26)
        fS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
        for i, p in enumerate(PAN):
            x0, y0 = m + (i % 2) * (pw + g), m + (i // 2) * (pw + g)
            img.paste(Image.fromarray(to_uint8(fit_panel(style_rgb(p, style), pw)[::-1])), (x0, y0))
            tag = f"x{zoom_label(p):,.0f}"
            dr.rectangle([x0, y0 + pw - 34, x0 + 11 * len(tag) + 24, y0 + pw], fill=BG[style])
            dr.text((x0 + 10, y0 + pw - 30), tag, font=fS, fill=FG[style])
        yc = m + 2 * pw + g + 30
        dr.text((m, yc), f"The order/chaos frontier of one random erf network (width {N}, depth {D}), magnified x1, x16, x1,024, x65,536", font=fB, fill=FG[style])
        dr.text((m, yc + 50), "centres " + ";  ".join(f"({0.5*(p['win'][0]+p['win'][1]):.8f}, {0.5*(p['win'][2]+p['win'][3]):.8f})" for p in PAN[1:]) + "  in (sigma_w, sigma_b)", font=fS, fill=FG[style])
        dr.text((m, yc + 80), "; ".join(f"{zoom_label(p):,.0f}x: {p['R']}^2 {'float64' if p['dtype']=='f64' else 'float32'}" for p in PAN) + ".  " + STYLE_NOTE[style][:120], font=fS, fill=FG[style])
        img.save(os.path.join(GAL, f"{args.prefix}_N{N}_poster_{style}.png"))
    print("wrote", style)
