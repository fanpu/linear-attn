"""Phase maps over (n = dataset size, lambda = real fraction kept), replace vs accumulate.

usage: python render_phase.py <target> <model> [--tag ""] [--G G] [--tau 0.35]
Reads cache/phase_<target>_<model>_{replace,accumulate}<tag>.npz, writes gallery/phase_*.png.

Measured per cell (one chain, seed 0, common random numbers across cells):
  E   = sliced W2 to the true distribution at generation G
  T   = escape generation: first g with sliced W2 > tau, linearly interpolated between g-1 and g
        (the analogue of a smoothed escape-time count; declared)
Rendered as nearest-neighbour cells (each cell = one chain; no smoothing).
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
import style as S

ap = argparse.ArgumentParser()
ap.add_argument("target"); ap.add_argument("model")
ap.add_argument("--tag", default=""); ap.add_argument("--G", type=int, default=0)
ap.add_argument("--tau", type=float, default=0.25); ap.add_argument("--cell", type=int, default=0)
ap.add_argument("--only", default=""); ap.add_argument("--regimes", default="replace,accumulate")
a = ap.parse_args()


def load(regime):
    d = np.load(f"cache/phase_{a.target}_{a.model}_{regime}{a.tag}.npz")
    return {k: d[k] for k in d.files}


def escape(sw, tau):
    """sw [..., G+1] -> smooth escape generation (inf if never)."""
    G = sw.shape[-1] - 1
    above = sw > tau
    first = np.where(above.any(-1), above.argmax(-1), -1)
    T = np.full(sw.shape[:-1], np.inf)
    idx = np.nonzero(first >= 0)
    for ij in zip(*idx):
        g = first[ij]
        if g == 0:
            T[ij] = 0.0
        else:
            s0, s1 = sw[ij][g - 1], sw[ij][g]
            T[ij] = g - 1 + (tau - s0) / max(s1 - s0, 1e-9)
    return T


def fields(D, G):
    # excess over the chain's own generation-0 fit (identical across lambda at fixed n under CRN),
    # so the finite-n error of a single fit is not mistaken for self-consumption
    sw = D["sw2"][0, :, :, :G + 1].astype(float)  # [L, Nn, G+1]
    E = sw[..., G] / sw[..., 0]
    sw = sw - sw[..., :1]
    T = escape(sw, a.tau)
    margin = a.tau - sw.max(-1)  # how close a surviving chain came to the threshold
    signed = np.where(np.isfinite(T), (G - T) + 1e-3, -np.maximum(margin, 1e-6))  # >=0 escaped, <0 stable
    return sw, E, T, signed


sets = [(k, load(k)) for k in a.regimes.split(",")]
R = sets[0][1]
G = a.G or (R["sw2"].shape[-1] - 1)
lams, ns = R["lams"], R["ns"]
F = {k: fields(D, min(G, D["sw2"].shape[-1] - 1)) for k, D in sets}
L_, N_ = len(lams), len(ns)


def to_img(field_rgb, cy, cx):
    """[L, Nn, 3] with row 0 = lambda 0 -> image with lambda increasing upward, nearest upsample."""
    im = np.flipud(field_rgb)
    return np.kron(im, np.ones((cy, cx, 1)))


def upsample_to(rgb, shape_cells):
    """Nearest-neighbour upsample a coarser grid (nested) onto the replace grid."""
    L2, N2 = rgb.shape[:2]
    ri = np.round(np.linspace(0, L2 - 1, shape_cells[0])).astype(int)
    ci = np.round(np.linspace(0, N2 - 1, shape_cells[1])).astype(int)
    return rgb[ri][:, ci]


def plate(panels, style, name, title, sub, cbar=None):
    cell = a.cell or max(8, 1200 // max(L_, N_))
    cx = cell * 2 if len(panels) == 1 else cell  # single panel: declared 2:1 cell aspect
    pw, ph = N_ * cx, L_ * cell
    ml, mr, mt, mb, gap = 210, 90, 290, 250, 120
    npan = len(panels)
    W = ml + npan * pw + (npan - 1) * gap + mr
    H = mt + ph + mb
    dark = style == "dark"
    img = np.tile(np.array(S.INK["night"] if dark else S.INK["paper"], float), (H, W, 1)) if dark else S.paper_texture(H, W)
    fg = S.INK["bone"] if dark else S.INK["iron_gall"]
    dim = (140, 132, 118) if dark else (112, 102, 90)
    for i, (lab, rgb) in enumerate(panels):
        x = ml + i * (pw + gap)
        if rgb.shape[:2] != (L_, N_):
            rgb = upsample_to(rgb, (L_, N_))
        img[mt:mt + ph, x:x + pw] = to_img(rgb, cell, cx)
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    ft, fs, fl, fa = S.font("serif", 54), S.font("serif_it", 30), S.font("serif", 36), S.font("serif", 26)
    tsz = 54
    while ft.getlength(title) > W - ml - 40 and tsz > 30:
        tsz -= 2; ft = S.font("serif", tsz)
    dr.text((ml, 60), title, font=ft, fill=fg)
    dr.text((ml, 140), sub, font=fs, fill=dim)
    for i, (lab, _) in enumerate(panels):
        x = ml + i * (pw + gap)
        dr.text((x, mt - 60), lab, font=fl, fill=fg)
        dr.rectangle([x - 1, mt - 1, x + pw, mt + ph], outline=dim, width=1)
        for nt in [8, 16, 32, 64, 128, 256, 512, 1024]:
            if ns[0] <= nt <= ns[-1]:
                xx = x + (np.log(nt) - np.log(ns[0])) / (np.log(ns[-1]) - np.log(ns[0])) * (pw - cx) + cx / 2
                dr.line([xx, mt + ph, xx, mt + ph + 12], fill=dim, width=2)
                dr.text((xx - 18, mt + ph + 18), str(nt), font=fa, fill=dim)
        dr.text((x + pw / 2 - 170, mt + ph + 60), "n, samples per generation (log)", font=fa, fill=dim)
        for lt in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = mt + ph - (lt - lams[0]) / (lams[-1] - lams[0]) * (ph - cell) - cell / 2
            dr.line([x - 12, yy, x, yy], fill=dim, width=2)
            if i == 0:
                dr.text((x - 80, yy - 15), f"{lt:.2f}", font=fa, fill=dim)
    dr.text((40, mt + ph / 2 + 160), "λ, real fraction", font=fa, fill=dim)
    if cbar is not None:
        cb_rgb, lo_lab, hi_lab, cap = cbar
        cw, chh = 700, 26
        x0, y0 = ml, H - 110
        strip = np.repeat(np.repeat(cb_rgb[None], chh, 0), max(1, cw // len(cb_rgb)), 1)
        im.paste(Image.fromarray(strip.astype(np.uint8)), (x0, y0))
        dr.text((x0, y0 + 34), lo_lab, font=fa, fill=dim)
        dr.text((x0 + strip.shape[1] - 120, y0 + 34), hi_lab, font=fa, fill=dim)
        dr.text((x0 + strip.shape[1] + 40, y0 - 4), cap, font=fa, fill=dim)
    S.save(im, name)


MODEL = {"gmm": "Gaussian mixture (K = 8, EM)", "kde": "Gaussian KDE (LOO-CV bandwidth)"}[a.model]
base = f"phase_{a.target}_{a.model}{a.tag}" + ("_" + a.regimes.replace(",", "-") if len(sets) == 1 else "")
lab = {"replace": "replace: train on λ·n real + (1−λ)·n fresh samples", "accumulate": "accumulate: keep every sample ever made"}

# ---- 1. sequential: log sliced W2 at generation G, shared scale
if not a.only or "seq" in a.only:
    logs_ = {k: np.log10(F[k][1]) for k, _ in sets}
    lo = min(np.percentile(v, 0.5) for v in logs_.values()); hi = max(np.percentile(v, 99.5) for v in logs_.values())
    for cmap, style in [("crameri_batlow", "dark"), ("crameri_lajolla", "dark"), ("sepia", "paper")]:
        Lut = S.lut(cmap)
        if style == "paper":
            Lut = Lut[::-1] if Lut[0].sum() < Lut[-1].sum() else Lut
        pans = [(lab[k], S.apply_lut((logs_[k] - lo) / (hi - lo), Lut)) for k, _ in sets]
        plate(pans, style, f"{base}_sw2_{cmap}.png", f"Where the snake eats its tail — {MODEL}",
              f"how much worse than its own generation-0 fit after {G} generations: sliced W2(G) / sliced W2(0), log colour.\nEach cell is one chain of {G} refits (seed 0, common random numbers across cells).",
              cbar=(Lut[::8], f"×{10**lo:.2f}", f"×{10**hi:.1f}", f"W2 ratio, generation {G} vs 0"))

# ---- 2. split: escaped (red side, coloured by how late) vs survived (purple side, by how close it came)
if not a.only or "split" in a.only:
    for pairing in ["sd_spectral", "aurora_ember", "hubble_sho"]:
        pans = []
        ref = np.concatenate([F[k][3].ravel() for k, _ in sets])
        for k, _ in sets:
            rgb = S.P.render_split(F[k][3], pairing, near_boundary="small", ref=ref) * 255
            pans.append((lab[k], rgb))
        plate(pans, "dark", f"{base}_split_{pairing}.png", f"Escape-time map of self-consumption — {MODEL}",
              f"red side: sliced W2 rose more than τ = {a.tau} above generation 0 within {G} generations (paler = earlier).\npurple side: never did (paler = further below τ). Each side rank-normalised, shared across panels; declared mapping.")
