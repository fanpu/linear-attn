"""Small multiples: one pipeline, four systems (+ many-init overlays).

Measured: visited iterates per pixel (fixed declared init unless 'overlay'), Lyapunov exponent strips.
  logistic map (reference / null), GD on 1/2(x1x2-1)^2, GD on 1/2(x1x2x3x4-1)^2,
  deep linear network 1/2||W3W2W1 - M||^2 (Ghosh et al. target: singular values 10, 6, 3), aligned and random init.
Overlays: 48 log-normal inits (scalar products), 17 random-init seeds (DLN); brightness = summed counts.
Aesthetic: tone curve, fire palette, layout.
"""
import glob
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from render_lib import *

PW, PH, SH = 3000, 900, 170


def tone(c, floor=50, ceil=99.5):
    nz = c[c > 0]
    c0 = np.percentile(nz, floor)
    sat = np.percentile(nz, ceil)
    t = np.clip(np.log1p(c / c0) / np.log1p(sat / c0), 0, 1)
    occ = (c > 0).mean(0)
    heavy = (c > sat) & (occ < 0.03)[None, :]
    return np.maximum(t, 0.92 * maximum_filter(heavy.astype(float), size=3))


def strip_img(e, lam, lo, hi):
    pos, neg, y0, _, _ = lyap_raster(e, np.clip(lam, -1.5, 1), lo, hi, PW, SH, -1.2, 0.8)
    s = np.zeros((SH, PW, 3))
    s[pos] = (235, 120, 40)
    s[neg] = (80, 80, 105)
    s[int(round(y0))] = (140, 140, 140)
    return s


def panel(title, sub, c, e, lam, lo, hi, ylo, yhi, xlab, ylab, rule=None):
    img = np.zeros((PH + 12 + SH, PW, 3))
    img[:PH] = apply_lut(tone(c), cmap_lut("cc:fire"))
    if lam is not None:
        img[PH + 12:] = strip_img(e, lam, lo, hi)
    if rule is not None:
        for r, col in rule:
            x = int((r - lo) / (hi - lo) * PW)
            if 0 <= x < PW:
                img[:PH:3, x] = col
    pil = to_img(img)
    canvas = Image.new("RGB", (PW + 240, PH + 12 + SH + 230), (6, 6, 8))
    canvas.paste(pil, (160, 130))
    dr = ImageDraw.Draw(canvas)
    dr.text((160, 20), title, fill=(245, 230, 210), font=font(48, "serif"))
    dr.text((160, 80), sub, fill=(170, 150, 140), font=font(28, "serif"))
    f = font(26, "mono")
    for v in np.linspace(lo, hi, 7):
        x = 160 + (v - lo) / (hi - lo) * PW
        dr.line([(x, 130 + PH + 12 + SH + 6), (x, 130 + PH + 12 + SH + 20)], fill=(150, 140, 130), width=2)
        dr.text((x, 130 + PH + 12 + SH + 26), f"{v:.4g}", fill=(170, 150, 140), font=f, anchor="ma")
    dr.text((160 + PW, 130 + PH + 12 + SH + 70), xlab, fill=(170, 150, 140), font=f, anchor="ra")
    for v in np.linspace(ylo, yhi, 5):
        y = 130 + (yhi - v) / (yhi - ylo) * PH
        dr.text((150, y), f"{v:.3g}", fill=(170, 150, 140), font=f, anchor="rm")
    dr.text((20, 100), ylab, fill=(170, 150, 140), font=f)
    return canvas


def main():
    panels = []
    # logistic
    d = np.load(f"{CACHE}/bif_logistic_hi.npz")
    lo, hi = 2.9, 4.0
    m = d["etas"] >= lo
    c = density(d["etas"][m], d["P"][m], lo, hi, 0, 1, PW, PH)
    panels.append(panel("logistic map  x → r x (1 − x)", "reference / null: the textbook unimodal map through the identical pipeline",
                        c, d["etas"], d["lyap"], lo, hi, 0, 1, "r", "x"))
    # prod2
    d = np.load(f"{CACHE}/bif_prod2_hi.npz")
    lo, hi = 0.9, 2.0
    m = d["etas"] >= lo
    c = density(d["etas"][m], d["P"][m], lo, hi, -0.02, 2.02, PW, PH)
    panels.append(panel("GD on ½(x₁x₂ − 1)²", "x₀ = (1.1, 0.9); first doubling at η = 2/s_min = 1 (s_min = 2, balanced minimum)",
                        c, d["etas"], d["lyap_bal"], lo, hi, -0.02, 2.02, "η", "P = x₁x₂", rule=[(1.0, (255, 80, 80))]))
    # prod4
    d = np.load(f"{CACHE}/bif_prod4_hi.npz")
    lo, hi = 0.45, 1.21
    c = density(d["etas"], d["P"], lo, hi, -0.03, 2.2, PW, PH)
    panels.append(panel("GD on ½(x₁x₂x₃x₄ − 1)²  (Zhu et al. minimalist model)", "x₀ = (1.1, 0.9, 1.05, 0.95); first doubling at η = 2/4",
                        c, d["etas"], d["lyap_bal"], lo, hi, -0.03, 2.2, "η", "P", rule=[(0.5, (255, 80, 80))]))
    # DLN
    for init, sub in [("aligned", "Ghosh et al. init W₃ = 0, W₁ = W₂ = 0.1·I; orange: σ₁ mode c₁, grey: σ₂ mode c₂"),
                      ("random", "random Gaussian init (seed 0, std 0.5/√5); gaps = runs that diverged from this init")]:
        d = np.load(f"{CACHE}/bif_dln_{init}.npz")
        e = d["etas"]
        lo, hi = e[0], e[-1]
        c = density(e, d["C1"], lo, hi, -2, 21, PW, PH) + density(e, d["C2"], lo, hi, -2, 21, PW, PH)
        onset = 2 / (3 * 10 ** (4 / 3))
        panels.append(panel(f"deep linear network  ½‖W₃W₂W₁ − M‖²  ({init} init)", sub + f";  onset 2/(Lσ₁^(2−2/L)) = {onset:.5f}",
                            c, e, d["lyap"], lo, hi, -2, 21, "η", "u*ᵀ W₃W₂W₁ v*", rule=[(onset, (255, 80, 80))]))
    Wt = panels[0].size[0]
    Ht = sum(p.size[1] for p in panels) + 200
    sheet = Image.new("RGB", (Wt, Ht), (6, 6, 8))
    dr = ImageDraw.Draw(sheet)
    dr.text((160, 60), "ONE PIPELINE, FOUR SYSTEMS  ·  every visited iterate after burn-in, one dot each; strip: Lyapunov exponent",
            fill=(245, 230, 210), font=font(56, "serif"))
    y = 200
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.size[1]
    sheet.save(f"{GAL}/systems_dark.png", optimize=True)
    print("wrote systems_dark.png", sheet.size)

    # overlays of many inits
    ov = []
    for name, lo, hi, ylo, yhi, rule in [("prod2", 0.85, 2.0, -0.02, 2.05, 1.0), ("prod4", 0.42, 1.21, -0.03, 2.25, 0.5)]:
        d = np.load(f"{CACHE}/bif_{name}_multi.npz")
        e = d["etas"]
        c = np.zeros((PH, PW))
        for j in range(d["P"].shape[0]):
            c += density(e, d["P"][j], lo, hi, ylo, yhi, PW, PH)
        alive = d["alive"].mean(0)
        ov.append(panel(f"{name}: 48 initialisations overlaid", "x₀ ~ exp(0.35·N(0,1)) per coordinate; strip: fraction of inits that stay bounded (orange) ",
                        c, e, None, lo, hi, ylo, yhi, "η", "P", rule=[(rule, (255, 80, 80))]))
        # replace the strip with the alive fraction
        arr = np.asarray(ov[-1]).copy()
        cx = np.clip(((e - lo) / (hi - lo) * PW).astype(int), 0, PW - 1)
        fr = np.zeros(PW)
        np.maximum.at(fr, cx, alive)
        for x in range(PW):
            h = int(fr[x] * (SH - 10))
            arr[130 + PH + 12 + SH - h:130 + PH + 12 + SH, 160 + x] = (235, 120, 40)
        ov[-1] = Image.fromarray(arr)
    files = sorted(glob.glob(f"{CACHE}/bif_dln_random_seed*.npz"))
    c = np.zeros((PH, PW))
    base = np.load(f"{CACHE}/bif_dln_random.npz")
    for f in files:
        d = np.load(f)
        e = d["etas"]
        lo, hi = e[0], e[-1]
        c += density(e, d["C1"], lo, hi, -2, 21, PW, PH)
    ov.append(panel(f"deep linear network: {len(files)} random-init seeds overlaid", "c₁ = u*₁ᵀ W₃W₂W₁ v*₁; 4000 η, 256 iterates each",
                    c, e, None, lo, hi, -2, 21, "η", "c₁", rule=[(2 / (3 * 10 ** (4 / 3)), (255, 80, 80))]))
    Ht = sum(p.size[1] for p in ov) + 200
    sheet = Image.new("RGB", (Wt, Ht), (6, 6, 8))
    dr = ImageDraw.Draw(sheet)
    dr.text((160, 60), "MANY INITIALISATIONS  ·  the attractor does not depend on the init; whether you reach it does",
            fill=(245, 230, 210), font=font(56, "serif"))
    y = 200
    for p in ov:
        sheet.paste(p, (0, y))
        y += p.size[1]
    sheet.save(f"{GAL}/overlay_inits_dark.png", optimize=True)
    print("wrote overlay_inits_dark.png", sheet.size)


if __name__ == "__main__":
    main()
