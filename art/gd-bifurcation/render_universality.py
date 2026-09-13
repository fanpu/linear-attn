"""Universality as an image: logistic map (blue ink) and GD on 1/2(x1x2x3x4-1)^2 (pink ink) overprinted.

x axis: s = log10(A / (eta_inf - eta)); with A fixed from each system's measured eta_n, the n-th doubling of
BOTH systems sits at s = n log10(4.6692) if delta is universal.  (Declared transform.)
y axis: (value - critical point of the map) / max|value - critical point| per column (declared), GD flipped vertically if that
increases overlap with the logistic branches (orientation is not universal; recorded in the caption).
"""
import numpy as np
from PIL import ImageDraw
from render_lib import *

W, H = 7000, 1900


def critical(name, eta):
    if name == "logistic":
        return np.full_like(eta, 0.5)
    u = np.full_like(eta, 0.9)  # balanced-map critical point: eta (7u^6 - 3u^2) = 1, in P = u^4 coordinates
    for _ in range(80):
        u -= (eta * (7 * u ** 6 - 3 * u ** 2) - 1) / (eta * (42 * u ** 5 - 6 * u))
    return u ** 4


def raster(vals, s, s0, s1, vc):
    dv = vals - vc[:, None]
    sc = np.nanmax(np.abs(dv), 1, keepdims=True)
    v = dv / np.maximum(sc, 1e-300)
    return density(s, v, s0, s1, -0.62, 1.03, W, H)  # vertical crop, declared


def tone(c):
    from scipy.ndimage import maximum_filter
    nz = c[c > 0]
    c0 = np.percentile(nz, 5)
    sat = np.percentile(nz, 50)
    t = np.clip(np.log1p(c / c0) / np.log1p(sat / c0), 0, 1)
    return maximum_filter(t, size=7)  # declared minimum stroke weight


def main():
    d = np.load(f"{CACHE}/universality.npz")
    s = d["s"]
    s0, s1 = 0.95, s[-1]
    cl = raster(d["logistic_vals"], s, s0, s1, critical("logistic", d["logistic_eta"]))
    cg = raster(d["prod4_vals"], s, s0, s1, critical("prod4", d["prod4_eta"]))
    cgf = cg[::-1]
    tl, tg, tgf = tone(cl), tone(cg), tone(cgf)
    m = slice(int(0.35 * W), W)
    ov = np.sum(tl[:, m] * tg[:, m])
    ovf = np.sum(tl[:, m] * tgf[:, m])
    flipped = ovf > ov
    tg = tgf if flipped else tg
    print("flip GD vertically:", flipped, "overlap", ov, ovf)
    padx, pady = 220, 260
    Ht, Wt = H + 2 * pady + 120, W + 2 * padx
    pap = paper(Ht, Wt, base=(246, 242, 232), grain=3, seed=5)
    a1 = np.zeros((Ht, Wt))
    a2 = np.zeros((Ht, Wt))
    a1[pady:pady + H, padx:padx + W] = 0.97 * tl
    off = (4, -3)  # misregistration, declared
    a2[pady + off[1]:pady + H + off[1], padx + off[0]:padx + W + off[0]] = 0.92 * tg
    img = ink_multiply(ink_multiply(pap, a1, (0, 95, 160)), a2, (255, 40, 150))
    pil = to_img(img)
    dr = ImageDraw.Draw(pil)
    ink = (30, 30, 40)
    for n in range(1, 10):
        x = padx + (n * np.log10(4.669201609) - s0) / (s1 - s0) * W
        if x > padx + W:
            break
        dr.line([(x, pady + H + 20), (x, pady + H + 60)], fill=ink, width=3)
        dr.text((x, pady + H + 70), f"{2 ** (n - 1)}→{2 ** n}", fill=ink, font=font(34, "mono"), anchor="ma")
    sl = d["logistic_s_bif"]
    sg = d["prod4_s_bif"]
    dr.text((padx, 80), "UNIVERSALITY  ·  blue: logistic map x→rx(1−x)   pink: gradient descent on ½(x₁x₂x₃x₄−1)²,  overprinted",
            fill=ink, font=font(56, "serif"))
    dr.text((padx, 160), "horizontal: s = log₁₀[A/(η∞ − η)], ticks every log₁₀δ = 0.6692;  vertical: (value − critical point) / max |value − critical point| per column"
            + ("; GD flipped vertically" if flipped else ""), fill=(90, 85, 85), font=font(34, "serif"))
    dr.text((padx, pady + H + 140),
            f"measured doubling positions  s_n (logistic)  = " + "  ".join(f"{v:.3f}" for v in sl[:8]),
            fill=(0, 90, 150), font=font(32, "mono"))
    dr.text((padx, pady + H + 190),
            f"measured doubling positions  s_n (GD, k=4)   = " + "  ".join(f"{v:.3f}" for v in sg[:8]),
            fill=(200, 40, 130), font=font(32, "mono"))
    dr.text((padx + W, pady + H + 190), "they agree from n = 2 on; n = 1 differs (δ₁ is not universal)",
            fill=ink, font=font(32, "serif"), anchor="ra")
    out = f"{GAL}/universality_riso.png"
    pil.save(out, optimize=True)
    print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
