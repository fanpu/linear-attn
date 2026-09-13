"""Render the two-step-size Lyapunov planes (cache/lyapplane_<pattern>_<res>.npz).

Measured: Lyapunov exponent of the oscillating mode for GD with a cyclic step-size pattern.
Aesthetic (declared): lambda < 0 (periodic/converging) in a gold ramp that darkens toward superstable
curves (lambda -> -inf); lambda > 0 (chaos) in a blue ramp; lambda = 0 is black.  The classic
Markus-Lyapunov colouring convention.  Paper variant: single ink, density = tanh(|lambda|) for lambda<0,
chaos left as bare paper with a light stipple.
"""
import sys
import numpy as np
from PIL import ImageDraw
from render_lib import *


def dark(lam):
    neg = np.clip(-lam, 0, None)
    pos = np.clip(lam, 0, None)
    gold = np.array([255, 196, 70.0])
    blue = np.array([70, 150, 255.0])
    tn = np.tanh(neg / 0.6) ** 0.7
    tp = np.tanh(pos / 0.9) ** 0.9
    img = tn[..., None] * gold + tp[..., None] * blue
    img = np.where(np.isfinite(lam)[..., None], img, 0)
    return img


def paper_ink(lam, seed=0):
    H, W = lam.shape
    pap = paper(H, W, base=(243, 238, 226), grain=3, seed=seed)
    neg = np.nan_to_num(np.clip(-lam, 0, None))
    a = np.tanh(neg / 0.5) ** 0.8
    rng = np.random.default_rng(seed)
    chaos = np.nan_to_num(lam) > 0
    a = np.where(chaos, 0.10 * (rng.random((H, W)) < 0.5), a)
    return ink_multiply(pap, 0.95 * a, (24, 26, 46))


def spectral(lam):
    """Declared: Sohl-Dickstein Spectral split; lambda<0 ranked by |lambda| (purple at lambda=0 -> pale yellow),
    lambda>0 ranked by lambda (deep red at lambda=0 -> pale yellow)."""
    fin = np.isfinite(lam)
    return spectral_split(-np.nan_to_num(lam), np.nan_to_num(lam), fin & (lam < 0), fin & (lam >= 0))


def main():
    for pattern in sys.argv[1:] or ["AABAB", "AB"]:
        import glob
        fs = sorted(glob.glob(f"{CACHE}/lyapplane_{pattern}_[0-9]*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
        d = np.load(fs[-1])
        tag = pattern
        lam = d["lam"][::-1]  # row 0 = top = largest B
        va = d["vals_a"] if "vals_a" in d else d["vals"]
        vb = d["vals_b"] if "vals_b" in d else d["vals"]
        for style, fn in [("dark", dark), ("paper", paper_ink), ("spectral", spectral)]:
            img = fn(lam)
            H, W = lam.shape
            pad = int(0.06 * W)
            bg = (243, 238, 226) if style == "paper" else (6, 6, 8)
            fg = (24, 26, 46) if style == "paper" else (230, 215, 190)
            canvas = np.full((H + 2 * pad + pad // 2, W + 2 * pad, 3), bg, float)
            if style == "paper":
                canvas = paper(*canvas.shape[:2], base=bg, grain=3, seed=9)
            canvas[pad:pad + H, pad:pad + W] = img
            # faint hairline on the diagonal eta_A = eta_B where it crosses the frame
            for col in range(W):
                a_val = va[col]
                rowf = (vb[-1] - a_val) / (vb[-1] - vb[0]) * (H - 1)
                if 0 <= rowf < H:
                    r_ = int(rowf)
                    canvas[pad + r_, pad + col] = 0.55 * canvas[pad + r_, pad + col] + 0.45 * np.array(fg, float)
            pil = to_img(canvas)
            dr = ImageDraw.Draw(pil)
            fs1 = font(int(W / 72), "serif")
            fs2 = font(int(W / 105), "mono")
            dr.text((pad, pad // 3), f"EXTRA SYSTEM: cyclic step-size schedule  {pattern.split('_')[0]}  ·  GD on ½(x₁x₂x₃x₄−1)²  ·  Lyapunov exponent of the oscillating mode",
                    fill=fg, font=fs1)
            dr.text((pad, pad + H + pad // 8), f"η_A → {va[0]:.5f} … {va[-1]:.5f}      ↑ η_B {vb[0]:.5f} … {vb[-1]:.5f}      zoom ×{0.55 / (va[-1] - va[0]):,.0f}",
                    fill=fg, font=fs2)
            dr.text((pad, pad + H + pad // 8 + int(W / 60)), {"dark": "gold λ<0 · black λ=0 · blue λ>0", "paper": "ink ∝ tanh|λ| for λ<0 · light stipple λ>0",
                       "spectral": "Spectral split (declared): λ<0 purple→yellow by rank of |λ|, λ>0 red→yellow by rank"}[style] + "   ·   hairline: η_A = η_B (constant step)",
                    fill=fg, font=fs2)
            out = f"{GAL}/lyapplane_{tag}_{style}.png"
            pil.save(out, optimize=True)
            print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
