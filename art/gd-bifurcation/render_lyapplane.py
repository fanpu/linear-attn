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


def main():
    for pattern in sys.argv[1:] or ["AABAB", "AB"]:
        import glob
        fs = sorted(glob.glob(f"{CACHE}/lyapplane_{pattern}_[0-9]*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
        d = np.load(fs[-1])
        tag = pattern
        lam = d["lam"][::-1]  # row 0 = top = largest B
        va = d["vals_a"] if "vals_a" in d else d["vals"]
        vb = d["vals_b"] if "vals_b" in d else d["vals"]
        for style, fn in [("dark", dark), ("paper", paper_ink)]:
            img = fn(lam)
            H, W = lam.shape
            pad = int(0.06 * W)
            bg = (6, 6, 8) if style == "dark" else (243, 238, 226)
            fg = (230, 215, 190) if style == "dark" else (24, 26, 46)
            canvas = np.full((H + 2 * pad + pad // 2, W + 2 * pad, 3), bg, float)
            if style == "paper":
                canvas = paper(*canvas.shape[:2], base=bg, grain=3, seed=9)
            canvas[pad:pad + H, pad:pad + W] = img
            pil = to_img(canvas)
            dr = ImageDraw.Draw(pil)
            fs1 = font(int(W / 55), "serif")
            fs2 = font(int(W / 80), "mono")
            dr.text((pad, pad // 3), f"cyclic step size  {pattern.split('_')[0]}  ·  GD on ½(x₁x₂x₃x₄−1)²  ·  Lyapunov exponent of the oscillating mode",
                    fill=fg, font=fs1)
            dr.text((pad, pad + H + pad // 5), f"η_A → {va[0]:.4f} … {va[-1]:.4f}      ↑ η_B {vb[0]:.4f} … {vb[-1]:.4f}      "
                    + ("gold λ<0 · black λ=0 · blue λ>0" if style == "dark" else "ink ∝ tanh|λ| for λ<0 · light stipple λ>0") + "      diagonal η_A = η_B is the constant-step cascade",
                    fill=fg, font=fs2)
            out = f"{GAL}/lyapplane_{tag}_{style}.png"
            pil.save(out, optimize=True)
            print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
