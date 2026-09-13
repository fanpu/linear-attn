"""Axis-free large Spectral prints of the cyclic-step-size Lyapunov planes, with a thin caption strip.

Measured: Lyapunov exponent of the oscillating mode of GD on 1/2(x1x2x3x4-1)^2 under a cyclic step-size
schedule (cache/lyapplane_<tag>_6144.npz).  Aesthetic (declared): Sohl-Dickstein Spectral split, each side
rank-normalised, the two dark ends meeting at lambda = 0; caption typography.
"""
import sys
import numpy as np
from PIL import ImageDraw
from render_lib import *
from render_lyapplane import spectral


def main():
    for tag in sys.argv[1:] or ["AB_big", "AABAB_zoombig"]:
        d = np.load(f"{CACHE}/lyapplane_{tag}_6144.npz")
        lam = d["lam"][::-1]
        va, vb = d["vals_a"], d["vals_b"]
        img = spectral(lam)
        H, W = lam.shape
        cap = 150
        canvas = np.full((H + cap, W, 3), 12.0)
        canvas[:H] = img
        pil = to_img(canvas)
        dr = ImageDraw.Draw(pil)
        pat = str(d["pattern"])
        dr.text((60, H + 28), f"Gradient descent on ½(x₁x₂x₃x₄ − 1)² with a cyclic step size ({' '.join(pat)} …).   "
                f"η_A → {va[0]:.4f} … {va[-1]:.4f},  η_B ↑ {vb[0]:.4f} … {vb[-1]:.4f}.",
                fill=(235, 225, 210), font=font(46, "serif"))
        dr.text((60, H + 88), "Colour: Lyapunov exponent of the oscillating mode, Spectral split (declared): periodic λ<0 purple→yellow, "
                "chaotic λ>0 red→yellow, each side rank-normalised; dark seams are λ = 0.  6144² runs, float64.",
                fill=(170, 160, 150), font=font(34, "serif"))
        out = f"{GAL}/print_lyapplane_{tag}_spectral.png"
        pil.save(out, optimize=True)
        print("wrote", out, pil.size)


if __name__ == "__main__":
    main()
