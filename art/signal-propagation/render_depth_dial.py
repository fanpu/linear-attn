"""Extra idea 2 renders: depth as the dial. The same fixed random erf network (N = 100) read out at
depth l = 1 ... 1000; finite width (left) beside infinite width (right, mean-field closed form).

  python render_depth_dial.py --styles spectral,magma,ink
"""
import argparse, os, shutil
import numpy as np
from render_common import *
from sp_core import boundary_mask
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--styles", default="spectral,magma,ink")
ap.add_argument("--tau", type=float, default=1e-5)
args = ap.parse_args()
Z = np.load(os.path.join(CACHE, "depthdial_N100_r512.npz"))
Ls, Lmf, lay, N = Z["L"], Z["L_mf"], Z["layers"], int(Z["N"])
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FLOOR = 1e-12  # declared display floor (float32 floor ~1e-13; mean-field values below ~1e-16 are roundoff)


def rgb(L, style):
    L = np.maximum(L.astype(np.float64), FLOOR)
    ch = L > args.tau
    mag = np.abs(np.log10(L / args.tau))
    if style == "spectral":
        return split_rgb(ch, mag, "sd_spectral", near_boundary="small")
    if style == "magma":
        return plt.get_cmap("magma")(np.clip((np.log10(L) + 16) / 16.5, 0, 1))[..., :3]
    if style == "ink":
        return P.overprint([np.maximum(ch * 0.12, boundary_mask(ch) * 0.95)], [INK])


BG = {"spectral": (17, 16, 20), "magma": (7, 6, 10), "ink": (244, 239, 227)}
FG = {"spectral": (233, 228, 218), "magma": (233, 228, 218), "ink": (29, 27, 25)}
for style in args.styles.split(","):
    fdir = os.path.join(CACHE, f"frames_depth_{style}")
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    seq = [0] * 20 + [i for i in range(len(lay)) for _ in range(4)] + [len(lay) - 1] * 60
    fb = ImageFont.truetype(FONT, 30); fs = ImageFont.truetype(FONT, 20); fm = ImageFont.truetype(MONO, 34)
    cache = {}
    for fi, i in enumerate(seq):
        if i not in cache:
            cache = {i: (rgb(Ls[i], style), rgb(Lmf[i], style))}
        a, b = cache[i]
        img = Image.new("RGB", (1920, 1080), BG[style])
        for j, t in enumerate((a, b)):
            img.paste(Image.fromarray(to_uint8(t[::-1])).resize((864, 864), Image.NEAREST), (48 + j * 912, 150))
        d = ImageDraw.Draw(img)
        d.text((48, 40), "Depth as the dial", font=fb, fill=FG[style])
        d.text((48 + 912, 40), f"depth l = {lay[i]:4d}", font=fm, fill=FG[style])
        d.text((48, 1030), f"width N = {N}: one fixed random erf network (512 x 512, float32)", font=fs, fill=FG[style])
        d.text((48 + 912, 1030), "N = infinity: mean-field closed form, same grid and threshold", font=fs, fill=FG[style])
        d.text((48, 100), "sigma_w, sigma_b in [0, 4]; colour: output distance L of two independent inputs at depth l, frontier L = 1e-5"
               + {"spectral": " (declared Spectral split)", "magma": " (log10 L, magma)", "ink": " (frontier cells in ink)"}[style],
               font=ImageFont.truetype(FONT, 17), fill=FG[style])
        img.save(os.path.join(fdir, f"f{fi:05d}.png"))
    ffmpeg_mp4(fdir, os.path.join(GAL, f"depth_dial_{style}.mp4"), fps=24)
    ffmpeg_gif(fdir, os.path.join(GAL, f"depth_dial_{style}.gif"), fps=12, width=960, step=2)
    print("wrote", style)

# still: small multiples, finite (top) vs infinite (bottom)
pick = [np.argmin(np.abs(lay - l)) for l in (3, 10, 30, 100, 300, 1000)]
W, H = 3000, 1350
fig = fig_px(W, H, bg="#111014")
fg = "#e9e4da"
fig.text(0.03, 0.95, f"Depth as the dial: the frontier sharpens with depth at infinite width, and frays at width {N}", fontsize=22, color=fg, va="center")
for r, S in enumerate((Ls, Lmf)):
    for c, i in enumerate(pick):
        ax = fig.add_axes([0.03 + c * 0.16, 0.50 - r * 0.43, 0.15, 0.37])
        ax.imshow(rgb(S[i], "spectral"), origin="lower", interpolation="nearest"); ax.set_axis_off()
        ax.set_title(f"{'N = ' + str(N) if r == 0 else 'N = inf'},  l = {lay[i]}", color=fg, fontsize=12)
fig.text(0.03, 0.03, "sigma_w (horizontal) and sigma_b (vertical) in [0, 4]; declared Spectral split of the output distance L at depth l around L = 1e-5, rank-normalised per tile.",
         fontsize=12, color=fg)
savefig(fig, "depth_dial_small_multiples_spectral.png")
