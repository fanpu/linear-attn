"""Animation: correlation vs depth over the (sigma_w^2, sigma_b^2) plane (the flow toward the fixed
point), from the measured empirical tanh sweep. Top panel: two independent inputs (c0 ~ 0) are
pulled together; bottom panel: two nearly identical inputs (c0 = 0.99) are pulled apart in the
chaotic phase. Also a small-multiples still.

  python render_flow.py --styles oslo,spectral,cyanotype
"""
import argparse, os, shutil
import numpy as np
from render_common import *
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--emp", default="cache/empirical_tanh.npz")
ap.add_argument("--styles", default="oslo,spectral,cyanotype")
ap.add_argument("--max_frames", type=int, default=100000)
args = ap.parse_args()
E = np.load(os.path.join(HERE, args.emp))
cAB = E["cAB"].astype(np.float32)
cAC = E["cAC"].astype(np.float32)
Dp = cAB.shape[0] - 1
N, K = int(E["N"]), int(E["K"])
ch = E["chi1"] > 1
nan = ~np.isfinite(E["chi1"])
ch[nan] = E["cstar"][nan] < 1 - 1e-3
H, W = cAB.shape[1:]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def colour(c, style):
    c = np.clip(c, -0.05, 1.0)
    if style == "oslo":
        return plt.get_cmap("cmc.oslo")(np.clip(c, 0, 1))[..., :3]
    if style == "cyanotype":
        return P.get("cyanotype")(np.clip(c, 0, 1))[..., :3]
    if style == "spectral":
        # declared: split by the measured phase; within a side the shade is the correlation itself
        u = 0.75 * np.clip(c, 0, 1)
        a = P.as_cmap(P.PAIRINGS["sd_spectral"]["neg"]); b = P.as_cmap(P.PAIRINGS["sd_spectral"]["pos"])
        return np.where(ch[..., None], b(u)[..., :3], a(u)[..., :3])
    raise ValueError(style)


BG = {"oslo": (8, 9, 12), "cyanotype": (244, 239, 227), "spectral": (17, 16, 20)}
FG = {"oslo": (232, 228, 218), "cyanotype": (29, 27, 25), "spectral": (233, 228, 218)}
NOTE = {"oslo": "colour: correlation c (Crameri oslo, 0 dark to 1 light)",
        "cyanotype": "colour: correlation c (cyanotype ramp, 0 paper to 1 deep blue)",
        "spectral": "declared split: purple half where measured chi_1 < 1, red half where chi_1 > 1; shade = correlation c"}
layers = list(range(0, 40)) + list(range(40, 120, 2)) + list(range(120, Dp + 1, 5))
hold = 36
seq = [0] * 24 + layers + [Dp] * hold
seq = seq[: args.max_frames]
scale = 4 if W * 4 <= 1920 else 2
pw, ph = W * 2, H * 2   # 960 x 480 panels (nearest-neighbour 2x; declared)


def frame(style, l):
    img = Image.new("RGB", (1920, 1080), BG[style])
    d = ImageDraw.Draw(img)
    ft = ImageFont.truetype(FONT, 34); fs = ImageFont.truetype(FONT, 20); fm = ImageFont.truetype(MONO, 22)
    for i, (c, lab) in enumerate(((cAB, "two unrelated inputs (c0 = 0): do they merge?"),
                                  (cAC, "two nearly identical inputs (c0 = 0.99): do they separate?"))):
        rgb = to_uint8(colour(c[l][::-1], style))
        tile = Image.fromarray(rgb).resize((pw, ph), Image.NEAREST)
        y0 = 60 + i * (ph + 40)
        img.paste(tile, (60, y0))
        d.text((60 + pw + 30, y0 + 10), lab, font=fs, fill=FG[style])
    x = 60 + pw + 30
    d.text((x, 150), "Order and Chaos", font=ft, fill=FG[style])
    d.text((x, 205), f"depth  l = {l:4d}", font=fm, fill=FG[style])
    lines = [f"Measured correlation c^l between two inputs", f"after l layers of random tanh MLPs,",
             f"width N = {N}, {K} networks per pixel.", "",
             "horizontal: sigma_w^2 from 0.5 to 4.5", "vertical: sigma_b^2 from 0 to 2", "",
             NOTE[style][:46], NOTE[style][46:92], NOTE[style][92:138], "",
             "Left of the critical line every pair is",
             "driven to c = 1 (order); right of it pairs",
             "settle at c* < 1 (chaos). The line itself",
             "is where the flow is slowest."]
    for j, t in enumerate(lines):
        d.text((x, 300 + j * 30), t, font=fs, fill=FG[style])
    return img


for style in args.styles.split(","):
    fdir = os.path.join(CACHE, f"frames_flow_{style}")
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    for i, l in enumerate(seq):
        frame(style, l).save(os.path.join(fdir, f"f{i:05d}.png"))
    ffmpeg_mp4(fdir, os.path.join(GAL, f"flow_correlation_{style}.mp4"), fps=24)
    ffmpeg_gif(fdir, os.path.join(GAL, f"flow_correlation_{style}.gif"), fps=12, width=960, step=2)
    print("wrote", style, len(seq))

# small multiples still (paper)
ls_ = [0, 1, 2, 4, 8, 16, 32, 64, 128, Dp]
Wf, Hf = 3000, 1500
fig = fig_px(Wf, Hf, bg=PAPER)
fig.text(0.04, 0.95, f"Correlation flow, measured (tanh, N = {N}, {K} nets per pixel): unrelated inputs (top) and near-identical inputs (bottom)",
         fontsize=20, color=INK, va="center")
for r, c in enumerate((cAB, cAC)):
    for j, l in enumerate(ls_):
        ax = fig.add_axes([0.04 + j * 0.095, 0.52 - r * 0.42, 0.09, 0.36])
        ax.imshow(np.clip(c[l], 0, 1), origin="lower", cmap="cmc.oslo", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"l = {l}", fontsize=12, color=INK)
fig.text(0.04, 0.05, "Each tile: the whole plane sigma_w^2 in [0.5, 4.5] (horizontal) x sigma_b^2 in [0, 2] (vertical); colour = correlation (oslo, 0 dark to 1 light).",
         fontsize=12, color=INK)
savefig(fig, "flow_correlation_small_multiples.png")
