"""'Width as time': the full-plane frontier as the network gets wider (8 ... 1024), with width-nested
common random numbers (every width-N network is the top-left N x N block of the same master draws).
Also a seeds x widths small-multiples plate.

  python render_width.py --styles spectral,magma,ink
"""
import argparse, glob, os, re, shutil
import numpy as np
from render_common import *
from sp_core import boundary_mask, edge_cells, box_counts
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--styles", default="spectral,magma,ink")
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--res", type=int, default=512)
args = ap.parse_args()
D = 1000
files = {}
for f in glob.glob(os.path.join(CACHE, f"widthmap_widths_N*_s*_D{D}_f32_r{args.res}.npz")):
    m = re.search(r"_N(\d+)_s(\d+)_", f)
    files[(int(m.group(1)), int(m.group(2)))] = f
Ns = sorted(n for n, s in files if s == 0)
MF = np.load(os.path.join(CACHE, f"widthmap_mf_D{D}_r{args.res}.npz"))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def fields(Z):
    L = Z["L_avg"]; ch = L > args.tau
    if "t_hit" in Z:
        mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / args.tau), (D + 1) - Z["t_hit"].astype(float))
    else:
        mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / args.tau), np.log10(args.tau / np.maximum(L, 1e-300)))
    return L, ch, mag


def rgb_of(Z, style):
    L, ch, mag = fields(Z)
    if style == "spectral":
        return split_rgb(ch, mag, "sd_spectral", near_boundary="small")
    if style == "magma":
        return plt.get_cmap("magma")(np.clip((np.log10(np.maximum(L, 1e-30)) + 16) / 16.5, 0, 1))[..., :3]
    if style == "ink":
        e = boundary_mask(ch)
        return P.overprint([np.maximum(ch * 0.12, e * 0.95)], [INK])
    raise ValueError(style)


def frac_mismatch(Z):
    return float(((Z["L_avg"] > args.tau) != (MF["L_avg"] > args.tau)).mean())


BG = {"spectral": (17, 16, 20), "magma": (7, 6, 10), "ink": (244, 239, 227)}
FG = {"spectral": (233, 228, 218), "magma": (233, 228, 218), "ink": (29, 27, 25)}
stats = {}
for n in Ns:
    Z = np.load(files[(n, 0)])
    stats[n] = dict(mismatch=frac_mismatch(Z), edges=int(edge_cells(Z["L_avg"] > args.tau).sum()))
stats["mf"] = dict(mismatch=0.0, edges=int(edge_cells(MF["L_avg"] > args.tau).sum()))
print({k: v for k, v in stats.items()})
np.save(os.path.join(CACHE, "width_stats.npy"), stats, allow_pickle=True)

for style in args.styles.split(","):
    fdir = os.path.join(CACHE, f"frames_width_{style}")
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    tiles = [(f"N = {n}", rgb_of(np.load(files[(n, 0)]), style), stats[n]) for n in Ns]
    tiles.append(("N = infinity (mean field)", rgb_of(MF, style), stats["mf"]))
    fi = 0
    ft = ImageFont.truetype(FONT, 30); fs = ImageFont.truetype(FONT, 19); fm = ImageFont.truetype(MONO, 40)
    for i, (lab, rgb, st) in enumerate(tiles):
        nxt = tiles[min(i + 1, len(tiles) - 1)][1]
        hold, fade = (30, 12) if i < len(tiles) - 1 else (72, 0)
        for t in range(hold + fade):
            a = 0 if t < hold else (t - hold + 1) / (fade + 1)
            mix = (1 - a) * rgb + a * nxt   # declared cross-dissolve between measured frames
            img = Image.new("RGB", (1920, 1080), BG[style])
            tile = Image.fromarray(to_uint8(mix[::-1])).resize((1024, 1024), Image.NEAREST)
            img.paste(tile, (28, 28))
            d = ImageDraw.Draw(img)
            x = 1100
            d.text((x, 60), "Finite Width", font=ft, fill=FG[style])
            d.text((x, 130), lab, font=fm, fill=FG[style])
            info = [f"random erf MLP, depth {D}, one fixed draw", "(width-N net = top-left N x N block of",
                    "the same 1024 x 1024 master weights)", "",
                    "sigma_w (horizontal) and sigma_b (vertical)", "both from 0 to 4; 512 x 512 pixels, float32", "",
                    f"frontier cells at this width: {st['edges']}",
                    f"pixels disagreeing with mean field: {100*st['mismatch']:.2f}%", "",
                    {"spectral": "Spectral split (declared): purple = ordered,", "magma": "log10 output distance L (magma, -16 to 0.5)",
                     "ink": "single ink: frontier cells; chaotic side tinted"}[style],
                    {"spectral": "red = chaotic, shade = rank of closeness", "magma": "black: the two inputs merged", "ink": ""}[style],
                    {"spectral": "to the frontier.", "magma": "", "ink": ""}[style], "",
                    "frames are separate measurements;", "dissolves between them are declared."]
            for j, s in enumerate(info):
                d.text((x, 230 + j * 32), s, font=fs, fill=FG[style])
            img.save(os.path.join(fdir, f"f{fi:05d}.png")); fi += 1
    ffmpeg_mp4(fdir, os.path.join(GAL, f"width_as_time_{style}.mp4"), fps=24)
    ffmpeg_gif(fdir, os.path.join(GAL, f"width_as_time_{style}.gif"), fps=12, width=960, step=2)
    print("wrote", style, fi)

# seeds x widths plate
seed_ns = sorted({n for n, s in files if s > 0})
if seed_ns:
    cols = len(seed_ns) + 1
    Wf, Hf = 3000, 2600
    fig = fig_px(Wf, Hf, bg="#111014")
    fg = "#e9e4da"
    fig.text(0.04, 0.975, "Finite Width: three seeds x three widths (Spectral split), and the infinite-width limit", fontsize=24, color=fg, va="top")
    for r, s in enumerate((0, 1, 2)):
        for c, n in enumerate(seed_ns):
            ax = fig.add_axes([0.04 + c * 0.235, 0.70 - r * 0.29, 0.22, 0.25])
            ax.imshow(rgb_of(np.load(files[(n, s)]), "spectral"), origin="lower", extent=[0, 4, 0, 4], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"width N = {n}, seed {s}", color=fg, fontsize=13)
    ax = fig.add_axes([0.04 + 3 * 0.235, 0.41, 0.22, 0.25])
    ax.imshow(rgb_of(MF, "spectral"), origin="lower", extent=[0, 4, 0, 4], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("N = infinity (mean field)", color=fg, fontsize=13)
    fig.text(0.04, 0.05, f"Each tile: sigma_w (horizontal) and sigma_b (vertical) in [0, 4], erf MLP depth {D}, L = |x1 - x2|^2 averaged over the last 20 layers, "
             f"frontier at L = {args.tau:g}; {args.res} x {args.res}, float32.\nSeeds change the frontier's fine texture; width changes its roughness "
             "and location. Declared Spectral split colouring, rank-normalised per tile.", fontsize=12, color=fg, va="top", linespacing=1.6)
    savefig(fig, "width_seeds_grid_spectral.png")
