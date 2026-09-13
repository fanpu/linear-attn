"""Continuous deep-zoom video from native-resolution keyframes (compute_zoom_chain.py, step sqrt(2) or 2).

Frame at continuous zoom t in [k, k+1]: the view rectangle's edges move from window k to window k+1
with an exponential side s(t) = s_k 2^{-(t-k) log2(step)} (edges are convex combinations, so the view
always lies inside keyframe k). Each output pixel is sampled from the deepest keyframe whose window
contains it (area-averaged when that keyframe is finer than the output, nearest otherwise).
With keyframe resolution R and step r, keyframe k never has fewer than R/r pixels across the view,
so R >= 1080 r means no upsampling anywhere. This compositing is a declared rendering choice; every
sampled value is a measurement.

  python render_zoom_video.py --chain cache/zoom_V_N100_D1000_s0_f32_r1536.npz --styles spectral,magma,ink
"""
import argparse, os, shutil
import numpy as np
from render_common import *
from sp_core import boundary_mask
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--chain", required=True)
ap.add_argument("--styles", default="spectral,magma,ink")
ap.add_argument("--tau", type=float, default=1e-5)
ap.add_argument("--out", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--sec_per_step", type=float, default=2.0)
ap.add_argument("--max_frames", type=int, default=100000)
ap.add_argument("--name", default="deepzoom")
ap.add_argument("--label", default=None)
ap.add_argument("--ss", type=int, default=2)
args = ap.parse_args()
Z = np.load(os.path.join(HERE, args.chain))
wins = Z["windows"].astype(np.float64)
nk, R = Z["L_avg"].shape[0], Z["L_avg"].shape[1]
N, D = int(Z["N"]), int(Z["D"])
step = (wins[0][1] - wins[0][0]) / (wins[1][1] - wins[1][0])
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


LABEL = args.label or (str(Z["label"]) if "label" in Z else "tau")


def keyframe_fields(k):
    """(chaotic bool, magnitude >= 0 with small = close to the frontier, log10 L) at native resolution."""
    L = Z["L_avg"][k]; t = Z["t_hit"][k].astype(float)
    if LABEL == "sync":
        ch = t > D; floor = 1e-10
    else:
        ch = L > args.tau; floor = args.tau
    mag = np.where(ch, np.log10(np.maximum(L, 1e-300) / floor), (D + 1) - t)
    e = boundary_mask(ch)
    return np.stack([ch.astype(float), mag, np.log10(np.maximum(L, 1e-30)), e], -1)


def fields_to_rgb(F, style):
    ch, mag, lL, e = F[..., 0] > 0.5, F[..., 1], F[..., 2], F[..., 3]
    if style in ("spectral", "aurora"):
        # per-frame, per-side rank normalisation (Sohl-Dickstein-style CDF) of the composited field: no seams
        # at keyframe borders; ranks from a fixed random subsample of the frame (declared)
        x = np.where(ch, 1.0, -1.0) * np.maximum(mag, 1e-300)
        ref = x.ravel()[RS]
        return split_rgb(ch, mag, "sd_spectral" if style == "spectral" else "aurora_ember", near_boundary="small", ref=ref)
    if style == "magma":
        return plt.get_cmap("magma")(np.clip((lL + 16) / 16.5, 0, 1))[..., :3]
    if style == "ink":
        return P.overprint([np.maximum(ch * 0.12, np.clip(e, 0, 1) * 0.95)], [INK])
    raise ValueError(style)


def sample(img, win, X, Y):
    """img (R,R,3) origin lower; X,Y coordinate arrays -> rgb, nearest sampling with 2x2 supersampling done by caller."""
    x0, x1, y0, y1 = win
    ix = np.clip(((X - x0) / (x1 - x0) * R).astype(np.int64), 0, R - 1)
    iy = np.clip(((Y - y0) / (y1 - y0) * R).astype(np.int64), 0, R - 1)
    return img[iy, ix]


BG = {"spectral": (17, 16, 20), "aurora": (7, 8, 11), "magma": (7, 6, 10), "ink": (244, 239, 227)}
FG = {"spectral": (233, 228, 218), "aurora": (233, 228, 218), "magma": (233, 228, 218), "ink": (29, 27, 25)}
ss = args.ss  # supersampling per axis (area average of nearest samples)
RS = np.random.default_rng(0).choice((args.out * ss) ** 2, 400000, replace=False)
O = args.out
for style in args.styles.split(","):
    kimgs = [keyframe_fields(k) for k in range(nk)]
    fdir = os.path.join(CACHE, f"frames_{args.name}_{style}")
    shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
    nsteps = nk - 1
    fps_step = int(round(args.sec_per_step * args.fps))
    total = nsteps * fps_step + 1
    frames = list(range(total))[: args.max_frames]
    hold_end = int(2.5 * args.fps)
    u = (np.arange(O * ss) + 0.5) / (O * ss)
    for fi in frames + [frames[-1]] * hold_end:
        t = fi / fps_step
        k = min(int(np.floor(t)), nsteps - 1) if nsteps > 0 else 0
        a = t - k
        g = (1 - step ** (-a)) / (1 - 1 / step)
        view = wins[k] + (wins[k + 1] - wins[k]) * g if nsteps > 0 else wins[0]
        X = view[0] + u[None, :] * (view[1] - view[0])
        Y = view[2] + u[:, None] * (view[3] - view[2])
        X = np.broadcast_to(X, (O * ss, O * ss)); Y = np.broadcast_to(Y, (O * ss, O * ss))
        out = sample(kimgs[k], wins[k], X, Y).copy()
        for j in range(k + 1, nk):
            w = wins[j]
            inside = (X >= w[0]) & (X < w[1]) & (Y >= w[2]) & (Y < w[3])
            if not inside.any():
                break
            out[inside] = sample(kimgs[j], w, X[inside], Y[inside])
        out = fields_to_rgb(out, style).reshape(O, ss, O, ss, 3).mean((1, 3))
        frame = Image.new("RGB", (O, O), BG[style])
        frame.paste(Image.fromarray(to_uint8(out[::-1])), (0, 0))
        d = ImageDraw.Draw(frame)
        side = view[1] - view[0]
        if fi >= frames[-1]:
            # caption only at the end (fractals doc: no captions until the end)
            d.rectangle([0, O - 150, O, O], fill=BG[style])
            fs = ImageFont.truetype(FONT, 22); fm = ImageFont.truetype(MONO, 18)
            d.text((30, O - 138), f"Finite Width: random erf network N = {N}, depth {D}, zoom x{4 / side:,.0f}", font=fs, fill=FG[style])
            d.text((30, O - 100), f"centre sigma_w = {0.5*(view[0]+view[1]):.10f}  sigma_b = {0.5*(view[2]+view[3]):.10f}", font=fm, fill=FG[style])
            d.text((30, O - 70), f"{nk} native {R}x{R} keyframes ({str(Z['dtype'])}), label: {LABEL}; composited, <=1.7x magnified between keyframes", font=fm, fill=FG[style])
        else:
            fm = ImageFont.truetype(MONO, 16)
            d.text((16, O - 28), f"x{4 / side:,.1f}", font=fm, fill=FG[style])
        frame.save(os.path.join(fdir, f"f{len(os.listdir(fdir)):05d}.png"))
    ffmpeg_mp4(fdir, os.path.join(GAL, f"{args.name}_{style}.mp4"), fps=args.fps)
    ffmpeg_gif(fdir, os.path.join(GAL, f"{args.name}_{style}.gif"), fps=12, width=540, step=3)
    print("wrote", style, len(os.listdir(fdir)))
