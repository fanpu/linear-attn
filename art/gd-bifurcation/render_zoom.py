"""Nested zoom film + plate sequence through the period-3 tower (3 -> 9 -> 27 -> 81 -> 243).

Every frame is an independent full 4-coordinate GD computation (cache/zoom/frame_*.npz), not an image
magnification.  Measured: counts of visited iterates, Lyapunov exponent along the balanced line.
Aesthetic: tone curve (log, floor/ceiling percentiles smoothed over +-15 frames to avoid flicker), fire
palette, easing of the camera path, typography.

Outputs: gallery/zoom_tower.mp4 (1920x1080, H.264, yuv420p, 30 fps), gallery/zoom_tower.gif,
         gallery/zoom_keyframes.png (keyframes)
"""
import glob
import json
import os
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, median_filter
from render_lib import *

FULL = (0.45, 0.99, -0.03, 1.80)
FPS = 30


def frame_tone(C, c0, sat):
    t = np.clip(np.log1p(C / c0) / np.log1p(sat / c0), 0, 1)
    occ = (C > 0).mean(0)
    heavy = (C > sat) & (occ < 0.03)[None, :]
    return np.maximum(t, 0.92 * maximum_filter(heavy.astype(float), size=3))


def sci(v):
    return f"{v:,.0f}" if v < 1e6 else f"{v:.3e}"


def fmt(v, span):
    digs = max(3, int(np.ceil(-np.log10(span))) + 2)
    return f"{v:.{digs}f}"


def main():
    files = sorted(glob.glob(f"{CACHE}/zoom/frame_*.npz"))
    path = json.load(open(f"{CACHE}/zoom/path.json"))
    tower = json.load(open(f"{CACHE}/windows.json"))["tower"]
    n = len(path)
    assert len(files) == n, (len(files), n)
    # percentiles per frame, then temporal smoothing
    stats = []
    for f in files:
        C = np.load(f)["C"].astype(float)
        nz = C[C > 0]
        stats.append((np.percentile(nz, 50), np.percentile(nz, 99.5)))
    stats = np.array(stats)
    sm = np.exp(median_filter(np.log(stats), size=(31, 1), mode="nearest"))
    lut = cmap_lut("cc:fire")
    tmp = "/tmp/gdb/zoomframes"
    os.makedirs(tmp, exist_ok=True)
    for f in glob.glob(tmp + "/*.png"):
        os.remove(f)
    segs = len(tower)
    k = 0
    fdiag, fsmall, fbig = font(30, "mono"), font(24, "mono"), font(44, "serif")
    for i, f in enumerate(files):
        d = np.load(f)
        lo, hi, ylo, yhi = [float(v) for v in d["rect"]]
        C = d["C"].astype(float)
        t = frame_tone(C, *sm[i])
        img = np.zeros((1080, 1920, 3))
        img[0:860] = apply_lut(t, lut)
        # Lyapunov strip (sign-preserving sqrt compression, declared)
        lam = d["lyap_bal"].astype(float)
        lam_c = np.sign(lam) * np.sqrt(np.abs(lam))
        pos, neg, y0, _, _ = lyap_raster(d["etas"], lam_c, lo, hi, 1920, 150, -1.2, 0.8)
        s = np.zeros((150, 1920, 3))
        s[pos] = (235, 120, 40)
        s[neg] = (80, 80, 105)
        s[int(round(y0))] = (140, 140, 140)
        img[900:1050] = s
        # translucent dark panels behind the text (declared legibility aid)
        img[18:150, 20:760] *= 0.35
        img[18:130, 1100:1900] *= 0.35
        pil = to_img(img)
        dr = ImageDraw.Draw(pil)
        zx = (FULL[1] - FULL[0]) / (hi - lo)
        zy = (FULL[3] - FULL[2]) / (yhi - ylo)
        seg = i / (n - 1) * segs
        lvl = min(int(seg + 0.5), segs)
        pl = [1] + [r["base"] for r in tower]
        dr.text((40, 30), f"η  {fmt(lo, hi - lo)} … {fmt(hi, hi - lo)}", fill=(255, 235, 210), font=fdiag)
        dr.text((40, 72), f"P  {fmt(ylo, yhi - ylo)} … {fmt(yhi, yhi - ylo)}", fill=(230, 205, 185), font=fsmall)
        dr.text((40, 108), f"zoom ×{sci(zx)} in η   ×{sci(zy)} in P", fill=(200, 175, 160), font=fsmall)
        dr.text((1880, 30), f"period-{pl[lvl]} window" if lvl else "full cascade", fill=(255, 235, 210), font=fbig, anchor="ra")
        dr.text((1880, 88), "GD on ½(x₁x₂x₃x₄ − 1)², all 4 coordinates, float64", fill=(170, 150, 140), font=fsmall, anchor="ra")
        dr.text((40, 1052), "λ (oscillating mode)", fill=(150, 140, 135), font=font(20, "mono"))
        reps = 1
        # hold at keyframes
        if abs(seg - round(seg)) < 0.5 / (n - 1) * segs + 1e-12:
            reps = int(1.5 * FPS)
        for _ in range(reps):
            pil.save(f"{tmp}/f{k:05d}.png")
            k += 1
        if i == n - 1 or abs(seg - round(seg)) < 1e-9:
            pass
    # plate sequence: keyframes
    key_idx = [int(round(j * (n - 1) / segs)) for j in range(segs + 1)]
    plates = [Image.open(sorted(glob.glob(f"{tmp}/*.png"))[0])]
    fr_files = sorted(glob.glob(f"{tmp}/*.png"))
    # map key frames to written files: recompute cumulative index
    cum = []
    kk = 0
    for i in range(n):
        seg = i / (n - 1) * segs
        reps = int(1.5 * FPS) if abs(seg - round(seg)) < 0.5 / (n - 1) * segs + 1e-12 else 1
        cum.append(kk)
        kk += reps
    plates = [Image.open(fr_files[cum[j]]) for j in key_idx]
    Wp, Hp = 1920, 1080
    sheet = Image.new("RGB", (Wp * 2 + 60 * 3, Hp * 3 + 60 * 4), (6, 6, 8))
    for j, im in enumerate(plates[:6]):
        r, c = divmod(j, 2)
        sheet.paste(im, (60 + c * (Wp + 60), 60 + r * (Hp + 60)))
    sheet.save(f"{GAL}/zoom_keyframes.png", optimize=True)
    print("wrote zoom_keyframes.png", sheet.size)
    mp4 = f"{GAL}/zoom_tower.mp4"
    # two-pass, bitrate-capped (the chaotic-band grain is incompressible; crf 18 gives 260 MB)
    common = ["-framerate", str(FPS), "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-b:v", "4500k", "-maxrate", "5500k", "-bufsize", "11000k", "-preset", "slow"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + common + ["-pass", "1", "-an", "-f", "null", "/dev/null"], check=True, cwd="/tmp/gdb")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + common + ["-pass", "2", mp4], check=True, cwd="/tmp/gdb")
    gif = f"{GAL}/zoom_tower.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=10,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    gif], check=True)
    print("wrote", mp4, os.path.getsize(mp4) // 1e6, "MB;", gif, os.path.getsize(gif) // 1e6, "MB")


if __name__ == "__main__":
    main()
