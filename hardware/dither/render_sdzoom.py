"""Render the sigma-delta idle-tone zooms (compute_sdzoom.py): zoom-sequence plates with p/q annotations,
an axis-free hero print, the order-2 companion, and a zoom movie (MP4 + GIF).

Annotation: every rational rho = p/q inside a panel with the smallest denominators is marked at the right edge; the
rays of fold(k rho) for all k that are multiples of... all k meet at that row (frequency j/q), so the brighter the
convergence, the smaller q. Styles: dark | paper (| riso for the hero).
Run: OMP_NUM_THREADS=4 python render_sdzoom.py [seq] [hero] [movie]
"""
import os
import subprocess
import sys
from fractions import Fraction
import numpy as np
from PIL import Image
import render_common as rc

Z = np.load(f"{rc.CACHE}/sdzoom_static.npz")
PHI = (1 + 5 ** 0.5) / 2
LO, HI, G = -52.0, -20.0, 0.55


def rationals(a, b, nmax=6, qmax=400):
    found = []
    for q in range(1, qmax + 1):
        for p in range(int(np.ceil(a * q)), int(np.floor(b * q)) + 1):
            fr = Fraction(p, q)
            if fr.denominator == q and a <= p / q <= b:
                found.append(fr)
        if len(found) >= nmax:
            break
    return found[:nmax]


def colorize(v, st):
    if st == "dark":
        return rc.cmap_rgb(v, "magma")
    if st == "paper":
        return rc.ink_on_paper(v, gamma=1.0)
    blue = rc.floyd_steinberg(v)
    return rc.multiply_layers([(blue, rc.RISO_BLUE), (rc.shift(rc.floyd_steinberg(np.clip((v - .5) / .4, 0, 1)), 2, -2), rc.RISO_PINK)])


def panel_img(S, size, lo=LO, hi=HI):
    v = rc.unit(S.astype(np.float32), lo, hi, G)
    return np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((size, size), Image.BOX)) / 65535


def seq(st, name, title, n=5, prefix="gold", lo=LO, hi=HI, order=1):
    P, gap, L, T, lab = 900, 150, 90, 250, 150
    dark = st == "dark"
    bg = (0.02, 0.018, 0.03) if dark else rc.PAPER
    fg = (0.86, 0.83, 0.78) if dark else rc.INK
    acc = (1.0, 0.75, 0.4) if dark else (0.75, 0.12, 0.1)
    W = L + n * P + (n - 1) * gap + lab
    cv = rc.canvas(W, T + P + 230, bg)
    rc.text(cv, (L, 40), title, 60, fg, rc.FONT_SERIF)
    hws = Z["hws"]
    for i in range(n):
        S = Z[f"{prefix}_{i}"]
        rho = Z[f"{prefix}_{i}_rho"]
        x0 = L + i * (P + gap)
        rc.paste(cv, colorize(panel_img(S, P, lo, hi), st), x0, T)
        a, b = float(rho[-1]), float(rho[0])
        rc.text(cv, (x0, T - 50), f"rho in [{a:.6f}, {b:.6f}]   x{int(round(hws[0] / hws[i]))}", 24, fg, rc.FONT_MONO)
        m = 0.03 * (b - a)
        for fr in rationals(a + m, b - m, nmax=5 if i < 3 else 4):
            y = T + int((b - fr.numerator / fr.denominator) / (b - a) * (P - 1))
            rc.line(cv, [(x0 + P, y), (x0 + P + 18, y)], acc, 3)
            rc.text(cv, (x0 + P + 24, y), f"{fr.numerator}/{fr.denominator}", 26, acc, rc.FONT_MONO, anchor="lm")
        if i < n - 1:  # bracket of the next window
            a2, b2 = float(Z[f"{prefix}_{i + 1}_rho"][-1]), float(Z[f"{prefix}_{i + 1}_rho"][0])
            y1 = T + int((b - b2) / (b - a) * (P - 1))
            y2 = T + int((b - a2) / (b - a) * (P - 1))
            rc.line(cv, [(x0 - 12, y1), (x0 - 4, y1), (x0 - 4, y2), (x0 - 12, y2)], acc, 3)
    rc.text(cv, (L, T + P + 40), "x: frequency 0 .. fs/2 (never zoomed).  y: DC input, as rotation number rho = (1+u)/2 (zoomed 5x per panel; "
            "orange bracket at the left of each panel = next window).  labels: rationals p/q in the window with the smallest q.", 26, fg, rc.FONT_SANS)
    rc.text(cv, (L, T + P + 80), f"1-bit {['', 'first', 'second'][order]}-order sigma-delta, 2048 inputs/panel, 2^14 samples each after 2048 warm-up, Blackman-Harris, "
            f"max-pooled 4x in f, colour [{lo:.0f}, {hi:.0f}] dB, gamma {G} (declared).  " + rc.STACK, 22, fg, rc.FONT_MONO)
    rc.save_png(cv, f"{rc.GAL}/{name}_{st}.png")
    print(name, st)


def hero(st):
    D = np.load(f"{rc.CACHE}/sd_dc_maps.npz")
    S = D["mz"].astype(np.float32)[::-1]       # 4096 x 4096, u in [0.30, 0.42] -> rho [0.65, 0.71]
    v = rc.unit(S, LO, HI, G)
    strip = 90
    dark = st == "dark"
    bg = (0, 0, 0) if dark else rc.PAPER
    fg = (0.6, 0.58, 0.55) if dark else (0.3, 0.3, 0.3)
    cv = rc.canvas(4096, 4096 + strip, bg)
    rc.paste(cv, colorize(v, st), 0, 0)
    rc.text(cv, (40, 4096 + 28), "idle tones of a 1-bit first-order sigma-delta modulator.  x: frequency 0 to fs/2.  y: DC input 0.30 "
            "(bottom) to 0.42 (top), i.e. rotation number 0.65 to 0.71; the star is rho = 2/3.  2^14 samples per row.  " + rc.STACK,
            34, fg, rc.FONT_MONO)
    rc.save_png(cv, f"{rc.GAL}/hero_sd_idle_zoom_{st}.png")
    print("hero", st)


def movie():
    F = np.load(f"{rc.CACHE}/sdzoom_movie.npy", mmap_mode="r")
    hw = np.load(f"{rc.CACHE}/sdzoom_movie_hw.npy")
    c = 1 / PHI
    out = f"{rc.GAL}/sd_zoom_golden.mp4"
    W = H = 1080
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames_small = []
    seqidx = list(range(len(F))) + [len(F) - 1] * 36
    for j, i in enumerate(seqidx):
        v = rc.unit(F[i].astype(np.float32), LO, HI, G)
        img = np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((1000, 1000), Image.BOX)) / 65535
        cv = rc.canvas(W, H, (0.02, 0.018, 0.03))
        rc.paste(cv, rc.cmap_rgb(img, "magma"), 0, 0)
        a, b = c - hw[i], c + hw[i]
        for fr in rationals(a + 0.03 * (b - a), b - 0.03 * (b - a), nmax=4):
            y = int((b - fr.numerator / fr.denominator) / (b - a) * 999)
            rc.line(cv, [(1000, y), (1012, y)], (1.0, 0.75, 0.4), 3)
            rc.text(cv, (1016, y), f"{fr.numerator}/{fr.denominator}", 17, (1.0, 0.75, 0.4), rc.FONT_MONO, anchor="lm")
        rc.text(cv, (12, 1010), f"zooming toward rho = 1/phi = 0.618034...   window half-width {hw[i]:.2e}   (x{hw[0] / hw[i]:,.0f})",
                24, (0.86, 0.83, 0.78), rc.FONT_MONO)
        rc.text(cv, (12, 1046), "1-bit sigma-delta idle tones; x: frequency 0..fs/2, y: DC input. labels: smallest-q rationals (Fibonacci ratios)",
                17, (0.7, 0.68, 0.64), rc.FONT_MONO)
        arr = np.array(cv)
        proc.stdin.write(arr.tobytes())
        if j % 2 == 0:
            frames_small.append(Image.fromarray(arr).resize((540, 540), Image.LANCZOS))
    proc.stdin.close(); proc.wait()
    gif = f"{rc.GAL}/sd_zoom_golden.gif"
    frames_small[0].save("/tmp/sdz.gif", save_all=True, append_images=frames_small[1:], duration=83, loop=0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/sdz.gif", "-vf",
                    "split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", gif], check=True)
    print("movie", os.path.getsize(out) / 1e6, "MB; gif", os.path.getsize(gif) / 1e6, "MB")


if __name__ == "__main__":
    which = sys.argv[1:] or ["seq", "hero", "movie"]
    if "seq" in which:
        for st in ("dark", "paper"):
            seq(st, "sd_zoomseq_golden", "Zooming into the fan toward rho = 1/phi: Fibonacci ratios everywhere", prefix="gold")
            seq(st, "sd_zoomseq_twothirds", "Zooming into the fan at rho = 2/3: a star that never resolves", prefix="r23")
            seq(st, "sd_zoomseq_twothirds_order2", "The same windows, second-order modulator: mostly noise, one star", n=3, prefix="r23o2",
                lo=-40.0, hi=-8.0, order=2)
    if "hero" in which:
        for st in ("dark", "paper", "riso"):
            hero(st)
    if "movie" in which:
        movie()
