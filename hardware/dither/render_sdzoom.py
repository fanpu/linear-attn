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


def panel_img(S, size, lo=LO, hi=HI, maxpool=False):
    S = S.astype(np.float32)
    if maxpool:  # 2x2 max-pool first so 1-px lines survive the downsample (declared, deep panels only)
        S = S[: S.shape[0] // 2 * 2, : S.shape[1] // 2 * 2].reshape(S.shape[0] // 2, 2, S.shape[1] // 2, 2).max((1, 3))
    v = rc.unit(S, lo, hi, G)
    return np.array(Image.fromarray((v * 65535).astype(np.uint16)).resize((size, size), Image.BOX)) / 65535


class _Merged(dict):
    pass


def deep_source():
    """Levels 0-2 from the N=2^14 static run, levels 3-5 from the N=2^20 deep run."""
    d = np.load(f"{rc.CACHE}/sdzoom_deep.npz")
    m = _Merged()
    for i in range(3):
        m[f"gold_{i}"] = Z[f"gold_{i}"]; m[f"gold_{i}_rho"] = Z[f"gold_{i}_rho"]
    for i in range(3, 6):
        m[f"gold_{i}"] = d[f"gold_{i}"]; m[f"gold_{i}_rho"] = d[f"gold_{i}_rho"]
    m["hws"] = np.array([0.03 / 5 ** i for i in range(6)])
    return m


def seq(st, name, title, n=5, prefix="gold", lo=LO, hi=HI, order=1, src=None):
    global Z
    Zsave = Z
    if src is not None:
        Z = src
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
        deep = src is not None and i >= 3
        rc.paste(cv, colorize(panel_img(S, P, -64.0 if deep else lo, -30.0 if deep else hi, maxpool=deep), st), x0, T)
        a, b = float(rho[-1]), float(rho[0])
        ftxt = ""
        if src is not None and i >= 3:
            ftxt = "   f in [0.372, 0.392]"
        rc.text(cv, (x0, T - 50), f"rho in [{a:.7f}, {b:.7f}]   x{int(round(hws[0] / hws[i]))}{ftxt}", 22, fg, rc.FONT_MONO)
        m = 0.05 * (b - a)
        used = []
        for fr in rationals(a + m, b - m, nmax=5 if i < 3 else 4):
            y = T + int((b - fr.numerator / fr.denominator) / (b - a) * (P - 1))
            if any(abs(y - u) < 34 for u in used):
                continue
            used.append(y)
            rc.line(cv, [(x0 + P, y), (x0 + P + 18, y)], acc, 3)
            rc.text(cv, (x0 + P + 24, y), f"{fr.numerator}/{fr.denominator}", 26, acc, rc.FONT_MONO, anchor="lm")
        if i < n - 1:  # bracket of the next window
            a2, b2 = float(Z[f"{prefix}_{i + 1}_rho"][-1]), float(Z[f"{prefix}_{i + 1}_rho"][0])
            y1 = T + int((b - b2) / (b - a) * (P - 1))
            y2 = T + int((b - a2) / (b - a) * (P - 1))
            rc.line(cv, [(x0 - 12, y1), (x0 - 4, y1), (x0 - 4, y2), (x0 - 12, y2)], acc, 3)
    rc.text(cv, (L, T + P + 40), "x: frequency (0 .. fs/2 unless marked).  y: DC input, as rotation number rho = (1+u)/2 (zoomed 5x per panel; "
            "bracket at the left of each panel = next window).  labels: rationals p/q in the window with the smallest q.", 26, fg, rc.FONT_SANS)
    floor_txt = ("resolution floor: inputs are distinguishable only if their rotation numbers differ by more than ~1/N. Panels x1-x25: "
                 "N = 2^14 (floor 6e-5), full band. Panels x125-x3125: N = 2^20 (floor 9.5e-7), and frequency cropped to 0.372-0.392 around fold(1/phi).") if src is not None else \
        ("resolution floor: two inputs are distinguishable only if their rotation numbers differ by more than ~1/N = 6e-5, "
         "so panels x125 and x625 show the finite-length (phase-slip / window) kernel, not finer number theory.")
    rc.text(cv, (L, T + P + 120), floor_txt, 26, fg, rc.FONT_SANS)
    rc.text(cv, (L, T + P + 165), f"1-bit {['', 'first', 'second'][order]}-order sigma-delta, 2048 inputs/panel, {'2^14 / 2^20' if src is not None else '2^14'} samples each after warm-up, Blackman-Harris, "
            f"max-pooled 4x in f, colour [{lo:.0f}, {hi:.0f}] dB{' (deep panels: 2x2 max-pool, [-64, -30] dB)' if src is not None else ''}, gamma {G} (declared).  " + rc.STACK, 22, fg, rc.FONT_MONO)
    rc.save_png(cv, f"{rc.GAL}/{name}_{st}.png")
    print(name, st)
    Z = Zsave


def hero(st):
    D = np.load(f"{rc.CACHE}/sd_dc_maps.npz")
    S = D["mz"].astype(np.float32)[::-1]       # 4096 x 4096, u in [0.30, 0.42] -> rho [0.65, 0.71]
    v = rc.unit(S, LO, HI, G)
    strip = 110
    dark = st == "dark"
    bg = (0, 0, 0) if dark else rc.PAPER
    fg = (0.6, 0.58, 0.55) if dark else (0.3, 0.3, 0.3)
    cv = rc.canvas(4096, 4096 + strip, bg)
    rc.paste(cv, colorize(v, st), 0, 0)
    rc.text(cv, (40, 4096 + 22), "idle tones of a 1-bit first-order sigma-delta.  x: frequency 0 to fs/2.  y: DC input 0.30 to 0.42 "
            "(rho 0.65 to 0.71); the star is rho = 2/3.", 34, fg, rc.FONT_MONO)
    rc.text(cv, (40, 4096 + 66), "2^14 samples per row, Blackman-Harris, [-52, -20] dB.  " + rc.STACK, 26, fg, rc.FONT_MONO)
    rc.save_png(cv, f"{rc.GAL}/hero_sd_idle_zoom_{st}.png")
    print("hero", st)


def movie():
    F = np.load(f"{rc.CACHE}/sdzoom_movie.npy", mmap_mode="r")
    hw = np.load(f"{rc.CACHE}/sdzoom_movie_hw.npy")
    c = 1 / PHI
    out = f"{rc.GAL}/sd_zoom_golden.mp4"
    W = H = 1080
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-vf", "scale=864:864:flags=area",
           "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
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
        rc.text(cv, (12, 1010), f"toward rho = 1/phi   half-width {hw[i]:.1e}   x{hw[0] / hw[i]:,.0f}",
                24, (0.86, 0.83, 0.78), rc.FONT_MONO)
        rc.text(cv, (12, 1046), "1-bit sigma-delta idle tones. x: freq 0..fs/2. y: DC input. labels: smallest-q p/q",
                17, (0.7, 0.68, 0.64), rc.FONT_MONO)
        arr = np.array(cv)
        proc.stdin.write(arr.tobytes())
        if j % 3 == 0:
            frames_small.append(Image.fromarray(arr).resize((400, 400), Image.LANCZOS))
    proc.stdin.close(); proc.wait()
    gif = f"{rc.GAL}/sd_zoom_golden.gif"
    frames_small[0].save("/tmp/sdz.gif", save_all=True, append_images=frames_small[1:], duration=125, loop=0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/sdz.gif", "-vf",
                    "split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", gif], check=True)
    print("movie", os.path.getsize(out) / 1e6, "MB; gif", os.path.getsize(gif) / 1e6, "MB")


if __name__ == "__main__":
    which = sys.argv[1:] or ["seq", "hero", "movie"]
    if "seq" in which:
        for st in ("dark", "paper"):
            seq(st, "sd_zoomseq_golden", "Zooming into the fan toward rho = 1/phi: Fibonacci ratios everywhere", prefix="gold")
            seq(st, "sd_zoomseq_twothirds", "Zooming into the fan at rho = 2/3, until finite length takes over", prefix="r23")
            seq(st, "sd_zoomseq_twothirds_order2", "The same windows, second-order modulator: mostly noise, one star", n=3, prefix="r23o2",
                lo=-40.0, hi=-8.0, order=2)
    if "deep" in which:
        src = deep_source()
        for st in ("dark", "paper"):
            seq(st, "sd_zoomseq_golden_deep", "Toward rho = 1/phi, x1 to x3125: Fibonacci ratios at every scale", n=6, src=src)
    if "hero" in which:
        for st in ("dark", "paper", "riso"):
            hero(st)
    if "movie" in which:
        movie()
