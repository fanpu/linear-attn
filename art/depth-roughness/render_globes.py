"""Globes: level sets / excursion sets of the depth-L GP draws on S^2, orthographic.
usage: render_globes.py stills | spin <kernel> <style> | series"""
import sys, os, numpy as np
from scipy import ndimage
from render_common import *
from globe import *
import cmcrameri.cm as cmc


def shade_style(f, mask, Z, c, style, ss, levels=None):
    """f: supersampled field (nan outside), returns rgb at f.shape//ss."""
    ff = np.where(mask, f, c)
    inner = ndimage.binary_erosion(mask, iterations=2)
    Zd = downsample(Z.astype(np.float32), ss)
    md = downsample(mask.astype(np.float32), ss)
    def lines(lv, w=1):
        e = boundary(ff, lv) & inner
        if w > 1:
            e = ndimage.binary_dilation(e, structure=np.ones((w, w), bool))
        return downsample(e.astype(np.float32), ss)
    n = f.shape[0] // ss
    s = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(s, -s)
    rim = np.clip(1 - np.abs(np.hypot(X, Y) - 1) / (1.6 / n * 2), 0, 1)   # outline ~1.6 px
    if style == "plotter":
        lv = [c] if levels is None else levels
        cov = np.zeros((n, n), np.float32)
        for l_ in lv:
            cov = np.maximum(cov, lines(l_))
        a = np.clip(cov * 2.0, 0, 1) ** 0.9
        rgb = mix(PAPER, INK, np.maximum(a, rim * 0.9))
        return rgb
    if style == "dark":
        fd = downsample(np.where(mask, f, np.nan).astype(np.float32), ss)
        lo, hi = np.nanpercentile(fd, [1, 99])
        t = np.clip((np.nan_to_num(fd, nan=lo) - lo) / (hi - lo), 0, 1)
        rgb = cmc.oslo(t)[..., :3] * (0.25 + 0.75 * np.clip(Zd, 0, 1) ** 0.5)[..., None]
        rgb = mix(DARK, rgb, md)
        ln = lines(c)
        rgb = mix(rgb, np.array([1.0, 0.84, 0.52]), np.clip(ln * 1.5, 0, 0.8) * (0.4 + 0.6 * np.clip(Zd, 0, 1)))
        return rgb
    if style == "gold":
        # flat lighting: the field is a quiet dark relief, the level set (gold) is the dominant mark
        fd = downsample(np.where(mask, f, np.nan).astype(np.float32), ss)
        lo, hi = np.nanpercentile(fd, [1, 99])
        t = np.clip((np.nan_to_num(fd, nan=lo) - lo) / (hi - lo), 0, 1)
        rgb = cmc.oslo(0.08 + 0.32 * t)[..., :3]
        rgb = mix(DARK, rgb, md)
        rgb = mix(rgb, np.array([0.55, 0.55, 0.6]), rim * 0.6)
        ln = lines(c)
        return mix(rgb, np.array([1.0, 0.80, 0.38]), np.clip(ln * 2.2, 0, 1))
    if style == "riso":
        up = downsample(((ff > c) & mask).astype(np.float32), ss)
        up = np.pad(up, ((2, 0), (0, 3)), mode="edge")[:-2, 3:]     # declared misregistration
        ln = np.maximum(lines(c, 2), rim)
        return multiply_ink(PAPER, [(RISO_PINK, 0.85 * up), (RISO_BLUE, 0.9 * np.clip(ln * 1.3, 0, 1))])
    raise ValueError(style)


def render_globe(name, lon, lat, n=1080, ss=2, style="plotter", lmax=2048, alm=None, c=None, levels=None):
    if alm is None:
        alm = make_alm(name, lmax)
    if c is None:
        c = sphere_median(alm, lmax)
    R = rotation(lon, lat)
    f, mask, Z = globe_field(alm, lmax, n * ss, R)
    return shade_style(f, mask, Z, c, style, ss, levels)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "series":
        # 2 x 3 sheet: Heaviside L=1..6 from one viewpoint, plus regular activations strip
        style = sys.argv[2] if len(sys.argv) > 2 else "plotter"
        G, PAD, TOP = 900, 90, 250
        keys = [f"heaviside_L{L}" for L in range(1, 7)] + ["relu_L4", "gelu_L6", "sin_L8"]
        cols, rows = 3, 3
        W = cols * G + (cols + 1) * PAD
        H = TOP + rows * (G + 110) + 120
        bg = DARK if style == "dark" else PAPER
        fgc = (225, 222, 214) if style == "dark" else (27, 27, 34)
        img = to_img(np.ones((H, W, 3)) * bg)
        dr = ImageDraw.Draw(img)
        dr.text((PAD, 60), "Depth as the Dial: globes", font=font(80), fill=fgc)
        dr.text((PAD, 160), "same random draw, viewed from (lon 30, lat 25); row 3: regular activations at depth",
                font=font(34, "italic"), fill=fgc)
        from common import theory
        for i, k in enumerate(keys):
            a, L = k.split("_L"); L = int(L)
            rgb = render_globe(k, 30, 25, n=G, ss=3, style=style, lmax=4096)
            x = PAD + (i % cols) * (G + PAD); y = TOP + (i // cols) * (G + 110)
            img.paste(to_img(rgb), (x, y))
            th = theory(a, L)
            lab = f"{a}  L={L}   dimH {th['dimH']:.3f}" if a == "heaviside" else f"{a}  L={L}   dimH 1   E len x{th['exp_len']/2/np.pi:.2f}"
            dr.text((x + 20, y + G + 20), lab, font=font(34, "mono"), fill=fgc)
            print(k, flush=True)
        img.save(f"gallery/globes_series_{style}.png")
    if mode == "hero":
        name, style = sys.argv[2], sys.argv[3]
        rgb = render_globe(name, 30, 25, n=2400, ss=2, style=style, lmax=4096)
        to_img(rgb).save(f"gallery/globe_hero_{name}_{style}.png")
    if mode == "spin":
        name, style = sys.argv[2], sys.argv[3]
        nfr = int(sys.argv[4]) if len(sys.argv) > 4 else 480
        d = f"cache/frames_spin_{name}_{style}"
        os.makedirs(d, exist_ok=True)
        lmax = 2048
        alm = make_alm(name, lmax); c = sphere_median(alm, lmax)
        for i in range(nfr):
            p = f"{d}/{i:05d}.png"
            if os.path.exists(p):
                continue
            lon = 30 + 360 * i / nfr
            lat = 25 * np.cos(2 * np.pi * i / nfr) * 0.6 + 10       # gentle nod (declared camera path)
            rgb = render_globe(name, lon, lat, n=1080, ss=2, style=style, lmax=lmax, alm=alm, c=c)
            to_img(rgb).save(p)
            if i % 40 == 0:
                print(name, style, i, flush=True)
        write_video(d, f"gallery/globe_spin_{name}_{style}.mp4", fps=30, gif=f"gallery/globe_spin_{name}_{style}.gif", gif_width=480, gif_fps=15)
