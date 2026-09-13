"""Continuous zoom videos (GPU). usage:
  render_zoom_anim.py gp <kernel> <seed>     -> gallery/zoom_<kernel>_{plotter,dark}.mp4
  render_zoom_anim.py widths                 -> gallery/zoom_width_limit.mp4 (GP | n=16384 | n=1024, L=2)
Band fade-in between B/2 and B is a declared rendering choice (see zoom_engine)."""
import sys, os, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(0.10)
from zoom_engine import *
from render_common import *
from nets import HeavisideNet
import cmcrameri.cm as cmc

CENTER = np.array([0.2, 0.6, 0.77])
FPD = 45                      # frames per doubling of magnification
mode = sys.argv[1]


def label(img, txt, xy, size=30, fill=(27, 27, 34)):
    ImageDraw.Draw(img).text(xy, txt, font=font(size, "mono"), fill=fill)


if mode == "gp":
    kernel, seed = sys.argv[2], int(sys.argv[3])
    L = int(kernel.split("_L")[1])
    M = MultiScaleField(kernel, seed, CENTER, nbands=10)
    u = float(np.load(f"cache/multiscale_{kernel}_s{seed}.npz")["u"])
    F0, F1 = np.pi / 4, 2.4e-7
    nfr = int(np.ceil(np.log2(F0 / F1) * FPD))
    dirs = {s: f"cache/frames_zoom_{kernel}_{s}" for s in ["plotter", "dark"]}
    for dd in dirs.values():
        os.makedirs(dd, exist_ok=True)
    for i in range(nfr):
        if all(os.path.exists(f"{dd}/{i:05d}.png") for dd in dirs.values()):
            continue
        F = F0 * 2.0 ** (-i / FPD)
        f = M.eval(0.0, 0.0, F, 2160, fade=True)
        cov = ink_coverage(f, u, 2, weight=1)
        im = to_img(mix(PAPER, INK, np.clip(cov * 1.8, 0, 1) ** 0.9))
        label(im, f"x{F0/F:,.0f}", (36, 1080 - 60), 28, (27, 27, 34))
        label(im, f"{F:.2e} rad", (1080 - 260, 1080 - 60), 28, (27, 27, 34))
        im.save(f"{dirs['plotter']}/{i:05d}.png")
        sd = np.sqrt(2 * d_of_theta(kernel.split('_L')[0], L, F / 4))
        g = downsample(f.astype(np.float32), 2)
        t = np.clip(0.5 + 0.5 * (g - u) / (2.2 * sd), 0, 1)
        rgb = mix(cmc.berlin(t)[..., :3], np.array([1.0, 0.93, 0.8]), np.clip(cov * 1.4, 0, 0.85))
        im = to_img(rgb)
        label(im, f"x{F0/F:,.0f}", (36, 1080 - 60), 28, (200, 196, 186))
        label(im, f"{F:.2e} rad", (1080 - 260, 1080 - 60), 28, (200, 196, 186))
        im.save(f"{dirs['dark']}/{i:05d}.png")
        if i % 45 == 0:
            print(i, nfr, f"F={F:.2e}", flush=True)
    for s, dd in dirs.items():
        write_video(dd, f"gallery/zoom_{kernel}_{s}.mp4", fps=30, gif=f"gallery/zoom_{kernel}_{s}.gif", gif_width=480, gif_fps=15)

if mode == "widths":
    kernel, seed = "heaviside_L2", 1
    M = MultiScaleField(kernel, seed, CENTER, nbands=10)
    u_gp = float(np.load(f"cache/multiscale_{kernel}_s{seed}.npz")["u"])
    nets = []
    for n in [16384, 1024]:
        z = np.load(f"cache/nets_zoom_L2_n{n}.npz")
        nets.append((n, HeavisideNet(n, 2, 11), z["p"], float(z["u"])))
    P, SS = 600, 2
    F0, F1 = np.pi / 4 / 16, 2.4e-7
    FPD = 30
    nfr = int(np.ceil(np.log2(F0 / F1) * FPD))
    dd = "cache/frames_zoom_widths"
    os.makedirs(dd, exist_ok=True)
    xs = [30, 660, 1290]
    for i in range(nfr):
        path = f"{dd}/{i:05d}.png"
        if os.path.exists(path):
            continue
        F = F0 * 2.0 ** (-i / FPD)
        img = to_img(np.ones((1080, 1920, 3)) * PAPER)
        dr = ImageDraw.Draw(img)
        f = M.eval(0.0, 0.0, F, P * SS, fade=True)
        panels = [("n = infinity (Gaussian process)", f, u_gp)]
        for n, net, p, u in nets:
            v, _, _ = gnomonic_patch(p, F / 2, P * SS)
            panels.append((f"n = {n:,}", net(v), u))
        for x, (title, ff, u) in zip(xs, panels):
            cov = ink_coverage(ff, u, SS, weight=1)
            img.paste(to_img(mix(PAPER, INK, np.clip(cov * 1.8, 0, 1) ** 0.9)), (x, 250))
            dr.text((x, 250 + P + 24), title, font=font(38), fill=(27, 27, 34))
        dr.text((30, 60), "Width as a zoom limit", font=font(64), fill=(27, 27, 34))
        dr.text((30, 150), "Heaviside, depth 2. A finite network is piecewise constant on the cells of its n first-layer great circles;\n"
                "zoom past the cell size (~1/n rad) and its coastline straightens into arcs. The infinite-width limit never does.",
                font=font(30, "italic"), fill=(110, 105, 96), spacing=8)
        dr.text((30, 250 + P + 90), f"field of view {F:.2e} rad    magnification x{np.pi/4/F:,.0f}", font=font(32, "mono"), fill=(27, 27, 34))
        img.save(path)
        if i % 15 == 0:
            print(i, nfr, f"F={F:.2e}", flush=True)
    write_video(dd, "gallery/zoom_width_limit.mp4", fps=30, gif="gallery/zoom_width_limit.gif", gif_width=720, gif_fps=12)
