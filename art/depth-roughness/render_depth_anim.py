"""Depth as time: the same white a_lm (common random numbers) scaled by sqrt(C_l) of kernel(tau),
where between integer depths the kernel is the declared mixture (1-tau) kappa_L + tau kappa_{L+1}
(a valid covariance; only integer-L holds are the paper's fields). Heaviside (left) vs ReLU (right)
on the tile patch. usage: render_depth_anim.py <style>"""
import sys, os, json, numpy as np
from render_common import *
from common import *
from globe import spec
import cmcrameri.cm as cmc

style = sys.argv[1] if len(sys.argv) > 1 else "plotter"
LMAX = 2048
P, SS = 880, 2
HOLD, TRANS = 50, 70
LMAXD = 8
d = f"cache/frames_depth_{style}"
os.makedirs(d, exist_ok=True)
z = white_alm(LMAX, 11)
v = lambert_patch(np.array([0.3, -0.5, 0.8]), 0.7, P * SS)
th, ph = vec_to_thetaphi(v)
bc = json.load(open("cache/boxcount_tiles.json"))["n2048"]
cal = json.load(open("cache/calibration.json"))
bg = DARK if style == "dark" else PAPER
fg = (226, 222, 212) if style == "dark" else (27, 27, 34)
dim = (140, 138, 132) if style == "dark" else (120, 116, 108)

def panel(C):
    f = synth(alm_from_white(z, C, LMAX), LMAX, th, ph, nthreads=6)
    c = np.median(f)
    if style == "plotter":
        cov = ink_coverage(f, c, SS, weight=1)
        return mix(PAPER, INK, np.clip(cov * 1.9, 0, 1) ** 0.9)
    g = downsample(f.astype(np.float32), SS)
    lo, hi = np.percentile(g, [1, 99])
    rgb = cmc.lajolla(np.clip((g - lo) / (hi - lo), 0, 1))[..., :3] * 0.92
    return mix(rgb, np.array([0.05, 0.05, 0.07]), np.clip(ink_coverage(f, c, SS) * 1.6, 0, 0.9))

schedule = []
for L in range(1, LMAXD + 1):
    schedule += [(L, 0.0)] * HOLD
    if L < LMAXD:
        schedule += [(L, (1 - np.cos(np.pi * (k + 1) / (TRANS + 1))) / 2) for k in range(TRANS)]
cache = {}
for i, (L, tau) in enumerate(schedule):
    path = f"{d}/{i:05d}.png"
    if os.path.exists(path):
        continue
    key = (L, round(tau, 6))
    if key not in cache:
        cache.clear()
        panels = []
        for a in ["heaviside", "relu"]:
            C = spec(f"{a}_L{L}", LMAX) if tau == 0 else (1 - tau) * spec(f"{a}_L{L}", LMAX) + tau * spec(f"{a}_L{L+1}", LMAX)
            panels.append(panel(C))
        cache[key] = panels
    panels = cache[key]
    img = to_img(np.ones((1080, 1920, 3)) * bg)
    dr = ImageDraw.Draw(img)
    x0s = [60, 1920 - 60 - P]
    for x0, pn, a in zip(x0s, panels, ["Heaviside", "ReLU"]):
        img.paste(to_img(pn), (x0, 60))
        dr.text((x0, 60 + P + 22), a, font=font(40), fill=fg)
    Ls = f"L = {L}" if tau == 0 else f"L = {L} → {L+1}"
    dr.text((960 - 70, 80), "depth", font=font(30, "italic"), fill=dim)
    dr.text((960 - 70, 118), Ls if tau == 0 else f"{L}→{L+1}", font=font(54), fill=fg)
    if tau == 0:
        k = f"heaviside_L{L}"
        s, cnt = np.array(bc[k]["sizes"]), np.array(bc[k]["counts"])
        D = fit_dim(s, cnt, 4, 64)[0]
        ce = cal[f"tile_H{2.0**-L:.5f}"]["D_meas"]
        tx = [f"dimH  {2-2.0**-L:.3f}", f"box D {D:.2f}", f"(exp. {ce:.2f})"]
        dr.text((60 + 300, 60 + P + 28), "   ".join(tx), font=font(30, "mono"), fill=fg)
        dr.text((x0s[1] + 200, 60 + P + 28), "dimH 1   E len x1.00", font=font(30, "mono"), fill=fg)
    img.save(path)
    if i % 50 == 0:
        print(i, len(schedule), flush=True)
write_video(d, f"gallery/depth_as_time_{style}.mp4", fps=30, gif=f"gallery/depth_as_time_{style}.gif", gif_width=720, gif_fps=15)
