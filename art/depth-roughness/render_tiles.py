"""Piece 1: the tile grid. rows = depth L, columns = activation. Each tile: level set of an exact
(band-limited, lmax 4096) sample of the depth-L infinite-width GP on the same Lambert patch of S^2,
common random numbers across all tiles. Level c = tile median (declared)."""
import sys, json, numpy as np
from render_common import *
from common import ACTS, ACT_LABEL, fit_dim, theory
import cmcrameri.cm as cmc

style = sys.argv[1] if len(sys.argv) > 1 else "plotter"
LS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 9))
tail = "--tail" in sys.argv
d = np.load("cache/tiles_2048.npz")
bc = json.load(open("cache/boxcount_tiles.json"))["n2048"]
cal = json.load(open("cache/calibration.json"))
T, GAP, LM, TOP, ROWTXT = 512, 44, 170, 300, 92
W = LM + len(ACTS) * T + (len(ACTS) - 1) * GAP + 90
H = TOP + len(LS) * (T + ROWTXT) + 230

bgcol = DARK if style in ("dark",) else PAPER
fgcol = np.array([0.85, 0.84, 0.8]) if style == "dark" else INK
canvas = np.ones((H, W, 3)) * bgcol
if style != "dark":
    canvas *= grain((H, W), 1, 0.012)[..., None]


def tile(key):
    f = d[key].astype(np.float64)
    info = bc[key]
    if tail:
        f = f + np.sqrt(max(1 - info["var_captured"], 0)) * d["tail_noise"]
    c = np.median(f)
    if style == "plotter":
        cov = ink_coverage(f, c, 4, weight=1)
        return mix(PAPER, INK, np.clip(cov * 2.2, 0, 1) ** 0.9)
    if style == "riso":
        up = downsample((f > c).astype(np.float32), 4)
        line = ink_coverage(f, c, 4, weight=3)
        up = np.pad(up, ((3, 0), (0, 2)), mode='edge')[:-3, 2:]   # declared misregistration (3,-2) px
        return multiply_ink(PAPER, [(RISO_BLUE, 0.78 * (1 - up)), (RISO_PINK, 0.92 * line)])
    if style == "dark":
        g = downsample(f.astype(np.float32), 4)
        lo, hi = np.percentile(g, [1, 99])
        rgb = cmc.oslo(np.clip((g - lo) / (hi - lo + 1e-12), 0, 1))[..., :3] * 0.9
        line = ink_coverage(f, c, 4, weight=2)
        line = ink_coverage(f, c, 4, weight=1)
        amax = 0.5 if key.startswith("heaviside") else 0.85   # round 2: let the field show through dense coastlines
        return mix(rgb, np.array([1.0, 0.86, 0.55]), np.clip(line * 1.6, 0, amax))
    if style == "topo":
        if key.startswith("heaviside"):
            qs, wts = [c], [1]
        else:
            qs = np.percentile(f[::4, ::4], np.linspace(4, 96, 13))
            wts = [2 if i == 6 else 1 for i in range(len(qs))]
        cov = ink_coverage(f, qs, 4, weights=wts)
        return mix(PAPER, np.array([0.45, 0.22, 0.1]), np.clip(cov * 1.8, 0, 1))


img = to_img(canvas)
dr = ImageDraw.Draw(img)
fg = tuple(int(x * 255) for x in fgcol)
dim = tuple(int(x * 255) for x in (fgcol * 0.6 + bgcol * 0.4))
dr.text((LM, 70), "Depth as the Dial", font=font(92, "serif"), fill=fg)
sub = {"plotter": "level sets of infinite-width random networks on the sphere, one ink",
       "riso": "excursion sets {f > c} (blue paper-out) and their boundary (pink), two inks",
       "dark": "the field itself (oslo, per-tile percentile stretch) with its level line",
       "topo": "regular activations: 13 quantile contours (median heavy); Heaviside: the median contour only"}[style]
dr.text((LM, 185), sub + (" — with sub-pixel variance restored" if tail else ""), font=font(38, "italic"), fill=dim)
for j, a in enumerate(ACTS):
    x = LM + j * (T + GAP)
    dr.text((x, TOP - 58), ACT_LABEL[a], font=font(44, "serif"), fill=fg)
for i, L in enumerate(LS):
    y = TOP + i * (T + ROWTXT)
    dr.text((34, y + T // 2 - 30), f"L={L}", font=font(46, "serif"), fill=fg)
    for j, a in enumerate(ACTS):
        key = f"{a}_L{L}"
        x = LM + j * (T + GAP)
        img.paste(to_img(tile(key)), (x, y))
        info = bc[key]
        s, cnt = np.array(info["sizes"]), np.array(info["counts"])
        D, se, _ = fit_dim(s, cnt, 4, 64)
        th = theory(a, L)["dimH"]
        if a == "heaviside":
            ce = cal[f"tile_H{2.0**-L:.5f}"]["D_meas"]
            txt = f"D {D:.2f} | exp {ce:.2f} | dimH {th:.3f}"
        else:
            txt = f"D {D:.2f} | dimH 1"
        dr.text((x, y + T + 12), txt, font=font(27, "mono"), fill=fg)
        extra = f"{100*(1-info['var_captured']):.0f}% var above l=4096" if a == "heaviside" else f"E len x{theory(a,L)['exp_len']/(2*np.pi):.2f}"
        dr.text((x, y + T + 48), extra, font=font(24, "mono"), fill=dim)
foot = ("Exact band-limited samples (l <= 4096) of the Gaussian-process limit of T_L on S^2 (Di Lillo et al. 2025); the same random spherical-harmonic draw in every tile.\n"
        "Lambert equal-area patch 82 deg across, 2048^2 samples, level c = tile median. D = box-counting slope over 4-64 px (2.7e-3 to 4.4e-2 rad).\n"
        "dimH = 2 - 2^-L (Heaviside, fractal class) or 1 (Kac-Rice class). 'exp' = what this same estimator returns on synthetic k^-(2+2H) Gaussian fields\n"
        "whose level sets have exactly dimension 2 - H (8 seeds, sd < 0.01): the finite-range bias, not a fit. 'E len x' = kappa'(1)^(L/2), the paper's nodal-length factor.")
dr.multiline_text((LM, H - 200), foot, font=font(24, "serif"), fill=dim, spacing=9)
name = f"gallery/tiles_{style}{'_tail' if tail else ''}.png"
img.save(name)
print(name, img.size)
