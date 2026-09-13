"""Cartographic renderings of cached loss surfaces (no GPU).

  python render_maps.py --tag g51            # every style for every model that has <model>_final_<tag>.npz
  python render_maps.py --tag g101 --models resnet56 resnet56_noshort --styles survey hachure

Height = log10(training loss on the fixed 1000-image subset), clipped to [LO, HI] shared by
all models (so paired sheets are directly comparable).  Smooth upsampling (cubic spline of
the log-loss grid) is a declared rendering step; contours are drawn on the upsampled field.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

ROOT = os.path.dirname(os.path.abspath(__file__))
SURF = os.path.join(ROOT, "cache", "surf")
GAL = os.path.join(ROOT, "gallery")
LO, HI = 0.08, 150.0           # shared loss clip (train loss of best model ~0.1; chance = ln 10; edges reach ~100)
CHANCE = np.log(10.0)
ORDER = ["resnet20", "resnet20_noshort", "resnet56", "resnet56_noshort"]
PRETTY = {"resnet20": "ResNet-20", "resnet20_noshort": "ResNet-20, no shortcuts",
          "resnet56": "ResNet-56", "resnet56_noshort": "ResNet-56, no shortcuts"}
TRAIN = {"resnet20": (95.6, 90.4), "resnet20_noshort": (92.9, 87.9),
         "resnet56": (96.8, 90.9), "resnet56_noshort": (81.8, 78.8)}   # train / test acc %, 40 epochs
NPARAM = {"resnet20": 272474, "resnet20_noshort": 269722, "resnet56": 855770, "resnet56_noshort": 853018}
PAPER, INK, RED = "#f3eee0", "#2a2119", "#b8452a"
FONT = "DejaVu Serif"


def load(model, tag):
    d = np.load(os.path.join(SURF, f"{model}_final_{tag}.npz"))
    return d["xs"], d["ys"], d["loss"], json.loads(str(d["meta"]))


def height(L):
    return np.log10(np.clip(L, LO, HI))


def upsample(xs, ys, Z, n):
    f = n / len(xs)
    Zu = ndimage.zoom(Z, f, order=3, mode="nearest")
    return np.linspace(xs[0], xs[-1], Zu.shape[1]), np.linspace(ys[0], ys[-1], Zu.shape[0]), Zu


def loss_levels(minor_per_decade=10):
    lv = 10 ** np.arange(-1.1, 2.0 + 1e-9, 1.0 / minor_per_decade)
    index = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    return np.log10(lv), np.log10(index)


def hillshade(Z, dx, az=300.0, alt=20.0, exag=1.0):
    gy, gx = np.gradient(Z * exag, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360.0 - az + 90.0), np.radians(alt)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


# ----------------------------------------------------------------------------- sheet furniture
def frame(ax, xs, ys, color=INK, ticks=0.25):
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(1.2)
    for t in np.arange(np.ceil(xs[0] / ticks) * ticks, xs[-1] + 1e-9, ticks):   # graticule ticks
        for yy, sgn in ((ys[0], 1), (ys[-1], -1)):
            ax.plot([t, t], [yy, yy + sgn * 0.025 * (ys[-1] - ys[0])], color=color, lw=0.6, clip_on=False)
    for t in np.arange(np.ceil(ys[0] / ticks) * ticks, ys[-1] + 1e-9, ticks):
        for xx, sgn in ((xs[0], 1), (xs[-1], -1)):
            ax.plot([xx, xx + sgn * 0.025 * (xs[-1] - xs[0])], [t, t], color=color, lw=0.6, clip_on=False)
    for t in (xs[0], 0, xs[-1]):
        ax.text(t, ys[0] - 0.045 * (ys[-1] - ys[0]), f"{t:+.1f}" if t else "0", ha="center", va="top",
                fontsize=7, color=color, family=FONT)
    for t in (ys[0], 0, ys[-1]):
        ax.text(xs[0] - 0.03 * (xs[-1] - xs[0]), t, f"{t:+.1f}" if t else "0", ha="right", va="center",
                fontsize=7, color=color, family=FONT, rotation=90)


def title_block(fig, model, sheet, meta, color=INK, extra=""):
    tr, te = TRAIN[model]
    fig.text(0.5, 0.955, "FILTER-NORMALIZED LOSS SURVEY  ·  CIFAR-10", ha="center", fontsize=9,
             color=color, family=FONT, style="italic")
    fig.text(0.5, 0.925, PRETTY[model], ha="center", fontsize=17, color=color, family=FONT)
    fig.text(0.5, 0.905, f"sheet {sheet} of 4", ha="center", fontsize=8.5, color=color, family=FONT)
    n = meta.get("n", 1000)
    lines = [f"Centre (0,0) = trained weights w*  ·  train acc {tr:.1f}%  ·  test acc {te:.1f}%",
             f"w = w* + α·δ + β·η   (δ, η: Gaussian directions, filter-normalized; BN & bias entries zero)",
             f"Heights: training cross-entropy on a fixed {n}-image subset, BN in eval mode, "
             f"{len(meta_xs(meta))}² grid",
             f"A 2-D slice of a {NPARAM[model]:,}-dimensional loss surface: non-convexity here implies it "
             f"globally; smoothness here implies little.", extra]
    for i, l in enumerate([l for l in lines if l]):
        fig.text(0.5, 0.085 - i * 0.017, l, ha="center", fontsize=6.6, color=color, family=FONT)


def meta_xs(meta):
    return range(int(meta.get("res", 51)))


# ----------------------------------------------------------------------------- styles
def survey(model, tag, sheet, out, ink=INK, paper=PAPER, second=RED, ax=None, small=False):
    xs, ys, L, meta = load(model, tag)
    X, Y, Z = upsample(xs, ys, height(L), 600)
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8.5, 10), facecolor=paper)
        ax = fig.add_axes([0.1, 0.16, 0.8, 0.8 * 8.5 / 10 * 1.0])
        ax.set_position([0.08, 0.14, 0.84, 0.84 * 8.5 / 10])
    ax.set_facecolor(paper)
    minor, index = loss_levels(10)
    ax.contour(X, Y, Z, levels=minor, colors=ink, linewidths=0.28 if not small else 0.2)
    cs = ax.contour(X, Y, Z, levels=index, colors=ink, linewidths=0.9 if not small else 0.55)
    if not small:
        ax.clabel(cs, fmt=lambda v: f"{10 ** v:g}", fontsize=6.5, inline_spacing=2)
    ax.contour(X, Y, Z, levels=[np.log10(CHANCE)], colors=second, linewidths=1.0, linestyles=[(0, (4, 2))])
    ax.plot([0], [0], marker="+", color=second, ms=9 if not small else 5, mew=1.0)
    frame(ax, xs, ys, ink)
    if own:
        title_block(fig, model, sheet, meta, ink,
                    extra="Contours every 1/10 decade of loss, index contours labelled; red dashes: chance level ln 10 = 2.30")
        fig.savefig(out, dpi=300, facecolor=paper); plt.close(fig)


def atlas(tag, out, models):
    fig = plt.figure(figsize=(12, 13), facecolor=PAPER)
    fig.text(0.5, 0.965, "FILTER-NORMALIZED LOSS SURVEY  ·  CIFAR-10  ·  ATLAS", ha="center", fontsize=11,
             family=FONT, color=INK, style="italic")
    for i, m in enumerate(models):
        r, c = divmod(i, 2)
        ax = fig.add_axes([0.07 + c * 0.47, 0.52 - r * 0.44, 0.40, 0.40])
        survey(m, tag, i + 1, None, ax=ax, small=True)
        tr, te = TRAIN[m]
        ax.set_title(f"{PRETTY[m]}   ·   train {tr:.1f}%", fontsize=10, family=FONT, color=INK, pad=14)
    fig.text(0.5, 0.045, "Shared height scale: log10 training loss clipped to [0.08, 150]; contours every 1/10 decade, "
             "bold at 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100; red dashes = chance level ln 10.", ha="center", fontsize=8,
             family=FONT, color=INK)
    fig.text(0.5, 0.027, "Each panel is a 2-D slice through ~10^5–10^6 dimensions. Crumpled slices prove non-convexity; "
             "smooth slices prove little (Li et al. 2018; Dinh et al. 2017).", ha="center", fontsize=8, family=FONT, color=INK)
    fig.savefig(out, dpi=220, facecolor=PAPER); plt.close(fig)


def hachure(model, tag, sheet, out, dark=False):
    """Lehmann-style hachures: strokes start on contour lines and run down the fall line until the
    field drops one contour interval (or a length cap); stroke weight grows with slope."""
    xs, ys, L, meta = load(model, tag)
    X, Y, Z = upsample(xs, ys, height(L), 700)
    dx = X[1] - X[0]
    gy, gx = np.gradient(Z, dx)
    ip = lambda A: RegularGridInterpolator((Y, X), A, bounds_error=False, fill_value=0.0)
    iz, igx, igy = ip(Z), ip(gx), ip(gy)
    dz = 0.1                                           # contour interval, decades
    spacing = 0.016 * (xs[-1] - xs[0]) / 2
    lv = np.arange(np.log10(LO) + dz, np.log10(HI), dz)
    cs = plt.contour(X, Y, Z, levels=lv); plt.close()
    seeds = []
    for k, segs in enumerate(cs.allsegs):
        for sg in segs:
            if len(sg) < 2:
                continue
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(sg, axis=0).T))]
            t = np.arange(np.random.default_rng(k).uniform(0, spacing), d[-1], spacing)
            seeds.append(np.c_[np.interp(t, d, sg[:, 0]), np.interp(t, d, sg[:, 1]), np.full(len(t), lv[k])])
    S = np.concatenate(seeds)
    pts = [S[:, :2].copy()]
    alive = np.ones(len(S), bool)
    step, cap = spacing * 0.25, spacing * 3.0
    trav = np.zeros(len(S))
    for _ in range(int(cap / step)):
        p = pts[-1]
        g = np.c_[igx((p[:, 1], p[:, 0])), igy((p[:, 1], p[:, 0]))]
        gn = np.hypot(*g.T) + 1e-12
        z = iz((p[:, 1], p[:, 0]))
        alive &= (z > S[:, 2] - dz) & (gn > 1e-3)
        q = p - (g / gn[:, None]) * step * alive[:, None]
        trav += step * alive
        pts.append(q)
    P3 = np.stack(pts, 1)                                # (n, steps, 2)
    slope = np.hypot(igx((S[:, 1], S[:, 0])), igy((S[:, 1], S[:, 0])))
    s98 = np.percentile(slope, 98)
    w = 0.08 + 1.1 * np.clip(slope / s98, 0, 1) ** 0.8
    keep = trav > step * 1.5
    bg, fg = ("#0e0d0c", "#e9dcc0") if dark else (PAPER, INK)
    fig = plt.figure(figsize=(8.5, 10), facecolor=bg)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.84 * 8.5 / 10])
    ax.set_facecolor(bg)
    segs = [P3[i, : int(trav[i] / step) + 1] for i in np.where(keep)[0]]
    ax.add_collection(LineCollection(segs, linewidths=w[keep], colors=fg, capstyle="round"))
    ax.plot([0], [0], marker="+", color=RED, ms=9, mew=1.0)
    frame(ax, xs, ys, fg)
    title_block(fig, model, sheet, meta, fg,
                extra="Hachures: strokes follow the fall line between contours 1/10 decade apart; weight ∝ slope of log-loss")
    fig.savefig(out, dpi=300, facecolor=bg); plt.close(fig)


def tanaka(ax, X, Y, Z, levels, gx, gy, lx=-np.sqrt(0.5), ly=np.sqrt(0.5)):
    """Tanaka illuminated contours: segment brightness = facing of the downhill normal toward the light."""
    from matplotlib.collections import LineCollection
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=0)
    dx = X[1] - X[0]; segs, cols, lws = [], [], []
    for path in [p for lvl in cs.allsegs for p in lvl]:
        if len(path) < 2:
            continue
        mid = 0.5 * (path[1:] + path[:-1])
        i = np.clip(((mid[:, 1] - Y[0]) / dx).astype(int), 0, Z.shape[0] - 1)
        j = np.clip(((mid[:, 0] - X[0]) / dx).astype(int), 0, Z.shape[1] - 1)
        g = np.stack([gx[i, j], gy[i, j]], 1); g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        f = -(g[:, 0] * lx + g[:, 1] * ly)            # +1: slope faces the light
        segs += list(np.stack([path[:-1], path[1:]], 1))
        cols += [(0.96, 0.9, 0.78, 0.15 + 0.6 * max(v, 0)) if v >= 0 else (0.0, 0.0, 0.0, 0.2 + 0.6 * -v) for v in f]
        lws += list(0.25 + 0.45 * np.abs(f))
    cs.remove()
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round"))


def raster_plate(model, tag, sheet, out, kind):
    xs, ys, L, meta = load(model, tag)
    X, Y, Z = upsample(xs, ys, height(L), 1600)
    dx = X[1] - X[0]
    hs = hillshade(Z, dx, az=300, alt=40, exag=0.35)
    hs = np.clip(hs / np.percentile(hs, 99.5), 0, 1)
    if kind == "hillshade":
        bg, fg = "#0b0a09", "#d8cdb6"
        # raking light, aspect-dominated: direction of the surface normal vs a NW light, slope saturated by tanh
        gy, gx = np.gradient(Z, dx); sl = np.hypot(gx, gy) + 1e-12
        lx, ly = -np.sqrt(0.5), np.sqrt(0.5)
        ill = (-(gx * lx + gy * ly) / sl) * np.tanh(sl / np.median(sl))
        hs2 = np.clip(0.52 + 0.42 * ill, 0, 1)
        rgb = (np.array([0.03, 0.028, 0.025]) + hs2[..., None] ** 1.4 * np.array([0.93, 0.86, 0.74]))
    elif kind == "hypsometric":
        bg, fg = PAPER, INK
        stops = ["#1f5a57", "#4f8c6a", "#9dbb7a", "#e8dc98", "#e2b26f", "#c47d4e", "#9b5a45", "#b99c93", "#e4dcd6", "#fbf8f4"]
        cm = matplotlib.colors.LinearSegmentedColormap.from_list("hyps", stops)
        band = np.floor((Z - np.log10(LO)) / (np.log10(HI / LO)) * 24) / 24      # 24 stepped tints
        rgb = cm(np.clip(band, 0, 1))[..., :3] * (0.62 + 0.38 * hs[..., None])
    elif kind in ("spectral", "hubble"):
        bg, fg = "#101014", "#d9d4c8"
        s = np.log(np.clip(ndimage.zoom(L, 1600 / len(xs), order=3, mode="nearest"), 1e-6, None)) - np.log(CHANCE)
        rgb = P.render_split(s, "sd_spectral" if kind == "spectral" else "hubble_sho", near_boundary="small")
    rgb = np.clip(rgb, 0, 1)
    fig = plt.figure(figsize=(8.5, 10), facecolor=bg)
    ax = fig.add_axes([0.08, 0.14, 0.84, 0.84 * 8.5 / 10])
    ax.imshow(rgb, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], interpolation="lanczos")
    minor, index = loss_levels(10)
    if kind == "hypsometric":
        ax.contour(X, Y, Z, levels=index, colors=INK, linewidths=0.45, alpha=0.8)
    if kind == "hillshade":
        tanaka(ax, X, Y, Z, minor, gx, gy)
    frame(ax, xs, ys, fg)
    cap = {"hillshade": "Raking light from the north-west on log10 loss (aspect shading, slope saturated at its median; declared) + Tanaka illuminated contours every 1/10 decade",
           "hypsometric": "Hypsometric tint: 24 stepped bands of log10 loss (declared palette) × hillshade; index contours",
           "spectral": "Spectral split at chance level ln 10: below (basin) purple→pale, above red→pale; rank-normalized per side",
           "hubble": "palettes.py hubble_sho split at chance level ln 10, rank-normalized per side (declared)"}[kind]
    title_block(fig, model, sheet, meta, fg, extra=cap)
    fig.savefig(out, dpi=300, facecolor=bg); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g51")
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--styles", nargs="*", default=["survey", "hachure", "hachure_dark", "hillshade",
                                                     "hypsometric", "spectral", "hubble", "atlas"])
    a = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    have = [m for m in a.models if os.path.exists(os.path.join(SURF, f"{m}_final_{a.tag}.npz"))]
    for m in have:
        sheet = ORDER.index(m) + 1
        for st in a.styles:
            out = os.path.join(GAL, f"{st}_{m}_{a.tag}.png")
            if st == "survey":
                survey(m, a.tag, sheet, out)
            elif st == "hachure":
                hachure(m, a.tag, sheet, out)
            elif st == "hachure_dark":
                hachure(m, a.tag, sheet, out, dark=True)
            elif st in ("hillshade", "hypsometric", "spectral", "hubble"):
                raster_plate(m, a.tag, sheet, out, st)
            else:
                continue
            print("wrote", out, flush=True)
    if "atlas" in a.styles and len(have) >= 2:
        atlas(a.tag, os.path.join(GAL, f"atlas_{a.tag}.png"), have)
        print("wrote atlas", flush=True)
