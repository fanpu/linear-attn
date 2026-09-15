"""Ribbon renders (M2): the edge-of-stability trajectory as strands + chords inside its loss canyon.

Reads only cache/ (prep_m2.py, canyon_box.py outputs). Every plate declares:
  chart     hero/honesty/stereo/plotter: fixed frame (u_ref, pc1, pc2) around theta_ref (Stage B, t_ref 3250);
            context: piecewise oscillation frames (prep_m2.py docstring). Orthographic cameras.
  scaling   world = chart offset x per-axis exaggeration factor (FACT), then one common scale.
  marks     strands = even / odd steps as sphere-splatted tubes (r3d.splat_spheres); chords = every GD step
            as a straight segment theta_t -> theta_{t+1}, drawn as additive Gaussian hairlines
            (r3d.splat_additive); no fitted surface.
  colour    measured lambda_1 * eta / 2 of the step, Spectral_r with a two-slope norm 0.90 | 1.00 | 1.12
            (declared diverging map centred at the edge).
  canyon    measured loss on the 32^3 grid, trilinear interpolation (declared; 64^3 check plate), transfer
            function TF_DOC, half cutaway: the half pc2 < pc2_cut is removed.
  light     Lambert shading on the strands shows form only.
Usage: render_ribbon.py <plate> [--size S] [--device cpu|cuda] [--out-suffix X]
  plates: hero, honesty, stereo, plotter, context, check64
"""
import argparse
import json
import os
import sys
import time

import matplotlib
import cmcrameri.cm  # noqa: F401  (registers cmc.* colormaps)
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(HERE, "cache", "m2")
GAL = os.path.join(HERE, "gallery")
os.makedirs(GAL, exist_ok=True)
# Module defaults (overridden by _parse_cli() when run as a script, or by importers such as render_film.py,
# which sets rr.DEV / rr.S / rr.DT directly to reuse these functions without going through argparse).
DEV = "cpu"
S = 512
DT = torch.float64
args = argparse.Namespace(plate=None, size=S, device=DEV, out_suffix="")


def _parse_cli():
    global DEV, S, DT, args
    p = argparse.ArgumentParser()
    p.add_argument("plate")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()
    torch.set_num_threads(4)
    DEV = args.device
    S = args.size
    DT = torch.float32 if DEV == "cuda" else torch.float64
NORM = TwoSlopeNorm(vmin=0.90, vcenter=1.0, vmax=1.12)
CMAP = matplotlib.colormaps["Spectral_r"]
BG = np.array([0.018, 0.020, 0.028])
SHELLS = np.linspace(0.08, 0.88, 6)
OSLO_UPPER = matplotlib.colors.ListedColormap(matplotlib.colormaps["cmc.oslo"](np.linspace(0.35, 1.0, 256)))
TF_DOC = ("upper 65 % of cmc.oslo (0.35-1.0) over the grid's loss range; opacity = exp(-x / 0.10), x = normalised "
          "loss (0 at the measured minimum), so opacity is highest at the valley floor and fades with increasing "
          "loss (M3: replaces M2's 6 equal shells, which read as flat sheets), density D")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def lam_rgb(lam):
    return CMAP(NORM(lam))[:, :3]


def load_hero(kind="fixed"):
    h = np.load(os.path.join(M2, "hero.npz"))
    return h[kind], h["lam"], h["t"]


class Chart:
    """chart offsets (a = axis 1, b = pc1, c = pc2) -> world (x = b, y = c, z = a), exaggerated and centred."""

    def __init__(self, lo, hi, fact):
        self.fact = np.asarray(fact, float)  # factors for (a, b, c)
        self.lo_c, self.hi_c = np.asarray(lo, float), np.asarray(hi, float)
        wl, wh = self._raw(self.lo_c[None])[0], self._raw(self.hi_c[None])[0]
        self.centre = 0.5 * (wl + wh)
        self.scale = 2.0 / np.max(wh - wl)
        self.lo_w, self.hi_w = self(self.lo_c[None])[0], self(self.hi_c[None])[0]

    def _raw(self, C):
        return np.stack([C[:, 1] * self.fact[1], C[:, 2] * self.fact[2], C[:, 0] * self.fact[0]], 1)

    def __call__(self, C):
        return (self._raw(np.asarray(C, float)) - self.centre) * self.scale


def tt(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=DT, device=DEV)


def ribbon_layers(Pw, lam, cam, R, clip_y=None):
    """Opaque strands image + depth, and chord glow split into behind / in front of the cut plane."""
    even, odd = Pw[0::2], Pw[1::2]
    spacing = R * 0.5
    cols, pts = [], []
    for par, Q in ((0, even), (1, odd)):
        q, s = r3d.sample_polyline(tt(Q), spacing)
        idx = np.clip(np.rint(s.cpu().numpy()).astype(int) * 2 + par, 0, len(lam) - 1)
        pts.append(q)
        cols.append(lam_rgb(lam[idx]))
    pts = torch.cat(pts)
    col = tt(np.concatenate(cols))
    sp = r3d.splat_spheres(pts, R, cam, attrs=col)
    m = sp["mask"]
    img = tt(np.broadcast_to(BG, (S, S, 3)).copy()) if cam.width == S else None
    img = tt(np.broadcast_to(BG, (cam.height, cam.width, 3)).copy())
    shade = 0.45 + 0.55 * r3d.lambert(sp["normal"][m], (-0.35, -0.8, 0.9), ambient=0.0)
    img[m] = sp["attr"][m] * shade[:, None]
    cs = 0.25 * R
    q, s = r3d.sample_polyline(tt(Pw), cs)
    step = np.clip(np.floor(s.cpu().numpy()).astype(int), 0, len(lam) - 1)
    ccol = tt(lam_rgb(lam[step]))
    wgt = cs / cam.pixel_scale()
    glow = {}
    front = torch.ones(len(q), dtype=torch.bool, device=DEV) if clip_y is None else q[:, 1] < clip_y
    for name, sel in (("front", front), ("back", ~front)):
        if sel.sum() == 0:
            glow[name] = torch.zeros_like(img)
            continue
        g = r3d.splat_additive(q[sel], cam, color=ccol[sel], weight=wgt, sigma_px=max(0.7, S / 1400),
                               depth=sp["depth"], eps=3 * R)
        glow[name] = g
    return img, sp["depth"], glow


FLOOR_TAU = 0.10  # M3 decision: opacity = exp(-x / FLOOR_TAU), tuned against 0.14/0.22/0.30 on the hero window


def opacity_floor(x):
    """M3 canyon TF: emphasise the valley floor. Highest opacity at the measured loss minimum (x=0),
    fading smoothly with increasing loss -- a monotone re-weighting of the same scalar field, not a new
    shape (declared in every canyon caption). Replaces M2's 6 equal Gaussian shells, which read as flat
    stacked sheets because loss is nearly quadratic along u_ref alone."""
    return torch.exp(-x / FLOOR_TAU)


_CANYON_CACHE = {}


def canyon_layer(name, chart, cam, depth, clip_y, density, opacity=None, step_div=400):
    if name not in _CANYON_CACHE:
        _CANYON_CACHE[name] = np.load(os.path.join(M2, f"canyon_{name}.npz"))["loss"].astype(np.float64)
    L = _CANYON_CACHE[name]  # [a, b, c]
    data = tt(np.transpose(L, (0, 2, 1)))  # data[k=a, j=c, i=b] -> world (x=b, y=c, z=a)
    lo, hi = chart.lo_w, chart.hi_w
    vmin, vmax = float(L.min()), float(L.max())
    opacity = opacity_floor if opacity is None else opacity
    span = vmax - vmin
    tf = r3d.TransferFunction(OSLO_UPPER, vmin, vmax, opacity, density=density)
    clip = [] if clip_y is None else [((0.0, clip_y, 0.0), (0.0, -1.0, 0.0))]
    step = float(np.min(np.asarray(hi) - np.asarray(lo))) / step_div
    rgb, a = r3d.render_volume(data, tuple(lo), tuple(hi), cam, tf, step, depth=depth, clip=clip, device=DEV)
    return rgb, a, {"vmin": vmin, "vmax": vmax, "step": step, "density": density}


def composite(img, glow, vol, expo):
    tm = lambda g: r3d.glow_tonemap(g, expo)  # noqa: E731
    back = (img + tm(glow["back"])).clamp(0, 1)
    if vol is not None:
        back = r3d.over(vol[0].to(back.dtype), vol[1].to(back.dtype), back)
    return (back + tm(glow["front"])).clamp(0, 1)


def annotate(arr, lines, scale=1.0):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    fs = max(11, int(im.width / 90 * scale))
    f = ImageFont.truetype(FONT, fs)
    y = im.height - (len(lines) + 0.6) * fs * 1.35
    for ln in lines:
        d.text((fs, y), ln, fill=(200, 200, 205), font=f)
        y += fs * 1.35
    return np.asarray(im).astype(np.float64) / 255


def camera(chart, az, el, w=None, h=None, zoom=1.0):
    w = S if w is None else w
    h = S if h is None else h
    ext = chart.hi_w - chart.lo_w
    return r3d.orbit((0, 0, 0), 8.0, az, el, width=w, height=h, ortho_height=1.35 * max(ext) / zoom * h / w)


def save(name, arr):
    path = os.path.join(GAL, name + args.out_suffix + ".png")
    r3d.save_png(path, torch.as_tensor(arr))
    print("wrote", path, flush=True)


INFO = json.load(open(os.path.join(M2, "prep_info.json")))
AZ, EL = float(os.environ.get("RIB_AZ", -120)), float(os.environ.get("RIB_EL", 38))
HERO_FACT = (4.0, 1.0, 4.0)  # (u_ref, pc1, pc2)
CTX_FACT = (20.0, 1.0, 2.0)
t0 = time.time()
meta = {}


def crop_to_content(arr, bg, pad_top_frac=0.06, pad_bot_min_frac=0.07, eps=0.03):
    """Crop rows to the rendered content's bounding box (canyon + ribbon), with a small top margin and enough
    bottom margin for the caption. Pure re-framing of the same render: camera, data and scalings are untouched."""
    diff = np.abs(arr[..., :3] - np.asarray(bg)).sum(-1)
    rows = np.where(diff.max(1) > eps)[0]
    if len(rows) == 0:
        return arr
    r0, r1 = int(rows.min()), int(rows.max())
    h = r1 - r0
    top_pad = max(int(pad_top_frac * h), 1)
    bot_pad = max(int(pad_bot_min_frac * h), 1)
    lo, hi = max(0, r0 - top_pad), min(arr.shape[0], r1 + bot_pad)
    return arr[lo:hi]


def hero_render(kind, cam, canyon="hero32", label=True, R=0.006, density=None, crop=False):
    C, lam, t = load_hero(kind)
    bx = json.load(open(os.path.join(M2, "boxes.json")))[canyon]
    chart = Chart(bx["lo"], bx["hi"], HERO_FACT)
    Pw = chart(C)
    y_cut = float(np.median(Pw[:, 1]))
    img, depth, glow = ribbon_layers(Pw, lam, cam, R, clip_y=y_cut)
    dens = density if density is not None else float(os.environ.get("RIB_DENS", 6.0)) / (chart.hi_w - chart.lo_w).max()
    vol = canyon_layer(canyon, chart, cam, depth, y_cut, dens)
    out = composite(img, glow, vol[:2], expo=1.3).cpu().numpy()
    meta.update(vol[2])
    meta["pc2_cut_chart"] = float(np.median(C[:, 2]))
    if crop:
        out = crop_to_content(out, BG)
    if label:
        lines = [f"steps {t[0]}–{t[-1]} · Stage B, 2/η = 80 · orthographic · u_ref ×4, pc2 ×4 relative to pc1",
                 "colour λ₁η/2 (Spectral_r, 0.90 | 1.00 | 1.12) · canyon: measured loss, 32³, half cut at pc2 = window median",
                 "canyon opacity emphasises the valley floor: exp(−loss_norm / 0.10), highest at the measured minimum"]
        out = annotate(out, lines)
    return out


if __name__ == "__main__":
    _parse_cli()
    t0 = time.time()

    if args.plate == "hero":
        cam = camera(Chart(**{k: json.load(open(os.path.join(M2, "boxes.json")))["hero32"][k] for k in ("lo", "hi")},
                           fact=HERO_FACT), az=AZ, el=EL)
        save("hero_glow", hero_render("fixed", cam, crop=True))
    elif args.plate == "check64":
        bx = json.load(open(os.path.join(M2, "boxes.json")))["hero32"]
        cam = camera(Chart(bx["lo"], bx["hi"], HERO_FACT), az=AZ, el=EL)
        a32 = hero_render("fixed", cam, "hero32", label=False)
        a64 = hero_render("fixed", cam, "hero64", label=False)
        both = np.concatenate([a32, a64], 1)
        save("check_canyon_32_vs_64", annotate(both, ["left: 32³ grid · right: 64³ grid (same box, same TF, trilinear)"], 0.5))
    elif args.plate == "honesty":
        W = S
        bx = json.load(open(os.path.join(M2, "boxes.json")))["hero32"]
        cam = camera(Chart(bx["lo"], bx["hi"], HERO_FACT), az=AZ, el=EL)
        A = hero_render("fixed", cam, label=False)
        B = hero_render("moving", cam, label=False)
        A = annotate(A, ["fixed frame: flip measured along u_ref (bank u₁ at step 3250)"], 0.9)
        B = annotate(B, ["moving frame: flip measured along the current u₁(t)"], 0.9)
        save("honesty_fixed_vs_moving", np.concatenate([A, B], 1))
    elif args.plate == "stereo":
        bx = json.load(open(os.path.join(M2, "boxes.json")))["hero32"]
        ch = Chart(bx["lo"], bx["hi"], HERO_FACT)
        L_ = hero_render("fixed", camera(ch, az=AZ - 2.5, el=EL), label=False)
        R_ = hero_render("fixed", camera(ch, az=AZ + 2.5, el=EL), label=False)
        save("stereo_crosseye", annotate(np.concatenate([R_, L_], 1), ["cross-eye pair (right-eye view on the left) · "
                                                                      "rotation stereo ±2.5° · u_ref ×4, pc2 ×4"], 0.5))
    elif args.plate == "context":
        c = np.load(os.path.join(M2, "context.npz"))
        C, lam, t = c["coords"], c["lam"], c["t"]
        bx = json.load(open(os.path.join(M2, "boxes.json")))["context32"]
        chart = Chart(bx["lo"], bx["hi"], CTX_FACT)
        H = int(S * 0.62)
        cam = camera(chart, az=-100, el=16, w=S, h=H, zoom=1.0)
        Pw = chart(C)
        y_cut = float(np.median(Pw[:, 1]))
        img, depth, glow = ribbon_layers(Pw, lam, cam, R=0.0018, clip_y=y_cut)  # M3: thinner tubes so chords show
        vol = canyon_layer("context32", chart, cam, depth, y_cut, 1.2)
        out = composite(img, glow, vol[:2], expo=0.6).cpu().numpy()
        out = annotate(out, [f"steps {t[0]}–{t[-1]} (EoS phase, Stage B) · axis 1 = piecewise oscillation frames (×20), "
                             "pc1 ×1, pc2 ×2 · orthographic · tubes thinned (R 0.0018) so chords show between strands",
                             "canyon: loss on the fixed (u_ref, pc1, pc2) slice through θ(3250), 32³, valley-floor "
                             "opacity · a fixed axis 1 only holds locally (median adjacent-frame overlap 0.39)"], 0.8)
        save("context_eos_glow", out)
        meta.update(vol[2])
    elif args.plate == "plotter":
        from skimage import measure
        C, lam, t = load_hero("fixed")
        bx = json.load(open(os.path.join(M2, "boxes.json")))["hero32"]
        chart = Chart(bx["lo"], bx["hi"], HERO_FACT)
        cam = camera(chart, az=AZ, el=EL)
        Pw = chart(C)
        R = 0.004
        q, _ = r3d.sample_polyline(tt(Pw), R * 0.5)
        depth = r3d.splat_spheres(q, R, cam)["depth"]
        path_runs = r3d.visible_runs(tt(Pw), cam, depth, 3 * R)
        g = np.load(os.path.join(M2, "canyon_hero32.npz"))
        L = g["loss"]  # [a, b, c]
        back = L[:, :, -1]  # far face pc2 = hi
        levels = L.min() + (L.max() - L.min()) * SHELLS
        cont = []
        for lv in levels:
            for cc in measure.find_contours(back, lv):  # rows = a index, cols = b index
                ai = np.interp(cc[:, 0], np.arange(len(g["a"])), g["a"])
                bi = np.interp(cc[:, 1], np.arange(len(g["b"])), g["b"])
                Cc = np.stack([ai, bi, np.full(len(ai), g["c"][-1])], 1)
                cont += r3d.visible_runs(tt(chart(Cc)), cam, depth, 3 * R)
        lo, hi = chart.lo_w, chart.hi_w
        corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        box = []
        for i, j in edges:
            seg = np.linspace(corners[i], corners[j], 50)
            box += r3d.visible_runs(tt(seg), cam, depth, 3 * R)
        svg = os.path.join(GAL, "plotter_hero.svg")
        tmp = [os.path.join(M2, f"_pen{i}.svg") for i in range(3)]
        r3d.write_svg(tmp[0], [r.cpu().numpy() if torch.is_tensor(r) else r for r in cont], S, S, stroke="#2b5d8a", stroke_width=0.8)
        r3d.write_svg(tmp[1], [r.cpu().numpy() if torch.is_tensor(r) else r for r in box], S, S, stroke="#9a948a", stroke_width=0.6)
        r3d.write_svg(tmp[2], [r.cpu().numpy() if torch.is_tensor(r) else r for r in path_runs], S, S, stroke="#1f1d1b", stroke_width=0.7)
        # merge the three pens into one layered SVG
        bodies = []
        for i, name in enumerate(["canyon contours (far face)", "box", "GD path"]):
            s = open(tmp[i]).read()
            inner = s[s.index(">", s.index("<svg")) + 1:s.rindex("</svg>")]
            bodies.append(f'<g inkscape:groupmode="layer" inkscape:label="{name}">{inner}</g>')
            os.remove(tmp[i])
        open(svg, "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
                             f'width="{S}" height="{S}" viewBox="0 0 {S} {S}"><rect width="100%" height="100%" fill="#f3efe6"/>'
                             + "".join(bodies) + "</svg>")
        print("wrote", svg, "runs", len(path_runs), len(cont), len(box))
    print(json.dumps(meta), f"{time.time() - t0:.1f}s", flush=True)
