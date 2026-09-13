"""Render every toy piece from cache/ (no recomputation).

  python render_toy.py atlas hero iter diptych steps gamma zoom verify
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy import ndimage

from toy import mixture
from common import CACHE, GALLERY, boundary_mask, box_count, fit_dimension, write_video
from render_common import (INK, NIGHT, PAPER, P, caption_strip, colorize, flipud, grid_images, hx, layout_palette,
                           line_art, margin_confidence, confidence_shade, nu_confidence, pad_to, riso, riso_levels, save, seam_shade, text_on, upscale_nearest)

KIND_NAMES = {"ddim10": "DDIM 10", "ddim50": "DDIM 50", "ddim1000": "DDIM 1000", "ode": "PF-ODE (RK4)",
              "ddpm1000s0": "DDPM 1000, frozen noise #0", "ddpm1000s1": "DDPM frozen #1", "ddpm1000s2": "DDPM frozen #2"}
RISO_INKS = [P.RISO["fluo_pink"], P.RISO["medium_blue"], P.RISO["yellow"]]


def night(lab, pal, d0=5.0, floor=0.16):
    return colorize(lab, pal, seam_shade(lab, d0, floor), bad=hx("#000000"))


def jac_from_x0(x0, h):
    """finite-difference Jacobian of the sampler map on the grid (x0: R,R,2 ; row = y)"""
    dxa = np.gradient(x0[..., 0], h, axis=1); dxb = np.gradient(x0[..., 0], h, axis=0)
    dya = np.gradient(x0[..., 1], h, axis=1); dyb = np.gradient(x0[..., 1], h, axis=0)
    det = dxa * dyb - dxb * dya
    tr = dxa ** 2 + dxb ** 2 + dya ** 2 + dyb ** 2
    smax = np.sqrt(0.5 * (tr + np.sqrt(np.maximum(tr ** 2 - 4 * det ** 2, 0))))
    return det, smax


def split_orientation(sign, logabsdet):
    """Sohl-Dickstein Spectral split of the signed Jacobian determinant: purple half = orientation
    preserved (det > 0), red half = reversed (det < 0); per-side rank of log|det|, small |det| (the fold
    curves, where the map is not locally invertible) at the dark seam."""
    x = np.where(sign > 0, 1.0, -1.0) * (logabsdet - np.nanmin(logabsdet) + 1e-6)
    return P.render_split(-x, "sd_spectral", near_boundary="small")  # neg (purple) side = preserved


# ----------------------------------------------------------------------------- atlas
def atlas():
    for score in ["analytic", "learned"]:
        tiles_paper, tiles_night, tiles_conf = [], [], []
        rows = ["scatter12", "ring8", "grid25"]
        kinds = ["ddim10", "ddim50", "ddim1000", "ode", "ddpm1000s0"]
        for lay in rows:
            f = f"{CACHE}/maps_{lay}_{score}.npz"
            if not os.path.exists(f):
                print("missing", f); return
            z = np.load(f)
            pal = layout_palette(lay)
            for k in kinds:
                lab, x0 = z[f"{k}_lab"], z[f"{k}_x0"]
                if lab.shape[0] > 512:
                    lab, x0 = lab[::2, ::2], x0[::2, ::2]
                lab, x0 = flipud(lab), flipud(x0)
                tiles_conf.append(colorize(lab, pal, confidence_shade(margin_confidence(x0, mixture(lay)[0]))))
                tiles_paper.append(line_art(lab, pal=pal, tint_strength=0.55, width=0.9))
                tiles_night.append(night(lab, pal, d0=4.0))
        cap = [f"Basin atlas, {score} score. Rows: scatter12, ring8, grid25 Gaussian mixtures. Columns: DDIM-10, DDIM-50, DDIM-1000, "
               f"probability-flow ODE (RK4), DDPM-1000 with one frozen noise sequence.",
               "Each pixel = a starting noise z in [-3,3]^2, coloured by the mixture mode its sample lands nearest. "
               "Measured: labels. Declared: palettes (ring: hue = mode angle; grid: hue = column, lightness = row), seam shading."]
        g = grid_images(tiles_paper, 5, 14, PAPER)
        save(caption_strip(g, cap, size=20), f"atlas_{score}_paper.png", "toy")
        g = grid_images(tiles_night, 5, 14, NIGHT)
        save(caption_strip(g, cap, ground=NIGHT, ink="#d8d4c8", size=20), f"atlas_{score}_night.png", "toy")
        capc = cap[:1] + ["Brightness = measured commitment margin 1 - d1/d2 of the generated sample (dark = landed between two modes). "
                          "Hue = mode (declared palette)."]
        g = grid_images(tiles_conf, 5, 14, NIGHT)
        save(caption_strip(g, capc, ground=NIGHT, ink="#d8d4c8", size=20), f"atlas_{score}_confidence.png", "toy")


# ----------------------------------------------------------------------------- ODE hero
def hero():
    v = np.load(f"{CACHE}/verify_2048.npz")
    lab = flipud(v["ode_scatter12"])
    pal = layout_palette("scatter12")
    save(night(lab, pal, d0=10, floor=0.12), "hero_ode_scatter12_night.png", "toy")
    zc = np.load(f"{CACHE}/maps_scatter12_analytic.npz")
    for kind in ["ode", "ddim10", "ddpm1000s0"]:
        x0 = flipud(zc[f"{kind}_x0"])
        save(colorize(flipud(zc[f"{kind}_lab"]), pal, confidence_shade(margin_confidence(x0, mixture("scatter12")[0]))),
             f"hero_{kind}_scatter12_confidence.png", "toy")
    save(line_art(lab, width=1.2, pal=pal, tint_strength=0.0), "hero_ode_scatter12_ink.png", "toy")
    save(riso(lab, riso_levels(12), RISO_INKS, cell=9), "hero_ode_scatter12_riso.png", "toy")
    # stretching plate: largest singular value of the sampler Jacobian (finite differences, 1024^2)
    z = np.load(f"{CACHE}/maps_scatter12_analytic.npz")
    for kind in ["ode", "ddim10"]:
        x0 = z[f"{kind}_x0"].astype(np.float64)
        h = 2 * float(z["hw"]) / x0.shape[0]
        det, smax = jac_from_x0(x0, h)
        v_ = np.log10(np.maximum(smax, 1e-6))
        lo, hi = np.percentile(v_, [1, 99.7])
        cm = P.get("cet_fire") if "cet_fire" in P.mpl.colormaps else P.mpl.colormaps["inferno"]
        img = cm(np.clip((flipud(v_) - lo) / (hi - lo), 0, 1))[..., :3]
        save(img, f"stretch_{kind}_scatter12_fire.png", "toy")
        np.save(f"{CACHE}/stretch_range_{kind}.npy", np.array([lo, hi, float((det < 0).mean())]))
        print(kind, "log10 smax range", lo, hi, "fraction det<0 (finite diff)", (det < 0).mean())


# ----------------------------------------------------------------------------- iterated map
def iter_pieces():
    for lay in ["ring8", "ring6", "scatter12"]:
        f = f"{CACHE}/iter_{lay}_hero.npz"
        if not os.path.exists(f):
            print("missing", f); continue
        z = np.load(f)
        lab, pal = flipud(z["lab"]), layout_palette(lay)
        d0 = 8 if lab.shape[0] >= 2048 else 5
        save(night(lab, pal, d0=d0, floor=0.14), f"iter_{lay}_night.png", "iterated")
        save(colorize(lab, pal, confidence_shade(nu_confidence(flipud(z["nu"])), floor=0.06, gamma=1.4)),
             f"iter_{lay}_confidence.png", "iterated")
        save(split_orientation(flipud(z["sign"]), flipud(z["logdet"])), f"iter_{lay}_spectral_orientation.png", "iterated")
        if lay == "ring8":
            save(line_art(lab, width=1.1), f"iter_{lay}_ink.png", "iterated")
            save(riso(lab, riso_levels(8, seed=3), RISO_INKS, cell=8), f"iter_{lay}_riso.png", "iterated")
            nu = flipud(z["nu"])
            # convergence-time plate, split by orientation sign is not used here: sequential lajolla on log(nu)
            vv = np.log(nu)
            lo, hi = np.percentile(vv[np.isfinite(vv)], [0.5, 99.8])
            img = P.mpl.colormaps["cmc.lajolla"](np.clip((vv - lo) / (hi - lo), 0, 1))[..., :3]
            save(img, f"iter_{lay}_convergence_lajolla.png", "iterated")


def diptych():
    zo = np.load(f"{CACHE}/maps_ring8_analytic.npz")
    zi = np.load(f"{CACHE}/iter_ring8_hero.npz")
    pal = layout_palette("ring8")
    L = flipud(zo["ode_lab"])
    Ri = flipud(zi["lab"])[::2, ::2]
    gap = 24
    for style in ["night", "spectral"]:
        if style == "night":
            a, b = night(L, pal, d0=5), night(Ri, pal, d0=5)
            ground, ink = NIGHT, "#d8d4c8"
        else:
            x0 = zo["ode_x0"].astype(np.float64)
            det, _ = jac_from_x0(x0, 2 * float(zo["hw"]) / x0.shape[0])
            a = split_orientation(np.sign(flipud(det)), np.log(np.abs(flipud(det)) + 1e-300))
            b = split_orientation(flipud(zi["sign"])[::2, ::2], flipud(zi["logdet"])[::2, ::2])
            ground, ink = PAPER, INK
        img = np.ones((1024 + 2 * gap, 2048 + 3 * gap, 3)) * hx(ground)
        img[gap:gap + 1024, gap:gap + 1024] = a
        img[gap:gap + 1024, 2 * gap + 1024:2 * gap + 2048] = b
        cap = ["LEFT  probability-flow ODE sampler (a composition of smooth invertible maps): noise z in [-3,3]^2 -> which of 8 ring modes. "
               "RIGHT  a different map: over-relaxed denoiser x <- x + 2.12 (D(x, sigma=0.4) - x), iterated 2000x, start x in [-4,4]^2.",
               ("Colour: which mode (declared cyclic palette), dark seams at basin boundaries." if style == "night" else
                "Colour: Spectral split of det J (purple = orientation kept, red = flipped), rank of log|det J| per side; seams = fold curves. "
                "Left: finite differences. Right: exact chain rule.")]
        save(caption_strip(img, cap, ground=ground, ink=ink, size=19), f"diptych_ode_vs_iterated_{style}.png", "diptych")


# ----------------------------------------------------------------------------- animations
def frames_to_video(frames_fn, n, name, sub, fps=24, gif_fps=12, gif_width=540):
    tmp = tempfile.mkdtemp(prefix="db_", dir=CACHE)
    for i in range(n):
        save(frames_fn(i), os.path.join(tmp, f"f{i:05d}.png"))
    os.makedirs(os.path.join(GALLERY, sub), exist_ok=True)
    write_video(tmp, "f%05d.png", os.path.join(GALLERY, sub, name + ".mp4"), os.path.join(GALLERY, sub, name + ".gif"),
                fps=fps, gif_fps=gif_fps, gif_width=gif_width)
    shutil.rmtree(tmp)


def steps_anim():
    z = np.load(f"{CACHE}/steps_scatter12_analytic.npz")
    steps, labs, x0s = z["steps"], z["lab"], z["x0"]
    from toy import mixture
    mu, _, s = mixture("scatter12")
    pal = layout_palette("scatter12")
    reps = np.clip(np.round(np.interp(np.arange(len(steps)), [0, len(steps) - 1], [14, 6])).astype(int), 1, None)
    seq = np.repeat(np.arange(len(steps)), reps)
    seq = np.concatenate([seq, np.full(48, len(steps) - 1)])

    def frame(i):
        j = seq[i]
        lab = flipud(labs[j])
        # brightness: how close the sample is to its mode centre (sharp sample = bright); declared exp(-d/3s)
        d = np.linalg.norm(flipud(x0s[j]) - mu[flipud(labs[j])], axis=-1)
        conf = 0.12 + 0.88 * np.exp(-d / (3 * s))
        img = colorize(lab, pal, conf * seam_shade(lab, 3.0, 0.35))
        fr = pad_to(img, 1080, 1080, NIGHT)
        fr = text_on(fr, (156, 40), f"DDIM steps  N = {steps[j]:4d}", size=34, fill=hx("#e8e4d8"))
        fr = text_on(fr, (156, 1000), "brightness = exp(-|x0 - mode| / 3s)   scatter12, exact score", size=22, fill=hx("#9d998e"))
        return fr
    frames_to_video(frame, len(seq), "steps_scatter12", "animations")


def gamma_anim():
    z = np.load(f"{CACHE}/iter_ring8_gamma.npz")
    gam, labs, sg, ld = z["gammas"], z["lab"], z["sign"], z["logdet"]
    pal = layout_palette("ring8")
    seq = np.concatenate([np.repeat(np.arange(len(gam)), 5), np.full(60, len(gam) - 1)])

    def frame(i):
        j = seq[i]
        lab = flipud(labs[j])
        a = upscale_nearest(night(lab, pal, d0=3.0), 2)
        fr = text_on(a, (30, 24), f"gamma = {gam[j]:.4f}", size=30, fill=hx("#f0ece0"))
        return fr

    def frame_split(i):
        j = seq[i]
        a = upscale_nearest(split_orientation(flipud(sg[j]), flipud(ld[j])), 2)
        return text_on(a, (30, 24), f"gamma = {gam[j]:.4f}", size=30, fill=hx("#1b1b24"))
    frames_to_video(frame, len(seq), "gamma_sweep_ring8_night", "animations")
    frames_to_video(frame_split, len(seq), "gamma_sweep_ring8_spectral", "animations")


def zoom_composite(labs, hw0, kappa, F=1080):
    K, R = labs.shape[0], labs.shape[1]
    k0 = int(np.floor(kappa))
    hwf = hw0 * 2.0 ** (-kappa)
    o = hwf * ((np.arange(F) + 0.5) / F * 2 - 1)
    out = None
    for k in range(k0, min(k0 + 3, K)):
        hwk = hw0 * 2.0 ** (-k)
        idx = np.floor((o / hwk + 1) / 2 * R).astype(int)
        inside = (idx >= 0) & (idx < R)
        idc = np.clip(idx, 0, R - 1)
        L = labs[k][np.ix_(idc, idc)]
        m = np.outer(inside, inside)
        out = L if out is None else np.where(m, L, out)
    return out


def zoom_anim(tag, frames_per_level=36):
    z = np.load(f"{CACHE}/zoom_{tag}.npz")
    labs, hw0 = z["lab"], float(z["hw0"])
    spec = json.loads(str(z["spec"]))
    pal = layout_palette(spec["layout"])
    K = labs.shape[0]
    kap = np.linspace(0, K - 1.001, int((K - 1) * frames_per_level))
    kap = np.concatenate([kap, np.full(48, kap[-1])])

    def frame(i):
        lab = flipud(zoom_composite(labs, hw0, kap[i]))
        img = night(lab, pal, d0=4.0, floor=0.16)
        w = 2 * hw0 * 2.0 ** (-kap[i])
        return text_on(img, (24, 1040), f"window width {w:.2e}", size=24, fill=hx("#e8e4d8"))
    frames_to_video(frame, len(kap), f"zoom_{tag}", "zooms")
    # a plate: 6 levels side by side
    picks = np.linspace(0, K - 1, 6).round().astype(int)
    tiles = []
    for k in picks:
        lab = flipud(labs[k])
        t = night(lab, pal, d0=3.0)
        tiles.append(text_on(t, (12, lab.shape[0] - 34), f"width {2*hw0*2.0**-k:.1e}", size=22, fill=hx("#e8e4d8")))
    save(grid_images(tiles, 3, 12, NIGHT), f"zoom_{tag}_plate.png", "zooms")


# ----------------------------------------------------------------------------- verification plots
def windowed_dimension(labs, lo=2, hi=128):
    out = []
    for L in labs:
        b = boundary_mask(L)
        if b.sum() < 10:
            out.append((np.nan, 0)); continue
        s, c = box_count(b)
        out.append((fit_dimension(s, c, lo, hi)[0], int(b.sum())))
    return np.array(out)


def verify_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = json.load(open(f"{CACHE}/verify_toy.json"))
    res = {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    cols = {"null_circle": "#8a8a84", "null_ring8_rays": "#b5b3ab", "ode_scatter12": "#2a78d6", "ddim50_scatter12": "#1baf7a",
            "ddim10_scatter12": "#eda100", "iter_ring8": "#e34948", "iter_ring6": "#e87ba4", "iter_ring8_gamma1": "#4a3aa7",
            "newton_z3": "#0b0b0b"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    ax = axs[0]
    for k, d in V["box2048"].items():
        s, c = np.array(d["sizes"]), np.array(d["counts"])
        ax.loglog(1 / s, c, "-o", ms=3, lw=1.6, color=cols.get(k, "#000"), label=f"{k}: D={d['D_2_256']:.2f}")
    ax.set_xlabel("1 / box size (px$^{-1}$), 2048$^2$ image"); ax.set_ylabel("occupied boxes N")
    ax.set_title("Box counting of basin boundaries (fit 2-256 px)", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1]
    for k, d in V["uncertainty"].items():
        e, f = np.array(d["eps"]), np.array(d["f"])
        ok = f > 0
        ax.loglog(e[ok], f[ok], "-o", ms=3, lw=1.6, color=cols.get(k, "#000"), label=f"{k}: alpha={d['alpha']:.2f} (D={d['D']:.2f})")
    ax.set_xlabel("perturbation eps"); ax.set_ylabel("fraction of eps-uncertain points f")
    ax.set_title("Uncertainty exponent  f ~ eps^alpha,  D = 2 - alpha", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[2]
    for tag, c in [("ode_scatter12", "#2a78d6"), ("iter_ring8", "#e34948"), ("iter_ring6", "#e87ba4"), ("ddim50_learned_scatter12", "#1baf7a")]:
        f = f"{CACHE}/zoom_{tag}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        wd = windowed_dimension(z["lab"])
        w = 2 * float(z["hw0"]) * 2.0 ** (-np.arange(len(wd)))
        res[tag] = dict(width=w.tolist(), D=wd[:, 0].tolist(), boundary_px=wd[:, 1].tolist())
        ax.semilogx(w, wd[:, 0], "-o", ms=3, lw=1.6, color=c, label=tag)
    ax.axhline(1, color="#b5b3ab", lw=1, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("zoom window width (plane units)"); ax.set_ylabel("box-count D inside window (2-128 px)")
    ax.set_title("Scale-resolved dimension along the zoom", loc="left")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    os.makedirs(f"{GALLERY}/verify", exist_ok=True)
    plt.savefig(f"{GALLERY}/verify/toy_dimension.png", dpi=150)
    json.dump(res, open(f"{CACHE}/zoom_dimensions.json", "w"), indent=1)
    print(json.dumps(res, indent=0)[:3000])


if __name__ == "__main__":
    fns = dict(atlas=atlas, hero=hero, iter=iter_pieces, diptych=diptych, steps=steps_anim, gamma=gamma_anim,
               verify=verify_plots)
    for a in sys.argv[1:]:
        if a.startswith("zoom:"):
            zoom_anim(a[5:])
        else:
            fns[a]()
