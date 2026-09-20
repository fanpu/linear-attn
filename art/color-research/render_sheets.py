"""Contact sheets: the same REAL data under many colour schemes (data read-only from other projects).

    python render_sheets.py all            # or: split_train split_basin split_lyap sequential spectrogram
                                           #     diverging cyclic categorical riso profiles heroes
Outputs lossless PNG in gallery/.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl
from scipy import ndimage

import palettes as P
import sources as S

GAL = "/home/fzeng/ml/research/art/color-research/gallery/"
FONTDIR = mpl.get_data_path() + "/fonts/ttf/"
F_TITLE = ImageFont.truetype(FONTDIR + "DejaVuSerif-Bold.ttf", 40)
F_SUB = ImageFont.truetype(FONTDIR + "DejaVuSans.ttf", 20)
F_LAB = ImageFont.truetype(FONTDIR + "DejaVuSans-Bold.ttf", 19)
F_SMALL = ImageFont.truetype(FONTDIR + "DejaVuSans.ttf", 15)
BG = (238, 234, 227)
INK = (28, 28, 32)
GREY = (105, 100, 95)


def u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def resize(img, size):
    """RGB float -> uint8 PIL image resized (Lanczos) to `size` (w, h)."""
    im = Image.fromarray(u8(img))
    return im.resize(size, Image.LANCZOS) if im.size != tuple(size) else im


def strip(cmap_or_cols, w, h=14):
    if isinstance(cmap_or_cols, np.ndarray):
        cols = cmap_or_cols
    else:
        cm = P.as_cmap(cmap_or_cols) if not isinstance(cmap_or_cols, mpl.colors.Colormap) else cmap_or_cols
        if isinstance(cm, mpl.colors.ListedColormap) and cm.N <= 24:
            idx = np.minimum((np.linspace(0, 1, w, endpoint=False) * cm.N).astype(int), cm.N - 1)
            cols = np.array(cm.colors)[idx, :3]
        else:
            cols = cm(np.linspace(0, 1, w))[:, :3]
    return Image.fromarray(u8(np.repeat(cols[None], h, 0)))


def sheet(tiles, fn, title, subtitle, ncols=5, tile=440, pad=26, bg=BG, footer=None):
    """tiles: list of dict(img=PIL or float array, label, sub, bar=cmap-or-colors or None)."""
    n = len(tiles)
    nrows = (n + ncols - 1) // ncols
    lab_h = 74
    top = 130
    W = pad + ncols * (tile + pad)
    H = top + nrows * (tile + lab_h + pad) + (60 if footer else 20)
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    d.text((pad, 26), title, font=F_TITLE, fill=INK)
    d.text((pad, 82), subtitle, font=F_SUB, fill=GREY)
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncols)
        x = pad + c * (tile + pad)
        y = top + r * (tile + lab_h + pad)
        im = t["img"] if isinstance(t["img"], Image.Image) else resize(t["img"], (tile, tile))
        canvas.paste(im, (x, y))
        yb = y + tile + 4
        if t.get("bar") is not None:
            canvas.paste(strip(t["bar"], tile), (x, yb))
            yb += 18
        d.text((x, yb + 2), t["label"], font=F_LAB, fill=INK)
        if t.get("sub"):
            d.text((x, yb + 26), t["sub"], font=F_SMALL, fill=GREY)
    if footer:
        d.text((pad, H - 48), footer, font=F_SMALL, fill=GREY)
    canvas.save(GAL + fn, optimize=True)
    print("wrote", GAL + fn, canvas.size)


def crop(x, frac=1.0, cy=0.5, cx=0.5):
    H, W = x.shape[:2]
    h, w = int(H * frac), int(W * frac)
    y0 = int(np.clip(cy * H - h / 2, 0, H - h))
    x0 = int(np.clip(cx * W - w / 2, 0, W - w))
    return x[y0:y0 + h, x0:x0 + w]


def up(x, k):
    return np.repeat(np.repeat(x, k, 0), k, 1)


# ============================================================================ split sheets
SPLIT_ORDER = ["sd_spectral", "verdigris_copper", "indigo_madder", "aurora_ember", "cyanotype_vandyke",
               "hubble_sho", "klimt_lapis", "synthwave", "cinestill", "hokusai_sunset",
               "crt_phosphor", "malachite_rhodochrosite", "morandi", "riso_pink_blue", "crameri_bukavu"]


SIDE_SUB = {"sd_spectral": "purple-blue-green | red-orange (matplotlib Spectral)",
            "crameri_bukavu": "counter-example: sea-level map has a light coast"}


def split_tiles(x, near, order=SPLIT_ORDER, extra_light=True, tile=440):
    tiles = []
    for k in order:
        p = P.PAIRINGS[k]
        img = P.render_split(x, k, near_boundary=near)
        tiles.append(dict(img=img, label=p["title"][:40], sub=SIDE_SUB.get(k, "x<0: left half of bar | x>0: right half"),
                          bar=P.split_cmap(p["neg"], p["pos"])))
    if extra_light:
        for k in ["aurora_ember"]:
            p = P.PAIRINGS[k]
            tiles.append(dict(img=P.render_split(x, k, near_boundary=near, seam="light"),
                              label=p["title"] + " (light seam)", sub="same ramps reversed: glowing boundary",
                              bar=P.split_cmap(p["neg"], p["pos"], seam="light")))
    return tiles


def sheet_split_train():
    d = S.trainability("liu")
    x = crop(d["x"], 0.5, 0.36, 0.72)
    sheet(split_tiles(x, "large"), "split_trainability.png",
          "Split-at-the-boundary pairings: trainability fractal",
          "Real data: Sohl-Dickstein convergence measure for Liu's rippled quadratic (eps 0.05), 2048^2 (eta0, eta1) grid, 1024^2 crop "
          "(trainability-fractal). Per-side rank normalisation, pastel 0.75.",
          footer="Declared aesthetic mapping: sign(M) picks the side (converged = left half of the bar, diverged = right half); "
                 "colour position = rank distance from the boundary.")


def sheet_split_basin():
    d = S.basin_signed()
    x = crop(d["x"], 0.62, 0.55, 0.7)
    sheet(split_tiles(x, "large"), "split_basins.png",
          "Split-at-the-boundary pairings: GD initialisation basins",
          "Real data: GD on 1/2(1 - x^2 y^2)^2, eta = 0.2 (gd-bifurcation zhu4 z1T, 4096^2 stride 2, crop). "
          "Converged: steps to loss < 1e-12. Diverged: smooth escape value.",
          footer="Declared aesthetic mapping: per-side rank of convergence / escape time; slowest (at the fractal boundary) = seam colour.")


def sheet_split_lyap():
    d = S.lyapunov("AABAB")
    x = d["x"]
    sheet(split_tiles(x, "small"), "split_lyapunov.png",
          "Split-at-the-boundary pairings: Lyapunov plane",
          "Real data: Lyapunov exponent of GD with alternating learning rates AABAB, 2048^2 (gd-bifurcation). "
          "lambda < 0 ordered = neg side, lambda > 0 chaotic = pos side.",
          footer="Declared aesthetic mapping: per-side rank of |lambda|; lambda = 0 (edge of chaos) = seam colour.")


# ============================================================================ sequential sheets
SEQ = ["crameri_batlow", "crameri_oslo", "crameri_lajolla", "cet_fire", "ironbow", "klimt_gold", "rothko",
       "ember", "aurora", "bioluminescence", "deep_sea", "nippon_aizome", "cyanotype", "platinum",
       "vandyke", "crt_green", "crt_amber", "synthwave", "tam", "cinestill"]


def seq_tiles(v, names, sub_fn=None):
    tiles = []
    for nm in names:
        s = P.SCHEMES[nm]
        cm = s.cmap()
        img = cm(np.nan_to_num(v))[..., :3]
        tiles.append(dict(img=img, label=s.title[:38], sub=f"{s.kind} | {s.family}", bar=cm))
    return tiles


def sheet_sequential():
    d = S.bifurcation_density("p05")
    C = d["x"]
    v = np.log1p(C) / np.log1p(np.percentile(C, 99.9))
    v = np.clip(v, 0, 1)
    # square-ish crop of the 1500 x 2400 page, centred
    v = v[:, 300:1800]
    names = SEQ
    sheet(seq_tiles(v, names), "sequential_bifurcation.png",
          "Sequential schemes: bifurcation density",
          "Real data: histogram of GD iterates vs learning rate, period-5 window page of the gd-bifurcation atlas (1500 x 1500 crop). "
          "log(1 + count), clipped at the 99.9th percentile.",
          footer="Measured: visit density. Aesthetic: colormap only (same normalisation in every tile).")


SPEC = ["crameri_batlow", "crameri_lajolla", "cet_bmy", "ironbow", "synthwave", "tam", "klimt_gold", "aurora",
        "bioluminescence", "crt_amber", "crt_green", "hokusai_wave", "nippon_beni", "portra", "cinestill", "malachite",
        "sepia", "landsat_false", "brewer_ylgnbu", "eink16"]


def sheet_spectrogram():
    d = S.sigma_delta_idle(3)
    x = d["x"]
    v = np.clip((x + 120) / 80, 0, 1)   # dB window [-120, -40]: the idle tones live here
    names = SPEC
    tiles = []
    for nm in names:
        s = P.SCHEMES[nm]
        cm = s.cmap()
        vv = v if nm != "brewer_ylgnbu" else v  # YlGnBu is light->dark: loud tones dark on paper
        img = cm(vv)[..., :3]
        tiles.append(dict(img=img, label=s.title[:38], sub=f"{s.kind} | {s.family}", bar=cm))
    sheet(tiles, "sequential_spectrum.png",
          "Sequential schemes: sigma-delta idle-tone spectrum",
          "Real data: 1-bit sigma-delta output spectrum vs DC input, golden-ratio zoom level 3 (hardware/dither sdzoom_deep), "
          "dB clipped to [-120, -40] so the idle-tone lattice fills the ramp.",
          footer="Measured: spectral power in dB. Aesthetic: colormap only. e-ink 16 shows 16 deliberate bands.")


# ============================================================================ diverging (symmetric)
DIV = ["brewer_spectral", "brewer_brbg", "crameri_vik", "crameri_berlin", "hiroshige", "okeeffe", "isfahan",
       "verdigris", "hubble_sho", "hypsometric", "crameri_bukavu"]


def sheet_diverging():
    d = S.random_field("heaviside_L3")
    x = d["x"]
    a = np.percentile(np.abs(x), 99)
    v = np.clip(x / a, -1, 1) * 0.5 + 0.5
    tiles = []
    for nm in DIV:
        s = P.SCHEMES[nm]
        cm = s.cmap()
        tiles.append(dict(img=cm(v)[..., :3], label=s.title[:38], sub=f"{s.kind} | {s.family} | linear, centred", bar=cm))
    for k in ["sd_spectral", "verdigris_copper", "indigo_madder", "hubble_sho"]:
        p = P.PAIRINGS[k]
        tiles.append(dict(img=P.render_split(x, k, near_boundary="small"), label="split: " + p["title"][:30],
                          sub="rank split at f = 0 (level set = seam)", bar=P.split_cmap(p["neg"], p["pos"])))
    sheet(tiles, "diverging_random_field.png",
          "Diverging schemes: random deep-network field on the sphere",
          "Real data: output of a random 3-layer Heaviside network on a sphere patch, 2048^2 (depth-roughness tiles_2048). "
          "Linear norm, +-99th pct of |f|; last row = rank split at f = 0.",
          footer="Measured: network output f. Aesthetic: colormap and normalisation. Split tiles turn the zero level set into a dark seam.")


# ============================================================================ cyclic
CYC = [("crameri_romaO", None), ("cet_c2", None), ("twilight", "matplotlib twilight"),
       ("cmc.vikO", "Crameri vikO"), ("cmc.brocO", "Crameri brocO"), ("cmc.bamO", "Crameri bamO"),
       ("verdigris_cyclic", None), ("shibori_cyclic", None), ("hsv", "hsv (not uniform)"), ("cet_CET_C5", "CET C5 grey cyclic")]


def sheet_cyclic():
    d = S.gradient_direction("heaviside_L2", smooth=2.0)
    th = crop(d["x"], 0.5)
    mag = crop(d["mag"], 0.5)
    shade = np.clip(mag / np.percentile(mag, 97), 0, 1) ** 0.5
    tiles = []
    for nm, lab in CYC:
        if nm in P.SCHEMES:
            s = P.SCHEMES[nm]
            cm, lab, sub = s.cmap(), s.title, f"cyclic | {s.family}"
        else:
            cm = mpl.colormaps[nm]
            sub = "cyclic | library"
        img = cm(th)[..., :3]
        tiles.append(dict(img=img, label=lab[:38], sub=sub, bar=cm))
    for nm in ["crameri_romaO", "verdigris_cyclic"]:
        s = P.SCHEMES[nm]
        cm = s.cmap()
        img = cm(th)[..., :3] * (0.15 + 0.85 * shade[..., None])
        tiles.append(dict(img=img, label=s.title + " x |grad|", sub="value = gradient magnitude (declared)", bar=cm))
    sheet(tiles, "cyclic_gradient_direction.png",
          "Cyclic schemes: gradient direction of a random-network field",
          "Real data: arg(grad f) of the 2-layer Heaviside field (depth-roughness), Gaussian pre-smoothing sigma = 2 px, centre crop.",
          footer="Measured: gradient angle in [0, 2 pi). Aesthetic: cyclic map; shibori is axial (theta ~ theta + pi) and misreads full-circle data.")


# ============================================================================ categorical
CAT = ["nippon_categorical", "hokusai_categorical", "klimt_categorical", "kente", "bauhaus", "memphis", "kodachrome",
       "morris", "lapis", "vaporwave", "morandi_categorical", "cet_glasbey"]


def sheet_categorical():
    d = S.basin_classes("wide_4096", stride=2)
    c = d["x"]
    tev = d["tev"]
    tiles = []
    # class order: diverged gets the 'ground-like' colour; choose per palette explicitly
    for nm in CAT:
        s = P.SCHEMES[nm]
        cols = np.array([P.hex2rgb(h) for h in s.sample_hex(8)])
        if len(cols) < 5:
            cols = np.concatenate([cols, cols[:5 - len(cols)] * 0.6])
        order = {"morris": [0, 1, 2, 3, 4], "lapis": [0, 3, 1, 2, 4], "bauhaus": [3, 0, 1, 2, 4],
                 "kente": [3, 0, 1, 2, 4], "vaporwave": [3, 0, 1, 2, 4]}.get(nm, [len(cols) - 1, 0, 1, 2, 3] if nm in ("nippon_categorical",) else [0, 1, 2, 3, 4])
        pal = cols[order]
        img = pal[c]
        tiles.append(dict(img=img, label=s.title[:38], sub=f"categorical | {s.family}", bar=pal))
    # two tiles with lightness modulated by convergence time (declared)
    for nm in ["nippon_categorical", "morris"]:
        s = P.SCHEMES[nm]
        cols = np.array([P.hex2rgb(h) for h in s.sample_hex(8)])
        order = [len(cols) - 1, 0, 1, 2, 3] if nm == "nippon_categorical" else [0, 1, 2, 3, 4]
        pal = cols[order]
        r = P.rank_normalize(np.log(tev))
        k = np.where(c == 0, 1.0, 0.55 + 0.45 * (1 - r))
        tiles.append(dict(img=pal[c] * k[..., None] + (1 - k[..., None]) * 0.97 * (c[..., None] > 0),
                          label=s.title[:30] + " + time", sub="tint = rank of convergence time (declared)", bar=pal))
    sheet(tiles, "categorical_basins.png",
          "Categorical palettes: which minimum does GD reach?",
          "Real data: GD on 1/2(1 - x^2 y^2)^2, eta = 0.2, init plane [0, 3.8]^2 (gd-bifurcation zhu4 wide, 4096^2 stride 2). "
          "5 classes: diverged + 4 sign branches of the minimum.",
          footer="Measured: basin label. Aesthetic: palette and class-to-colour assignment (diverged = darkest/ground colour).")


# ============================================================================ riso
def am_screen(shape, angle, period):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    a = np.deg2rad(angle)
    u = (xx * np.cos(a) + yy * np.sin(a)) / period
    w = (-xx * np.sin(a) + yy * np.cos(a)) / period
    return (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * w)) / 4 + 0.5


def riso_render(x, near, inks, paper, size=880, period=5.0, misreg=(2, -1), seed=0):
    """Two-ink halftone overprint of a signed field: ink 0 where x < 0, ink 1 where x > 0; dot coverage =
    closeness to the boundary (rank), so the seam prints solid. Third ink (if any) prints the boundary
    edge set. Aesthetic: screens, misregistration, grain."""
    H, W = x.shape
    yy = (np.arange(size) * H / size).astype(int)
    xx = (np.arange(size) * W / size).astype(int)
    xs = x[np.ix_(yy, xx)]
    v = P.signed_rank_normalize(xs, near_boundary=near, pastel=1.0)
    dens = 0.1 + 0.85 * (1 - np.abs(v)) ** 1.4
    neg = xs < 0
    covA = (neg & (dens > am_screen((size, size), 15, period))).astype(float)
    covB = ((~neg) & (dens > am_screen((size, size), 75, period))).astype(float)
    covB = np.roll(covB, misreg, (0, 1))
    rng = np.random.default_rng(seed)
    grain = ndimage.gaussian_filter(rng.standard_normal((size, size)), 1.0)
    covs = [np.clip(covA * (0.88 + 0.1 * grain), 0, 1), np.clip(covB * (0.88 + 0.1 * np.roll(grain, 9, 0)), 0, 1)]
    if len(inks) > 2:
        e = np.zeros((size, size), bool)
        e[:-1, :] |= neg[:-1, :] != neg[1:, :]
        e[:, :-1] |= neg[:, :-1] != neg[:, 1:]
        covs.append(np.roll(e.astype(float), (-1, 2), (0, 1)) * 0.95)
    img = P.overprint(covs, inks[:len(covs)], paper)
    return img * (1 + 0.02 * grain[..., None])


RISO_SETS = [("fluo_pink", "blue"), ("federal_blue", "sunflower"), ("teal", "bright_red"), ("aqua", "orange"),
             ("burgundy", "mint"), ("indigo", "melon"), ("medium_blue", "fluo_pink", "yellow"),
             ("sea_blue", "copper"), ("hunter_green", "fluo_orange"), ("grape", "light_lime")]


def sheet_riso():
    d = S.basin_signed()
    x = crop(d["x"], 0.62, 0.55, 0.7)
    tiles = []
    for ks in RISO_SETS:
        inks = [P.RISO[k] for k in ks]
        img = riso_render(x, "large", inks, P.RISO_PAPER, size=440)
        tiles.append(dict(img=Image.fromarray(u8(img)), label=" + ".join(k.replace("_", " ").title() for k in ks)[:40],
                          sub="riso inks (stencil.wiki hex)" + (" | 3rd ink = boundary" if len(ks) > 2 else ""),
                          bar=np.array([P.hex2rgb(P.RISO_PAPER)] + [P.hex2rgb(i) for i in inks])[
                              np.repeat(np.arange(len(inks) + 1), 440 // (len(inks) + 1) + 1)[:440]]))
    for nm in ["technicolor2", "letterpress"]:
        s = P.SCHEMES[nm]
        img = riso_render(x, "large", s.stops, s.ground, size=440, misreg=(1, 0))
        tiles.append(dict(img=Image.fromarray(u8(img)), label=s.title[:40], sub="2 spot inks | " + s.family,
                          bar=np.array([P.hex2rgb(s.ground)] + [P.hex2rgb(i) for i in s.stops])[
                              np.repeat(np.arange(3), 147)[:440]]))
    sheet(tiles, "riso_inks_basins.png", "Spot-ink pairs: halftone overprint of a signed field",
          "Real data: same GD basin crop as split_basins.png. Ink 1 = converged, ink 2 = diverged; dot area = closeness to the boundary (rank).",
          ncols=4, footer="Aesthetic: AM screens at 15/75 deg, 5 px period, 2 px misregistration, multiplicative overprint, paper grain.")


# ============================================================================ lightness profiles
def sheet_profiles():
    import matplotlib.pyplot as plt
    names = [s.name for s in P.SCHEMES.values() if s.kind not in ("inks",)]
    n = len(names)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(ncol * 6.2, nrow * 0.82 + 1.4), dpi=110, facecolor="#eeeae3")
    fig.text(0.012, 1 - 0.3 / (nrow * 0.82 + 1.4), "Lightness profiles (CAM02-UCS J') and deuteranopia simulation",
             fontsize=17, family="serif", weight="bold", va="top")
    fig.text(0.012, 1 - 0.75 / (nrow * 0.82 + 1.4),
             "Each row: normal strip / Machado-2009 deuteranopia strip / J' curve (0-100, grey). Numbers: speed CV (0 = perfectly uniform) and J' reversals.",
             fontsize=10, va="top", color="#555")
    H = nrow * 0.82 + 1.4
    for i, nm in enumerate(names):
        s = P.SCHEMES[nm]
        r, c = divmod(i, ncol)
        x0 = 0.012 + c / ncol
        y0 = 1 - (1.2 + (r + 1) * 0.82) / H
        ax = fig.add_axes([x0 + 0.1, y0 + 0.12 / H, 1 / ncol - 0.16, 0.62 / H])
        cm = s.cmap()
        if s.kind == "categorical":
            cols = np.array([P.hex2rgb(h) for h in s.sample_hex(16)])
        else:
            cols = cm(np.linspace(0, 1, 256))[:, :3]
        deut = P.simulate_cvd(cols, "deuteranopia")
        ax.imshow(np.stack([cols, cols, deut], 0), aspect="auto", extent=[0, 1, 0, 1], interpolation="nearest")
        J = P.rgb_to_cam02ucs(cols)[:, 0]
        ax2 = ax.twinx()
        t = np.linspace(0, 1, len(J)) if s.kind != "categorical" else (np.arange(len(J)) + 0.5) / len(J)
        ax2.plot(t, J, color="white", lw=2.6)
        ax2.plot(t, J, color="#222", lw=1.1, marker="o" if s.kind == "categorical" else None, ms=3)
        ax2.set_ylim(0, 100)
        for a in (ax, ax2):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        m = P.metrics(s)
        if s.kind == "categorical":
            info = f"min dE00 {m['min_dE']:.0f} / deut {m['min_dE_deut']:.0f}"
        else:
            info = f"cv {m['speed_cv']:.2f}  rev {m['J_reversals']}"
        fig.text(x0, y0 + 0.5 / H, s.name, fontsize=9.5, weight="bold", va="center")
        fig.text(x0, y0 + 0.25 / H, info, fontsize=8, va="center", color="#555")
    fig.savefig(GAL + "lightness_profiles.png", facecolor=fig.get_facecolor())
    print("wrote lightness_profiles.png")


SHEETS = dict(split_train=sheet_split_train, split_basin=sheet_split_basin, split_lyap=sheet_split_lyap,
              sequential=sheet_sequential, spectrogram=sheet_spectrogram, diverging=sheet_diverging,
              cyclic=sheet_cyclic, categorical=sheet_categorical, riso=sheet_riso, profiles=sheet_profiles)

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for k, f in SHEETS.items():
        if "all" in which or k in which:
            f()
