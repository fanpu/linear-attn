"""palettes.py - a catalogue of colour schemes for the ML/hardware art projects.

Import from any project (read-only use):

    import sys; sys.path.insert(0, '/home/fzeng/ml/research/art/color-research')
    import palettes as P

    cm = P.get('aizome')                 # a matplotlib Colormap (also registered as 'art.aizome')
    plt.imshow(x, cmap='art.aizome')     # registration happens on import
    rgb = P.render_split(M, 'verdigris_copper', near_boundary='large')  # Sohl-Dickstein-style split
    rgb = P.render_split(lam, 'sd_spectral', near_boundary='small')     # Lyapunov exponent etc.
    P.SCHEMES['aizome'].stops, P.PAIRINGS.keys(), P.list_schemes(kind='sequential')

Conventions
-----------
* Every scheme has a `kind`: 'sequential', 'diverging', 'cyclic', 'categorical', 'inks' (2-3 spot
  inks + paper) or 'multiseq' (two sequential halves meeting at a boundary, e.g. topography).
* Hand-written stop lists are interpolated in CIELAB (not sRGB), which keeps lightness steps
  smooth between stops and avoids the muddy mid-tones of sRGB blends. Library maps (ColorBrewer
  via matplotlib, Crameri via cmcrameri, CET via colorcet) are used as-is.
* Split pairings (`PAIRINGS`): each side is a ramp from the SEAM colour (dark, at the boundary)
  outward to a pastel. Side 'neg' is used where x < 0, side 'pos' where x > 0. In the
  trainability measure x < 0 = converged, so 'neg' = the converged side (purple/blue/green in
  Sohl-Dickstein's Spectral), 'pos' = diverged (deep red/orange).
* Colour science helpers implemented here (no extra deps): sRGB <-> CIELAB (D65), CIEDE2000,
  Machado et al. (2009) colour-vision-deficiency simulation (severity 1.0).

Hex stops for art/film/nature schemes are declared curated evocations unless a source is given;
see COLOR_SCHEMES.md for lineage and links.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Colormap, ListedColormap, to_rgb

try:  # registers 'cmc.*' maps
    import cmcrameri.cm  # noqa: F401
except Exception:  # pragma: no cover
    pass
try:  # registers 'cet_*' maps
    import colorcet  # noqa: F401
except Exception:  # pragma: no cover
    pass

PREFIX = "art."

# ============================================================================ colour science
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def hex2rgb(h):
    return np.array(to_rgb(h), dtype=float)


def rgb2hex(rgb):
    r = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(r[:3])


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    """sRGB in [0,1] (..., 3) -> CIELAB (D65)."""
    xyz = srgb_to_linear(rgb) @ _M_RGB2XYZ.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    d = 6 / 29
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29)) * _WHITE
    return linear_to_srgb(xyz @ _M_XYZ2RGB.T)


def deltaE2000(lab1, lab2):
    """CIEDE2000 colour difference, broadcasting over leading dims."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# CIECAM02 -> CAM02-UCS (Luo, Cui & Li 2006), same default viewing conditions as colorspacious:
# D65 white (Y=100), L_A = 64/pi/5 cd/m^2, Y_b = 20, average surround.
_M_CAT02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
_M_HPE = np.array([[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.0, 0.0, 1.0]])


def _cam02_setup(LA=64 / np.pi / 5, Yb=20.0, F=1.0, c=0.69, Nc=1.0):
    XYZw = _WHITE * 100
    RGBw = _M_CAT02 @ XYZw
    D = np.clip(F * (1 - (1 / 3.6) * np.exp((-LA - 42) / 92)), 0, 1)
    Dk = (100 * D / RGBw + 1 - D)
    k = 1 / (5 * LA + 1)
    FL = 0.2 * k ** 4 * (5 * LA) + 0.1 * (1 - k ** 4) ** 2 * (5 * LA) ** (1 / 3)
    n = Yb / 100
    Nbb = 0.725 * (1 / n) ** 0.2
    z = 1.48 + np.sqrt(n)
    M = _M_HPE @ np.linalg.inv(_M_CAT02)

    def post(rgbc):
        rp = rgbc @ M.T
        x = (FL * np.abs(rp) / 100) ** 0.42
        return np.sign(rp) * 400 * x / (27.13 + x) + 0.1

    aw = post(RGBw * Dk)
    Aw = (2 * aw[0] + aw[1] + aw[2] / 20 - 0.305) * Nbb
    return dict(Dk=Dk, FL=FL, n=n, Nbb=Nbb, z=z, post=post, Aw=Aw, c=c, Nc=Nc)


_CAM = _cam02_setup()


def rgb_to_cam02ucs(rgb):
    """sRGB [0,1] (..., 3) -> CAM02-UCS (J', a', b')."""
    p = _CAM
    xyz = srgb_to_linear(rgb) @ _M_RGB2XYZ.T * 100
    rgbc = (xyz @ _M_CAT02.T) * p["Dk"]
    r = p["post"](rgbc)
    Ra, Ga, Ba = r[..., 0], r[..., 1], r[..., 2]
    a = Ra - 12 * Ga / 11 + Ba / 11
    b = (Ra + Ga - 2 * Ba) / 9
    h = np.arctan2(b, a)
    et = 0.25 * (np.cos(h + 2) + 3.8)
    A = (2 * Ra + Ga + Ba / 20 - 0.305) * p["Nbb"]
    J = 100 * np.maximum(A / p["Aw"], 0) ** (p["c"] * p["z"])
    t = (50000 / 13 * p["Nc"] * p["Nbb"] * et * np.hypot(a, b)) / (Ra + Ga + 21 / 20 * Ba)
    C = np.maximum(t, 0) ** 0.9 * np.sqrt(J / 100) * (1.64 - 0.29 ** p["n"]) ** 0.73
    Mc = C * p["FL"] ** 0.25
    Jp = 1.7 * J / (1 + 0.007 * J)
    Mp = np.log1p(0.0228 * Mc) / 0.0228
    return np.stack([Jp, Mp * np.cos(h), Mp * np.sin(h)], -1)


def deltaE_ucs(rgb1, rgb2):
    return np.linalg.norm(rgb_to_cam02ucs(rgb1) - rgb_to_cam02ucs(rgb2), axis=-1)


# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB.
CVD_MATRICES = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def simulate_cvd(rgb, kind="deuteranopia"):
    lin = srgb_to_linear(np.asarray(rgb, float)[..., :3])
    return linear_to_srgb(np.clip(lin @ CVD_MATRICES[kind].T, 0, 1))


# ============================================================================ building colormaps
def _uniformize(rgb, pieces=1):
    """Resample a fine RGB ramp to constant CAM02-UCS speed (piecewise, keeping piece borders fixed)."""
    n = len(rgb)
    out = []
    edges = np.linspace(0, n - 1, pieces + 1).round().astype(int)
    for i in range(pieces):
        seg = rgb[edges[i]:edges[i + 1] + 1]
        d = np.linalg.norm(np.diff(rgb_to_cam02ucs(seg), axis=0), axis=-1)
        arc = np.concatenate([[0], np.cumsum(d)])
        arc /= arc[-1] if arc[-1] > 0 else 1
        m = len(seg)
        tt = np.linspace(0, 1, m)
        res = np.stack([np.interp(tt, arc, seg[:, k]) for k in range(3)], -1)
        out.append(res if i == 0 else res[1:])
    return np.concatenate(out)


def lab_ramp(stops, n=256, positions=None, name="ramp", cyclic=False, uniform=False, pieces=1):
    """ListedColormap interpolating hex stops in CIELAB.

    uniform=True re-parameterises the ramp to constant CAM02-UCS speed (in `pieces` equal parts,
    e.g. pieces=2 for a diverging map so the centre stop stays at 0.5)."""
    cols = [hex2rgb(s) for s in stops]
    if cyclic and not np.allclose(cols[0], cols[-1]):
        cols = cols + [cols[0]]
    lab = rgb_to_lab(np.array(cols))
    pos = np.linspace(0, 1, len(cols)) if positions is None else np.asarray(positions, float)
    m = n * 8 if uniform else n
    t = np.linspace(0, 1, m)
    out = np.clip(lab_to_rgb(np.stack([np.interp(t, pos, lab[:, i]) for i in range(3)], -1)), 0, 1)
    if uniform:
        out = _uniformize(out, pieces)
        out = np.stack([np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(out)), out[:, k]) for k in range(3)], -1)
    return ListedColormap(out, name=name)


def sub_cmap(cmap, t0, t1, n=256, name=None):
    """Portion of a colormap from t0 to t1 (t1 < t0 reverses)."""
    cm = as_cmap(cmap)
    return ListedColormap(cm(np.linspace(t0, t1, n))[:, :3], name=name or f"{cm.name}[{t0}:{t1}]")


@dataclass
class Scheme:
    name: str
    kind: str                      # sequential | diverging | cyclic | categorical | inks | multiseq
    family: str
    title: str
    stops: list = field(default_factory=list)   # hex stops (or sampled from `lib` for docs)
    lib: str | None = None         # name of an existing matplotlib-registered map
    ground: str | None = None      # paper / background colour for inks and bright-on-dark maps
    lineage: str = ""
    sources: list = field(default_factory=list)
    suits: str = ""
    caveat: str = ""
    positions: list | None = None

    def cmap(self, n=256):
        if self.lib:
            cm = mpl.colormaps[self.lib]
            if self.kind == "categorical":
                return cm
            return cm
        if self.kind in ("categorical", "inks"):
            return ListedColormap([hex2rgb(s) for s in self.stops], name=PREFIX + self.name)
        uni = self.positions is None and self.kind in ("sequential", "diverging", "cyclic")
        pieces = 2 if (self.kind == "diverging" and len(self.stops) % 2 == 1) else 1
        return lab_ramp(self.stops, n=n, positions=self.positions, name=PREFIX + self.name,
                        cyclic=self.kind == "cyclic", uniform=uni, pieces=pieces)

    def sample_hex(self, k=9):
        if self.stops:
            return list(self.stops)
        cm = mpl.colormaps[self.lib]
        if self.kind == "categorical" and hasattr(cm, "colors"):
            return [rgb2hex(c) for c in list(cm.colors)[:k]]
        return [rgb2hex(c) for c in cm(np.linspace(0, 1, k))]


SCHEMES: dict[str, Scheme] = {}


def _add(**kw):
    s = Scheme(**kw)
    SCHEMES[s.name] = s
    return s


# ---------------------------------------------------------------------------- A. scientific / cartographic
_add(name="brewer_spectral", kind="diverging", family="scientific", title="ColorBrewer Spectral",
     lib="Spectral",
     lineage="Cynthia Brewer's ColorBrewer (2002), 11-class diverging scheme; matplotlib 'Spectral'. "
             "The map Sohl-Dickstein uses for trainability fractals (split at the boundary).",
     sources=["https://colorbrewer2.org", "https://github.com/Sohl-Dickstein/fractal"],
     suits="signed two-sided fields when split at the boundary (see PAIRINGS['sd_spectral']).",
     caveat="Not CVD-safe (red/green ends); hue-heavy, mid pale-yellow is the lightness peak.")
_add(name="brewer_brbg", kind="diverging", family="scientific", title="ColorBrewer BrBG",
     lib="BrBG", lineage="ColorBrewer diverging brown-blue-green; soil/water cartography.",
     sources=["https://colorbrewer2.org"],
     suits="symmetric signed fields around zero (random-field level sets, dither error).",
     caveat="Colour-blind safe per ColorBrewer; pale centre hides small |x|.")
_add(name="brewer_ylgnbu", kind="sequential", family="scientific", title="ColorBrewer YlGnBu",
     lib="YlGnBu", lineage="ColorBrewer multi-hue sequential.", sources=["https://colorbrewer2.org"],
     suits="densities on paper-white ground (bifurcation densities, histograms).",
     caveat="Only 9 native classes; lightness compresses in the yellow end.")
for _n, _k, _t, _s, _c in [
    ("batlow", "sequential", "Crameri batlow", "all-round sequential; spectrograms, escape times",
     "Perceptually uniform, CVD-friendly, readable in greyscale."),
    ("oslo", "sequential", "Crameri oslo", "dark-ground densities; convergence time; engraved plates",
     "Uniform, near-monochrome blue; very CVD-safe."),
    ("lajolla", "sequential", "Crameri lajolla", "warm densities on light ground; heat/energy",
     "Uniform; reverse for dark ground."),
    ("bukavu", "multiseq", "Crameri bukavu", "topography-like fields with a meaningful zero (sea level)",
     "Two sequential halves with a jump at 0.5: intended to be used with a centred norm."),
    ("berlin", "diverging", "Crameri berlin", "signed fields with a DARK centre: boundary as a dark seam",
     "Dark-centre diverging, uniform; centre is near black so small |x| disappears on dark ground."),
    ("vik", "diverging", "Crameri vik", "signed fields, light centre (random fields, dither error)",
     "Uniform, CVD-friendly blue/brown."),
    ("romaO", "cyclic", "Crameri romaO", "cyclic phase (gradient direction, residues)",
     "Uniform cyclic; hue-based so check under CVD."),
]:
    _add(name="crameri_" + _n, kind=_k, family="scientific", title=_t, lib="cmc." + _n,
         lineage="Fabio Crameri, Scientific colour maps (2018-2023), designed for perceptual "
                 "uniformity and colour-vision-deficiency readability.",
         sources=["https://www.fabiocrameri.ch/colourmaps/", "https://doi.org/10.1038/s41467-020-19160-7"],
         suits=_s, caveat=_c)
for _n, _lib, _k, _s, _c in [
    ("cet_fire", "cet_fire", "sequential", "dark-ground densities, glowing filaments", "Uniform (CET L3), black->white through red/yellow."),
    ("cet_bmy", "cet_bmy", "sequential", "dark-ground spectrograms, synthwave-adjacent", "Uniform blue-magenta-yellow."),
    ("cet_cbtl1", "cet_CET_CBTL1", "sequential", "CVD-safe sequential for mixed audiences", "Designed for protan/deutan viewers."),
    ("cet_c2", "cet_CET_C2", "cyclic", "cyclic phase", "Four-colour cyclic; bright, strong hue cycle."),
    ("cet_glasbey", "cet_glasbey_dark", "categorical", "many-class basins", "Max-distinct categorical; loud, not 'designed'."),
]:
    _add(name=_n, kind=_k, family="scientific", title="colorcet " + _lib.replace("cet_", ""), lib=_lib,
         lineage="Peter Kovesi, 'Good Colour Maps: How to Design Them' (2015), CET maps; packaged by colorcet (HoloViz).",
         sources=["https://colorcet.com", "https://arxiv.org/abs/1509.03700"], suits=_s, caveat=_c)
_add(name="hypsometric", kind="multiseq", family="scientific", title="Hypsometric tints (atlas)",
     stops=["#0b2a4a", "#1f5f8b", "#6fa8d6", "#cfe7f3", "#4f8a55", "#9cc27a", "#e7dc97", "#d7a15a", "#a36e45", "#7a6353", "#f4f2ee"],
     positions=[0, .2, .38, .5, .5001, .6, .7, .8, .88, .95, 1],
     lineage="Layer tinting of elevation (Hauslab 1840s; Imhof's Swiss relief school): sea blues deepen with "
             "depth, land runs green -> yellow -> brown -> snow. Curated stops.",
     sources=["https://en.wikipedia.org/wiki/Hypsometric_tints", "https://www.shadedrelief.com/hypso/hypso.html"],
     suits="signed fields with a physical-feeling zero: LIGHT seam at the coastline (x=0).",
     caveat="Non-monotone lightness by design (dark sea floor, light coast, dark land then white peaks): "
            "reads as terrain, not as magnitude.")
_add(name="ironbow", kind="sequential", family="scientific", title="Thermal 'ironbow'",
     stops=["#000004", "#1d0b5a", "#6a1a8c", "#b4306f", "#e25a2f", "#f89b14", "#fdd65c", "#fffbe8"],
     lineage="Evocation of the FLIR 'Ironbow' thermal-camera palette (black->indigo->magenta->orange->white).",
     sources=["https://www.flir.com/discover/industrial/picking-a-thermal-color-palette/"],
     suits="energy/heat-like densities on dark ground; loss magnitude; Hessian spectra.",
     caveat="Close to inferno but more saturated in the magenta; minor lightness wobble near orange.")
_add(name="landsat_false", kind="sequential", family="scientific", title="Landsat NIR false colour",
     stops=["#07142b", "#27435a", "#6b8f9b", "#a67f82", "#c9474f", "#e3182f", "#f6b9b5"],
     lineage="Evocation of NASA/USGS Landsat colour-infrared composites (NIR->red): water near black, "
             "bare ground cyan-grey, vegetation red. Curated stops.",
     sources=["https://earthobservatory.nasa.gov/features/FalseColor"],
     suits="sequential 'vigour' quantities (trainability speed, fraction trained).",
     caveat="Hue flips cool->warm mid-ramp: can read as a two-class split; not CVD-safe.")

# ---------------------------------------------------------------------------- B. print traditions
RISO = {  # from stencil.wiki via mattdesl/riso-colors (Feb 2019 scrape), see sources/riso-colors.json
    "black": "#000000", "burgundy": "#914e72", "blue": "#0078bf", "green": "#00a95c", "medium_blue": "#3255a4",
    "bright_red": "#f15060", "federal_blue": "#3d5588", "purple": "#765ba7", "teal": "#00838a",
    "flat_gold": "#bb8b41", "hunter_green": "#407060", "red": "#ff665e", "yellow": "#ffe800",
    "orange": "#ff6c2f", "fluo_pink": "#ff48b0", "metallic_gold": "#ac936e", "cornflower": "#62a8e5",
    "sea_blue": "#0074a2", "indigo": "#484d7a", "midnight": "#435060", "mist": "#d5e4c0",
    "steel": "#375e77", "turquoise": "#00aa93", "sea_foam": "#62c2b1", "violet": "#9d7ad2",
    "orchid": "#aa60bf", "scarlet": "#f65058", "maroon": "#9e4c6e", "brick": "#a75154",
    "sunflower": "#ffb511", "melon": "#ffae3b", "copper": "#bd6439", "bubble_gum": "#f984ca",
    "aqua": "#5ec8e5", "mint": "#82d8d5", "fluo_orange": "#ff7477", "fluo_green": "#44d62c",
    "lake": "#235ba8", "kelly_green": "#67b346", "light_lime": "#e3ed55", "grape": "#6c5d80",
}
RISO_PAPER = "#f4efe3"
_riso_src = ["https://stencil.wiki/colors", "https://github.com/mattdesl/riso-colors"]
for _n, _a, _b, _s in [
    ("riso_fluopink_blue", "fluo_pink", "blue", "the canonical zine pair; converge/diverge phase maps"),
    ("riso_federalblue_sunflower", "federal_blue", "sunflower", "strong value contrast; line-art over a density"),
    ("riso_teal_brightred", "teal", "bright_red", "complementary two-sided fields; overprint gives near-black"),
    ("riso_aqua_orange", "aqua", "orange", "light, poster-like; spectrogram overlays"),
    ("riso_burgundy_mint", "burgundy", "mint", "muted, bookish; basins on cream"),
    ("riso_indigo_melon", "indigo", "melon", "evening palette; Lyapunov planes"),
]:
    _add(name=_n, kind="inks", family="print", title="Riso " + " + ".join(
        x.replace("_", " ").title() for x in (_a, _b)), stops=[RISO[_a], RISO[_b]], ground=RISO_PAPER,
        lineage="Risograph soy inks as listed by stencil.wiki (hex are screen approximations of the drum inks).",
        sources=_riso_src, suits=_s,
        caveat="Two inks + paper = 4 printable tones incl. overprint; screen density carries the magnitude.")
_add(name="riso_trio_pink_yellow_blue", kind="inks", family="print", title="Riso Fluo Pink + Yellow + Medium Blue",
     stops=[RISO["fluo_pink"], RISO["yellow"], RISO["medium_blue"]], ground=RISO_PAPER,
     lineage="Three-drum 'CMY-ish' riso set (stencil.wiki hexes).", sources=_riso_src,
     suits="three-class basin maps with overprint mixtures at shared boundaries.",
     caveat="Yellow on cream has very low contrast (dL* ~ 7); use yellow only for areas, never lines.")
_add(name="cyanotype", kind="sequential", family="print", title="Cyanotype (Prussian blue)",
     stops=["#07172c", "#0e2f55", "#1b4f82", "#3c77a8", "#86aecd", "#d3e0e6", "#f3f0e6"],
     lineage="Herschel 1842; Anna Atkins' 'Photographs of British Algae' (1843). Ferric ammonium citrate + "
             "potassium ferricyanide -> Prussian blue (pigment ~#003153). Curated tonal ramp.",
     sources=["https://en.wikipedia.org/wiki/Cyanotype", "https://www.metmuseum.org/art/collection/search/285381"],
     suits="sequential density on paper; single-hue line-art ground; boundary maps (white lines on blue).",
     caveat="Monochrome: very CVD-safe, uniform after Lab interpolation.")
_add(name="vandyke", kind="sequential", family="print", title="Van Dyke brown print",
     stops=["#1b0f08", "#3d2415", "#664226", "#95704f", "#c6a98a", "#ecdfcb", "#f6f0e4"],
     lineage="Van Dyke brown (kallitype family) iron-silver print, named after the pigment used by Anthony van Dyck. Curated.",
     sources=["https://en.wikipedia.org/wiki/Van_Dyke_brown"],
     suits="sequential density; pairs with cyanotype as the second half of a split (see PAIRINGS).",
     caveat="Monochrome; the top end is paper, so keep a margin to avoid clipping into the ground.")
_add(name="platinum", kind="sequential", family="print", title="Platinum/palladium print",
     stops=["#1f1b18", "#3e3732", "#665c54", "#948779", "#c1b4a2", "#e3d9c8", "#f5efe2"],
     lineage="Pt/Pd printing (Willis 1873): long tonal scale, warm neutral blacks, matte paper. Curated.",
     sources=["https://en.wikipedia.org/wiki/Platinum_print"],
     suits="archival 'observatory plate' renders of densities and relief; greyscale-safe.",
     caveat="Low chroma: structure must be carried by lightness alone (which is perceptually honest).")
_add(name="sepia", kind="sequential", family="print", title="Sepia toning",
     stops=["#24160b", "#4b2e14", "#704214", "#9c6d3c", "#c9a275", "#ecdcc2"],
     lineage="Sepia-toned silver prints (sodium sulphide toning). #704214 is the conventional 'sepia'.",
     sources=["https://en.wikipedia.org/wiki/Sepia_(color)"], suits="sequential density; nostalgia plates.",
     caveat="Monochrome, uniform; slightly compressed highlights.")
_add(name="blueprint", kind="inks", family="print", title="Blueprint (diazo/cyanotype)",
     stops=["#eef3f8"], ground="#1d3f78",
     lineage="Engineering blueprint (cyanotype reprographics, 1870s-1940s): white lines on Prussian-blue ground.",
     sources=["https://en.wikipedia.org/wiki/Blueprint"],
     suits="line-art: boundaries, level sets, contour stacks; one light ink on dark ground.",
     caveat="Single ink: magnitude only via line weight/hatching.")
_add(name="letterpress", kind="inks", family="print", title="Letterpress vermilion + black",
     stops=["#d63a2a", "#1c1b1a"], ground="#f3eee2",
     lineage="Two-colour letterpress (rubrication tradition: black text, red initials) on cotton stock. Curated.",
     sources=["https://en.wikipedia.org/wiki/Rubrication"],
     suits="line-art plates: black for structure, vermilion for the measured boundary.",
     caveat="Red/black are separable under deuteranopia only by lightness (L* 51 vs 10): fine.")

# ---------------------------------------------------------------------------- C. painting / design
_mb = ["https://github.com/BlakeRMills/MetBrewer"]
_add(name="hokusai_wave", kind="sequential", family="painting", title="Hokusai blues (MetBrewer Hokusai2)",
     stops=["#0a3351", "#134b73", "#2f70a1", "#4692b0", "#72aeb6", "#abc9c8", "#f2ece0"],
     lineage="MetBrewer 'Hokusai2' (sampled from 'The Great Wave off Kanagawa', c.1831, Prussian-blue woodblock) "
             "+ an added paper end.", sources=_mb + ["https://www.metmuseum.org/art/collection/search/36491"],
     suits="sequential density; cool plates; ocean-like spectrograms.",
     caveat="MetBrewer lists Hokusai2 as colourblind-friendly; cyan middle slightly compresses L*.")
_add(name="hokusai_categorical", kind="categorical", family="painting", title="Hokusai1 (MetBrewer)",
     stops=["#6d2f20", "#b75347", "#df7e66", "#e09351", "#edc775", "#94b594", "#224b5e"],
     lineage="MetBrewer 'Hokusai1'.", sources=_mb, suits="basin classes with an ukiyo-e warmth.",
     caveat="MetBrewer flags NOT colourblind-safe; three adjacent warm oranges are close.")
_add(name="hiroshige", kind="diverging", family="painting", title="Hiroshige (MetBrewer)",
     stops=["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7", "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"],
     lineage="MetBrewer 'Hiroshige' (Utagawa Hiroshige prints): sunset orange to indigo.", sources=_mb,
     suits="symmetric signed fields; a softer Spectral-like diverging map; split halves work too.",
     caveat="MetBrewer lists as colourblind-friendly; the orange end is lighter than the blue end (asymmetric L*).")
_add(name="nippon_aizome", kind="sequential", family="textile", title="Aizome indigo dips (Nippon colors)",
     stops=["#08192d", "#0f2540", "#0b346e", "#006284", "#33a6b8", "#81c7d4", "#a5dee4", "#fcfaf2"],
     lineage="Japanese traditional colour names for successive indigo dips, dark to light: KACHI 褐, KON 紺, "
             "RURIKON 瑠璃紺, HANADA 縹, ASAGI 浅葱, MIZU 水, KAMENOZOKI 瓶覗, SHIRONERI 白練. Hexes from nipponcolors.com.",
     sources=["https://nipponcolors.com", "https://en.wikipedia.org/wiki/Aizome"],
     suits="sequential density; indigo textile renders; the cool half of a split.",
     caveat="Hue drifts navy->cyan; uniform after Lab interpolation, CVD-safe (blue axis).")
_add(name="nippon_beni", kind="sequential", family="painting", title="Beni reds (Nippon colors)",
     stops=["#3f2b36", "#64363c", "#9f353a", "#cb1b45", "#e87a90", "#f8c3cd", "#fedfe1"],
     lineage="KUROBENI 黒紅, KUWAZOME 桑染, ENJI 臙脂, KURENAI 紅, USUBENI 薄紅, TAIKOH 退紅, SAKURA 桜 (nipponcolors.com).",
     sources=["https://nipponcolors.com"], suits="warm sequential; the diverged half of a split; dark-seam reds.",
     caveat="KURENAI is very saturated: a chroma spike mid-ramp can look like a band on smooth data.")
_add(name="nippon_categorical", kind="categorical", family="painting", title="Nippon traditional set",
     stops=["#cb1b45", "#ffb11b", "#1b813e", "#005caf", "#592c63", "#ca7a2c", "#86a697", "#1c1c1c"],
     lineage="KURENAI 紅, YAMABUKI 山吹, TOKIWA 常磐, RURI 瑠璃, MURASAKI 紫, KOHAKU 琥珀, SABISEIJI 錆青磁, SUMI 墨.",
     sources=["https://nipponcolors.com"], suits="basin classes; Edo-textile look on SHIRONERI (#fcfaf2) paper.",
     caveat="KURENAI vs TOKIWA collapse under deuteranopia (check table); separate them spatially.")
_add(name="bauhaus", kind="categorical", family="design", title="Bauhaus primaries",
     stops=["#d4282d", "#f2c21a", "#1f4e9a", "#141414", "#e9e2d0"],
     lineage="Itten/Kandinsky colour-form correspondences (yellow triangle, red square, blue circle; 1923 questionnaire). Curated.",
     sources=["https://en.wikipedia.org/wiki/Bauhaus", "https://www.moma.org/artists/2981"],
     suits="3-4 class basins with geometric composition; poster-style plates.",
     caveat="Red/black similar-ish under protanopia in lightness terms; yellow/cream weak contrast.")
_add(name="swiss", kind="inks", family="design", title="Swiss style (Muller-Brockmann)",
     stops=["#e2231a", "#111111"], ground="#f5f3ee",
     lineage="Swiss International Style posters (Josef Muller-Brockmann, Tonhalle 'musica viva' series 1950s-70s): "
             "red + black on white, grid and Akzidenz-Grotesk. Curated.",
     sources=["https://en.wikipedia.org/wiki/Josef_M%C3%BCller-Brockmann"],
     suits="line-art/small multiples: black data, red boundary or single highlighted curve.",
     caveat="Spot colours only; keep red for one semantic role.")
_add(name="memphis", kind="categorical", family="design", title="Memphis Group",
     stops=["#f7a6c0", "#18a4a0", "#f7d23e", "#2b3990", "#ee4c3a", "#8ccf5a", "#111111"],
     lineage="Memphis Group (Ettore Sottsass, Milan 1981): candy pastels + primaries + black squiggles. Curated.",
     sources=["https://en.wikipedia.org/wiki/Memphis_Group"],
     suits="playful categorical basins; confetti-like scatter plots of attractors.",
     caveat="Deliberately clashing; several pairs collide under CVD. Declared aesthetic only.")
_add(name="rothko", kind="sequential", family="painting", title="Rothko colour field (maroon->orange)",
     stops=["#16080a", "#3f0d12", "#7a1b16", "#b3371b", "#df6a25", "#f0a54a", "#f5d59a"],
     lineage="Evocation of Mark Rothko's warm fields (e.g. 'Orange and Yellow' 1956; Seagram murals' maroons). Curated.",
     sources=["https://www.moma.org/artists/5047"],
     suits="dark-ground densities with a glowing core; soft-edged fields (smooth Lyapunov interiors).",
     caveat="Very close to lajolla/inferno in L*; distinctive by its oxblood low end.")
_add(name="morandi", kind="sequential", family="painting", title="Morandi muted",
     stops=["#4a4744", "#716c64", "#958c7f", "#b3a898", "#cdc2b2", "#e2dacd", "#f1ede5"],
     lineage="Giorgio Morandi still lifes: dusty greys, putty, bone. Curated.",
     sources=["https://www.moma.org/artists/4102"],
     suits="quiet sequential plates where form (relief, hachure) carries the structure.",
     caveat="Low chroma, L* 30->94: fine, but faint differences vanish on screens with poor gamma.")
_add(name="morandi_categorical", kind="categorical", family="painting", title="Morandi muted set",
     stops=["#a39e93", "#c9b1a0", "#8d9b8f", "#b8a3a8", "#7f8a91", "#d8cfc4"],
     lineage="Morandi-style desaturated set. Curated.", sources=["https://www.moma.org/artists/4102"],
     suits="gentle basin maps with few classes; overlays under black line-art.",
     caveat="Min pairwise dE2000 is small by design: use with outlines.")
_add(name="klimt_gold", kind="sequential", family="painting", title="Klimt gold",
     stops=["#120d05", "#3a2a0c", "#6f5316", "#a8862c", "#d4b24f", "#ecd68c", "#faf0cf"],
     lineage="Gustav Klimt's golden phase (The Kiss, 1907-08; Adele Bloch-Bauer I, 1907): gold leaf on dark. "
             "Curated; MetBrewer 'Klimt' (#df9ed4 #c93f55 #eacc62 #469d76 #3c4b99 #924099) for accents.",
     sources=_mb + ["https://en.wikipedia.org/wiki/The_Kiss_(Klimt)"],
     suits="dark-ground densities (bifurcation atlas, spectrogram harmonics) that should shimmer.",
     caveat="Olive mid-tones can look dirty on low-quality displays; uniform in L*.")
_add(name="klimt_categorical", kind="categorical", family="painting", title="Klimt (MetBrewer)",
     stops=["#df9ed4", "#c93f55", "#eacc62", "#469d76", "#3c4b99", "#924099"], lineage="MetBrewer 'Klimt'.",
     sources=_mb, suits="jewel-like basin classes on dark or gold ground.", caveat="Not colourblind-safe per MetBrewer.")
_add(name="morris", kind="categorical", family="textile", title="William Morris 'Strawberry Thief'",
     stops=["#1d2b45", "#a8322d", "#d9a441", "#5e7f4a", "#ede3cf", "#6b4a32"],
     lineage="William Morris 'Strawberry Thief' (1883), indigo-discharge block print with madder red and weld yellow. Curated.",
     sources=["https://collections.vam.ac.uk/item/O78889/strawberry-thief-furnishing-fabric-morris-william/"],
     suits="categorical basins on indigo ground; ornament-like renders of intricate boundaries.",
     caveat="Red and green are close in L*: fails under deuteranopia unless outlined.")
_add(name="tam", kind="sequential", family="painting", title="Tam (MetBrewer)",
     stops=["#341648", "#62205f", "#9f2d55", "#bb292c", "#de4f33", "#ef8737", "#ffb242", "#ffd353"],
     lineage="MetBrewer 'Tam' (reversed to run dark->light).", sources=_mb,
     suits="dark-ground sequential with synthwave warmth; escape times.",
     caveat="Colourblind-friendly per MetBrewer; L* speeds up in the orange section.")
_add(name="isfahan", kind="diverging", family="painting", title="Isfahan (MetBrewer Isfahan1)",
     stops=["#4e3910", "#845d29", "#ae8548", "#e3c28b", "#4fb6ca", "#178f92", "#175f5d", "#054544"],
     lineage="MetBrewer 'Isfahan1' (Safavid tilework: turquoise glaze and ochre).", sources=_mb,
     suits="signed fields; almost a ready-made split (dark ends, light middle) in bronze/turquoise.",
     caveat="Jump from #e3c28b to #4fb6ca at the centre: a hard hue seam (useful as a boundary).")
_add(name="okeeffe", kind="diverging", family="painting", title="O'Keeffe (MetBrewer OKeeffe1)",
     stops=["#6b200c", "#973d21", "#da6c42", "#ee956a", "#fbc2a9", "#f6f2ee", "#bad6f9", "#7db0ea", "#447fdd", "#225bb2", "#133e7e"],
     lineage="MetBrewer 'OKeeffe1' (Georgia O'Keeffe desert reds and sky blues).", sources=_mb,
     suits="symmetric signed fields on white; random-field level sets.", caveat="Colourblind-friendly per MetBrewer.")

# ---------------------------------------------------------------------------- D. film / photography
_add(name="kodachrome", kind="categorical", family="film", title="Kodachrome",
     stops=["#c1272d", "#f2b500", "#1f5aa6", "#2f7d3b", "#e8d6b3", "#2a1d16"],
     lineage="Kodachrome (1935-2009) dye-coupler reversal film: dense blacks, saturated primaries. Curated evocation.",
     sources=["https://en.wikipedia.org/wiki/Kodachrome"], suits="saturated categorical basins on dark ground.",
     caveat="Red/green pair collapses under deuteranopia.")
_add(name="portra", kind="sequential", family="film", title="Kodak Portra 400",
     stops=["#2e3539", "#4f5f61", "#7c7f76", "#a88f7c", "#d3aa91", "#efcfb9", "#f8ece2"],
     lineage="Portra: low-contrast colour negative with lifted cool shadows and warm skin highlights. Curated evocation.",
     sources=["https://en.wikipedia.org/wiki/Kodak_Portra"],
     suits="soft sequential plates; smooth interiors (Lyapunov stable regions).",
     caveat="Hue flips cool->warm with little L* help mid-ramp; subtle, not for fine magnitude reading.")
_add(name="cinestill", kind="sequential", family="film", title="CineStill 800T (tungsten + halation)",
     stops=["#061520", "#0c3342", "#1c5e6b", "#4f9a9a", "#b9d8cf", "#fff4e0", "#ff8a5c", "#e4322b"],
     positions=[0, .15, .3, .45, .62, .78, .9, 1],
     lineage="CineStill 800T (Kodak Vision3 500T with remjet removed): tungsten-balanced teal nights and red halation around highlights.",
     sources=["https://cinestillfilm.com/products/800tungsten-high-speed-color-film-35mm-135-36exp"],
     suits="dark-ground densities where the very brightest values should 'bleed' red (bifurcation branches).",
     caveat="Deliberately non-monotone at the top (red is darker than cream): the peak reads as a halo, not as 'more'.")
_add(name="technicolor2", kind="inks", family="film", title="Technicolor two-strip (Process 3)",
     stops=["#e2553b", "#2e9a8e"], ground="#f7f1e6",
     lineage="Technicolor Process 2/3 (1922-1932): two dye-imbibition records, orange-red and blue-green, cemented/imbibed. Curated.",
     sources=["https://en.wikipedia.org/wiki/Technicolor", "https://filmcolors.org/timeline-entry/1244/"],
     suits="two-sided fields as two dye layers; flesh-and-sea look.",
     caveat="Only two primaries: no true blue or yellow; overlap gives a brown-black.")
_add(name="autochrome", kind="inks", family="film", title="Autochrome Lumiere",
     stops=["#d0583a", "#5c963e", "#584aa0"], ground="#1a1712",
     lineage="Autochrome (Lumiere 1907): potato-starch grains dyed orange-red, green, blue-violet as a random additive mosaic. Curated.",
     sources=["https://en.wikipedia.org/wiki/Autochrome_Lumi%C3%A8re"],
     suits="stochastic 3-colour dithering of densities; grainy additive renders.",
     caveat="Additive on dark: render as random grain assignment, not as a smooth map.")

# ---------------------------------------------------------------------------- E. nature
_add(name="aurora", kind="sequential", family="nature", title="Aurora borealis",
     stops=["#03081a", "#081f3a", "#0b4a52", "#11845f", "#35c27b", "#9cf0a6", "#effbe0"],
     lineage="Auroral oxygen green line (557.7 nm) over night sky; red 630 nm and N2+ violet above. Curated.",
     sources=["https://en.wikipedia.org/wiki/Aurora#Colours"],
     suits="dark-ground densities, spectrograms, cool half of a split.",
     caveat="Green mid-ramp is bright and saturated: very legible, but red/green CVD viewers lose some separation vs ember.")
_add(name="bioluminescence", kind="sequential", family="nature", title="Bioluminescent sea",
     stops=["#010409", "#031a33", "#054b70", "#0791a8", "#35d0d0", "#a8f6ee", "#f2fffd"],
     lineage="Dinoflagellate (Noctiluca) blue-cyan glow, emission ~475 nm. Curated.",
     sources=["https://en.wikipedia.org/wiki/Bioluminescence"],
     suits="sparse bright structures on black: bifurcation branches, harmonic lattices.",
     caveat="Uniform and CVD-safe (blue axis). Greyscale-honest.")
_add(name="verdigris", kind="diverging", family="nature", title="Verdigris / copper",
     stops=["#3a1a0c", "#7c3d1b", "#b8703f", "#e3b48a", "#eee7da", "#9fd3c2", "#43a58f", "#24936e", "#0f4a3e"],
     lineage="Oxidised copper: metal (copper #b87333) to patina (ROKUSYOH 緑青 #24936E, Japanese verdigris pigment). Curated + nipponcolors.",
     sources=["https://en.wikipedia.org/wiki/Verdigris", "https://nipponcolors.com/#rokusyoh"],
     suits="symmetric signed fields; as split halves see PAIRINGS['verdigris_copper'].",
     caveat="Copper/patina differ mainly in hue: deuteranopes see brown vs grey-blue (still separable by b*).")
_add(name="malachite", kind="sequential", family="nature", title="Malachite",
     stops=["#04221a", "#0b4430", "#136a47", "#23925f", "#5bbb86", "#aee0bd", "#eaf6ee"],
     lineage="Malachite banding (Cu2CO3(OH)2): concentric green bands. Curated.",
     sources=["https://en.wikipedia.org/wiki/Malachite"], suits="sequential density; banded mineral renders of level sets.",
     caveat="Single hue, uniform.")
_add(name="lapis", kind="categorical", family="nature", title="Lapis lazuli + pyrite",
     stops=["#0f1a4a", "#26619c", "#6d8fcb", "#d4af37", "#efe6d2"],
     lineage="Lapis lazuli (lazurite blue, pyrite gold flecks, calcite white); ultramarine pigment source. Curated.",
     sources=["https://en.wikipedia.org/wiki/Lapis_lazuli"], suits="few-class basins; gold for the rare class.",
     caveat="Blue/gold is the safest CVD axis; the two blues need lightness separation (they have it).")
_add(name="deep_sea", kind="sequential", family="nature", title="Deep sea (photic zones)",
     stops=["#000510", "#001733", "#00305c", "#0b5487", "#2e86b0", "#7cc0d8", "#d2f0f7"],
     lineage="Ocean light attenuation: midnight/twilight/sunlit zones. Curated.",
     sources=["https://oceanexplorer.noaa.gov/facts/light-travel.html"], suits="depth-like sequential values, dark ground.",
     caveat="Very close to cmocean 'ice'/Crameri 'oslo': uniform, CVD-safe.")
_add(name="ember", kind="sequential", family="nature", title="Ember / lava",
     stops=["#070202", "#300806", "#6e140a", "#b3300e", "#e5641c", "#f9a444", "#ffe2a8"],
     lineage="Blackbody-like glow of cooling lava. Curated.", sources=["https://en.wikipedia.org/wiki/Black-body_radiation"],
     suits="dark-ground densities; warm half of aurora/ember split.", caveat="Near 'cet_fire'/'inferno' but redder; uniform.")

# ---------------------------------------------------------------------------- F. digital
_add(name="synthwave", kind="sequential", family="digital", title="Synthwave sunset",
     stops=["#0d0221", "#2b0a4f", "#6a1270", "#b3246f", "#ee4d63", "#ff8e53", "#ffd76e"],
     lineage="1980s retro-futurist sunset gradients (Outrun/synthwave album art). Curated.",
     sources=["https://en.wikipedia.org/wiki/Synthwave"], suits="dark-ground densities, escape times, neon heroes.",
     caveat="Uniform-ish but chroma very high; magenta section speeds up in dE.")
_add(name="vaporwave", kind="categorical", family="digital", title="Vaporwave",
     stops=["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff", "#fffb96"],
     lineage="Vaporwave/aesthetic palette as circulated on colour sites (2010s): neon pink, cyan, mint, violet, pale yellow.",
     sources=["https://en.wikipedia.org/wiki/Vaporwave", "https://www.color-hex.com/color-palette/10223"],
     suits="loud categorical basins on #1b1035 ground.", caveat="All light (L* 70-97): needs a dark ground; poor as sequence.")
_add(name="crt_green", kind="sequential", family="digital", title="CRT P1 green phosphor",
     stops=["#000400", "#002a06", "#00590f", "#00911c", "#20d13a", "#8cff9a", "#e6ffe9"],
     lineage="P1 phosphor (peak ~525 nm), monochrome terminals (IBM 3270, Apple II monitors). Curated.",
     sources=["https://en.wikipedia.org/wiki/Phosphor#Standard_phosphor_types"],
     suits="glowing line-art and densities on black (bitplanes, spectrograms, oscilloscope traces).",
     caveat="Monochrome and uniform; highly readable. Add gaussian bloom for authenticity (declared).")
_add(name="crt_amber", kind="sequential", family="digital", title="CRT P3 amber phosphor",
     stops=["#050200", "#2a1500", "#5c3000", "#9a5600", "#e08a00", "#ffb000", "#ffe7b5"],
     lineage="P3 amber phosphor (~602 nm), 1980s terminals (DEC, IBM 5151 amber). Curated.",
     sources=["https://en.wikipedia.org/wiki/Phosphor#Standard_phosphor_types"],
     suits="as crt_green; warmer. Quantization bitplanes.", caveat="Monochrome, uniform.")
_add(name="eink16", kind="categorical", family="digital", title="E-ink 16-level greyscale",
     stops=[rgb2hex(np.array([0.12, 0.12, 0.12]) + (np.array([0.95, 0.94, 0.90]) - 0.12) * i / 15) for i in range(16)],
     lineage="E Ink Carta 4-bit greyscale waveforms: 16 levels, paper-like off-white. Used as a quantized sequential map.",
     sources=["https://www.eink.com/tech/detail/How_it_works"],
     suits="honest quantized sequential (16 declared bands) - fits int4 quantization projects literally.",
     caveat="Bands are real (16 levels) and intended; don't use where false contours would mislead.")

# ---------------------------------------------------------------------------- G. astronomy
_add(name="hubble_sho", kind="diverging", family="astronomy", title="Hubble palette (SHO)",
     stops=["#0f2a33", "#155e63", "#2e9a93", "#8fd1c2", "#f4ecd2", "#f0c064", "#d98a26", "#9c4f16", "#3c1a08"],
     lineage="Narrowband 'Hubble palette': [S II, H-alpha, O III] -> (R, G, B) as in 'Pillars of Creation' (1995). "
             "Processed images read teal (O III) vs gold (S II/H-alpha). Curated.",
     sources=["https://esahubble.org/images/heic1501a/", "https://en.wikipedia.org/wiki/Pillars_of_Creation"],
     suits="signed fields: gold vs teal; very popular split pair.",
     caveat="Teal/gold is roughly along the CVD-safe blue-yellow axis: good for deutan/protan viewers.")
_add(name="star_classes", kind="categorical", family="astronomy", title="Stellar spectral classes OBAFGKM",
     stops=["#9bb0ff", "#aabfff", "#cad7ff", "#f8f7ff", "#fff4ea", "#ffd2a1", "#ffcc6f"],
     lineage="Mitchell Charity, 'What color are the stars?' (blackbody sRGB, D65 white) for O B A F G K M.",
     sources=["http://www.vendian.org/mncharity/dir3/starcolor/"],
     suits="bright points on black (attractor scatter coloured by a sequential 'temperature'); gentle sequential.",
     caveat="All very light (L* 72-97) and low chroma: only works on a dark ground; tiny steps between A/F/G.")

# ---------------------------------------------------------------------------- H. textiles / natural dyes
_add(name="madder", kind="sequential", family="textile", title="Madder root dye",
     stops=["#2a0b0b", "#5a1a17", "#8f2a22", "#bd4a36", "#d9806a", "#ecb9a6", "#f7e6dc"],
     lineage="Madder (Rubia tinctorum, alizarin/purpurin) on wool: Turkey red, pink to brick with mordant. Curated.",
     sources=["https://en.wikipedia.org/wiki/Rubia_tinctorum"], suits="warm sequential; the diverged half of indigo/madder.",
     caveat="Uniform; pairs with indigo along a red-blue axis that survives deuteranopia by lightness.")
_add(name="weld", kind="sequential", family="textile", title="Weld yellow dye",
     stops=["#2b2408", "#5c4c0f", "#8e7a1e", "#bfa434", "#dcc760", "#efe29d", "#faf4d8"],
     lineage="Weld (Reseda luteola, luteolin), the brightest historic European yellow; overdyed with woad for Lincoln green. Curated.",
     sources=["https://en.wikipedia.org/wiki/Reseda_luteola"], suits="warm-green sequential; accent category.",
     caveat="Olive low end; uniform.")
_add(name="kente", kind="categorical", family="textile", title="Kente (Asante/Ewe)",
     stops=["#f2b705", "#0a7b3e", "#c8102e", "#141414", "#1f4e9a"],
     lineage="Kente strip-woven silk/cotton (Ghana): gold, green, red, black, blue with documented symbolic meanings. Curated.",
     sources=["https://en.wikipedia.org/wiki/Kente_cloth", "https://africa.si.edu/exhibits/kente/"],
     suits="bold categorical basins; strip/weave compositions of small multiples.",
     caveat="Red/green collapse under deuteranopia; black separates everything by lightness.")
_add(name="shibori_cyclic", kind="cyclic", family="textile", title="Shibori indigo (cyclic)",
     stops=["#0f2540", "#2e5c8a", "#a9c7de", "#f3f1e8", "#8fb3cf", "#1f4a78"],
     lineage="Indigo resist-dye repeats (arashi/itajime); dark->white->dark cycle. Curated using KON 紺.",
     sources=["https://en.wikipedia.org/wiki/Shibori"],
     suits="cyclic phase where orientation +-pi should look the same (gradient direction mod pi). ",
     caveat="Two-fold symmetric lightness cycle: angle theta and theta+pi look similar - use only for axial (mod pi) data.")
_add(name="verdigris_cyclic", kind="cyclic", family="nature", title="Copper patina (cyclic)",
     stops=["#6b3419", "#c07a45", "#e9d3b4", "#7cc2ad", "#24936e", "#1c4f52", "#4a2a3a"],
     lineage="Copper -> patina -> oxidised dark cycle. Curated.", sources=["https://en.wikipedia.org/wiki/Verdigris"],
     suits="cyclic phase (full 2pi) with a warm/cool hemisphere split.", caveat="Lightness not constant: one bright pole at the cream.")


def list_schemes(kind=None, family=None):
    return [s for s in SCHEMES.values() if (kind is None or s.kind == kind) and (family is None or s.family == family)]


# ============================================================================ split-at-the-boundary
def as_cmap(c, n=256):
    """Colormap from a Colormap, a registered name, a SCHEMES/PAIRINGS side key, or a list of hex stops."""
    if isinstance(c, Colormap):
        return c
    if isinstance(c, (list, tuple)):
        return lab_ramp(list(c), n=n, uniform=True)
    if isinstance(c, str):
        if c in SCHEMES:
            return SCHEMES[c].cmap(n)
        return mpl.colormaps[c]
    raise TypeError(c)


# Each side: stops from the SEAM (at the boundary) outward.
SIDES = {
    # Sohl-Dickstein Spectral halves (exact matplotlib Spectral sub-ranges)
    "spectral_purple": sub_cmap("Spectral", 1.0, 0.5),   # #5e4fa2 -> blue -> green -> pale yellow
    "spectral_red": sub_cmap("Spectral", 0.0, 0.5),      # #9e0142 -> orange -> pale yellow
}

PAIRINGS: dict[str, dict] = {}


def _pair(name, neg, pos, title, lineage, ground=None):
    PAIRINGS[name] = dict(neg=neg, pos=pos, title=title, lineage=lineage, ground=ground)


_pair("sd_spectral", SIDES["spectral_purple"], SIDES["spectral_red"], "Sohl-Dickstein Spectral (reference)",
      "matplotlib Spectral, split at 0.5; converged=purple half, diverged=red half; rank-normalized per side.")
_pair("verdigris_copper",
      ["#0e3a33", "#1c6b5b", "#3f9a84", "#86c8b2", "#d5ece2"],
      ["#3d1a0b", "#7e3a18", "#b86f3c", "#deaa7c", "#f5e3cc"],
      "Verdigris / copper", "Patina (ROKUSYOH) against bare metal; both sides earthy, seam = dark oxide.")
_pair("indigo_madder",
      ["#08192d", "#0b346e", "#2f6aa3", "#7fb0d2", "#dcebf2"],
      ["#2a0b0b", "#7a1f1d", "#b8473a", "#e19a82", "#f6e2d8"],
      "Indigo / madder", "The two great natural dyes (aizome KACHI->KAMENOZOKI vs Rubia tinctorum). Morris textiles.")
_pair("aurora_ember",
      ["#03122a", "#0b4a52", "#1b9a70", "#76dca0", "#e3fbdc"],
      ["#1c0503", "#6e140a", "#c4401a", "#f59a3e", "#ffe6b0"],
      "Aurora / ember", "Cold night-sky oxygen green vs blackbody lava glow.")
_pair("cyanotype_vandyke",
      ["#07172c", "#153f6e", "#3c74a6", "#9cc0dd", "#eef3f6"],
      ["#1b0f08", "#4a2e1c", "#7e573a", "#bc9a78", "#f1e6d6"],
      "Cyanotype / Van Dyke", "Two iron-process prints; a darkroom diptych. Low chroma, archival.")
_pair("hubble_sho",
      ["#061c22", "#0f4e58", "#2a8c8c", "#8fd0c2", "#eaf6ef"],
      ["#221003", "#6e3a0c", "#c0761f", "#edbb5e", "#fff2cc"],
      "Hubble SHO teal / gold", "O III teal against S II/H-alpha gold (Pillars of Creation processing).")
_pair("klimt_lapis",
      ["#0b0f2e", "#1e2a6e", "#3e55a8", "#8e9fd6", "#e6eaf7"],
      ["#1c1405", "#5e4410", "#a77f24", "#ddbf5e", "#f8edc4"],
      "Lapis / Klimt gold", "Ultramarine against gold leaf (Byzantine mosaic, Klimt's golden phase).")
_pair("synthwave",
      ["#060a2a", "#10307a", "#1f7fc0", "#5fd0ea", "#d8f8ff"],
      ["#1a0626", "#5b0f6b", "#b0247e", "#f2629a", "#ffd8e6"],
      "Synthwave cyan / magenta", "Neon grid vs sunset; VHS-era chromatic aberration.")
_pair("cinestill",
      ["#04121a", "#0f3c4a", "#2b7a86", "#9ccac4", "#eef7f2"],
      ["#1d0403", "#6b0f0b", "#c9281c", "#f28a6a", "#ffe3d6"],
      "CineStill tungsten / halation", "Tungsten-balanced night teal vs the red halation ring of 800T.")
_pair("hokusai_sunset",
      ["#0a2e57", "#295384", "#5a97c1", "#95c9c3", "#e7f0e2"],
      ["#3a160c", "#6d2f20", "#b75347", "#e09351", "#f5e2b8"],
      "Hokusai wave / Hiroshige sunset", "MetBrewer Hokusai3 blues vs Hokusai1/Hiroshige warm ramp.")
_pair("crt_phosphor",
      ["#000400", "#005a10", "#15b030", "#8cff9a", "#eaffec"],
      ["#050200", "#6a3a00", "#d98400", "#ffc24a", "#fff0cc"],
      "CRT green / amber phosphor", "Two terminal phosphors (P1 vs P3) on a black glass seam.")
_pair("malachite_rhodochrosite",
      ["#04221a", "#0f5a3c", "#2f9a66", "#9bd8b0", "#eef8f0"],
      ["#2a0914", "#6e1c35", "#b44a6a", "#e79bb0", "#fbe6ec"],
      "Malachite / rhodochrosite", "Two banded minerals: copper-carbonate green vs manganese-carbonate rose.")
_pair("morandi",
      ["#3e4744", "#62716b", "#8e9c94", "#bfc8c0", "#eef0ea"],
      ["#4a3c38", "#77605a", "#a58d84", "#cdb9ae", "#f3ebe4"],
      "Morandi sage / clay", "Muted still-life pair: the seam is a soft umber, interiors powdery.")
_pair("crameri_bukavu", sub_cmap("cmc.bukavu", 0.5, 0.0), sub_cmap("cmc.bukavu", 0.5, 1.0),
      "Crameri bukavu (light 'coastline' seam)",
      "Crameri's multi-sequential topographic map split at sea level; seam is where both halves are lightest->darkest.")
# bukavu: lower half runs dark sea -> light coast, upper half dark lowland -> white peaks.
# sub_cmap(0.5 -> 0) starts at the LIGHT coast, so this pairing has a light seam; kept as an honest
# cartographic counter-example (the coast is light in every atlas).
_pair("riso_pink_blue",
      ["#123a78", "#3255a4", "#62a8e5", "#bcdcf4", "#f4efe3"],
      ["#6a1742", "#c02c7e", "#ff48b0", "#f9b0d6", "#f4efe3"],
      "Riso blue / fluo pink (contone)", "Continuous-tone version of the two-ink zine pair, both halves ending at cream paper.")


def split_cmap(neg_cmap, pos_cmap, n=512, name="split", seam="dark"):
    """One Colormap on [0,1] with the seam at 0.5.

    neg_cmap / pos_cmap are ramps from the seam outward (index 0 = colour AT the boundary).
    t < 0.5 samples neg_cmap at (0.5 - t) * 2, t > 0.5 samples pos_cmap at (t - 0.5) * 2.
    seam='light' reverses both ramps (pale boundary, dark interiors).
    Use with a value v in [-1, 1] as cmap((v + 1) / 2), e.g. v = signed_rank_normalize(x).
    """
    a, b = as_cmap(neg_cmap), as_cmap(pos_cmap)
    t = np.linspace(0, 1, n)
    u = np.abs(t - 0.5) * 2
    if seam == "light":
        u = 1 - u
    cols = np.where((t < 0.5)[:, None], a(u)[:, :3], b(u)[:, :3])
    return ListedColormap(cols, name=name)


def signed_rank_normalize(x, near_boundary="small", pastel=0.75, ref=None):
    """Sohl-Dickstein-style per-side histogram equalisation of a signed field.

    Returns v in [-pastel, pastel] with sign(v) = sign(x) (x == 0 counts as positive) and
    |v| = pastel * (rank distance from the boundary, 0 = closest).

    near_boundary='small' : small |x| is near the boundary (Lyapunov exponent, f - c, level sets).
    near_boundary='large' : large |x| is near the boundary (his trainability measure, where the
                            slowest convergence/divergence sits at the fractal edge).
    pastel=0.75 reproduces his buffer=0.25 (the palest quarter of each half is never used, so
    the two light ends stay distinguishable).
    ref: optional reference array for the ranks (e.g. for zoom sequences use frame 0 or pooled data).
    Non-finite entries return NaN.
    Equivalence: render_split(M, 'sd_spectral', near_boundary='large') == Spectral(cdf_img(M)) from
    his colab (checked in test_palettes.py to <= 1 LUT step).
    """
    x = np.asarray(x, float)
    r = x[np.isfinite(x)] if ref is None else np.asarray(ref, float)[np.isfinite(ref)]
    out = np.full(x.shape, np.nan)
    fin = np.isfinite(x)
    for sgn, sel_x, sel_r in ((-1, fin & (x < 0), r < 0), (1, fin & (x >= 0), r >= 0)):
        a = np.sort(np.abs(r[sel_r]))
        if a.size == 0:
            continue
        if a.size > 1:
            ax = np.abs(x[sel_x])
            if ref is None:  # exact knots: average rank for ties (== his np.interp for unique values)
                lo = np.searchsorted(a, ax, "left")
                hi = np.searchsorted(a, ax, "right") - 1
                q = (lo + hi) / 2 / (a.size - 1)
            else:
                q = np.interp(ax, a, np.linspace(0, 1, a.size))
        else:
            q = np.zeros(sel_x.sum())
        d = q if near_boundary == "small" else 1 - q
        out[sel_x] = sgn * pastel * d
    return out


def render_split(x, pairing="sd_spectral", near_boundary="small", pastel=0.75, seam="dark",
                 nan_color="#000000", ref=None, neg=None, pos=None):
    """Signed field -> RGB float image (H, W, 3) using a split pairing (name in PAIRINGS) or explicit neg/pos ramps."""
    p = PAIRINGS.get(pairing, {}) if isinstance(pairing, str) else {}
    a = as_cmap(neg if neg is not None else p["neg"])
    b = as_cmap(pos if pos is not None else p["pos"])
    v = signed_rank_normalize(x, near_boundary=near_boundary, pastel=pastel, ref=ref)
    u = np.abs(np.nan_to_num(v))
    if seam == "light":
        u = pastel - u
    img = np.where((np.asarray(x) < 0)[..., None], a(u)[..., :3], b(u)[..., :3])
    img[~np.isfinite(v)] = hex2rgb(nan_color)
    return img


def rank_normalize(x, ref=None):
    """Histogram-equalise to [0,1] (ignores non-finite)."""
    x = np.asarray(x, float)
    r = x[np.isfinite(x)] if ref is None else np.asarray(ref, float)[np.isfinite(ref)]
    a = np.sort(r)
    out = np.full(x.shape, np.nan)
    f = np.isfinite(x)
    out[f] = np.interp(x[f], a, np.linspace(0, 1, a.size))
    return out


def overprint(coverages, inks, paper=RISO_PAPER):
    """Multiplicative (subtractive-ish) overprint: coverages list of (H,W) in [0,1], inks list of hex."""
    img = np.ones(np.shape(coverages[0]) + (3,)) * hex2rgb(paper)
    for c, ink in zip(coverages, inks):
        img = img * (1 - np.asarray(c)[..., None] * (1 - hex2rgb(ink)))
    return img


# ============================================================================ diagnostics
def lightness_profile(cmap, n=256):
    cm = as_cmap(cmap)
    rgb = cm(np.linspace(0, 1, n))[:, :3] if not (isinstance(cm, ListedColormap) and cm.N <= 32) else np.array(cm.colors)[:, :3]
    return rgb_to_lab(rgb)[:, 0]


def metrics(scheme: Scheme | str, n=256):
    """Perceptual diagnostics used in COLOR_SCHEMES.md.

    continuous maps: CIELAB L* start/end/min/max; CAM02-UCS lightness J' reversals (direction changes
      > 3 units); perceptual speed CV (std/mean of per-step CAM02-UCS dE over n samples) and spike ratio
      (max/median step over the central 92%: > 3 means a visible false band / hard seam); deuteranopia retention (UCS path
      length under simulated deuteranopia / normal); greyscale retention (|dJ'| path / dE path).
    categorical/inks: min pairwise dE2000 normal, under deuteranopia/protanopia, and min |dL*|.
    """
    s = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    out = dict(name=s.name, kind=s.kind)
    if s.kind in ("categorical", "inks"):
        cols = [hex2rgb(h) for h in s.sample_hex(16)]
        if s.kind == "inks" and s.ground:
            cols = cols + [hex2rgb(s.ground)]
        rgb = np.array(cols)
        lab = rgb_to_lab(rgb)
        iu = np.triu_indices(len(rgb), 1)
        out["min_dE"] = float(deltaE2000(lab[:, None], lab[None])[iu].min()) if len(rgb) > 1 else float("nan")
        for k in ("deuteranopia", "protanopia"):
            lk = rgb_to_lab(simulate_cvd(rgb, k))
            out["min_dE_" + k[:4]] = float(deltaE2000(lk[:, None], lk[None])[iu].min()) if len(rgb) > 1 else float("nan")
        out["min_dL"] = float(np.abs(lab[:, None, 0] - lab[None, :, 0])[iu].min()) if len(rgb) > 1 else float("nan")
        out["L"] = [float(v) for v in lab[:, 0]]
        return out
    cm = s.cmap(n)
    rgb = cm(np.linspace(0, 1, n))[:, :3]
    L = rgb_to_lab(rgb)[:, 0]
    ucs = rgb_to_cam02ucs(rgb)
    J = ucs[:, 0]
    dE = np.linalg.norm(np.diff(ucs, axis=0), axis=-1)
    out.update(L0=float(L[0]), L1=float(L[-1]), Lmin=float(L.min()), Lmax=float(L.max()),
               J_reversals=count_reversals(J), speed_cv=float(dE.std() / dE.mean()),
               spike=float(dE[n // 25: -n // 25].max() / np.median(dE)), path_ucs=float(dE.sum()))
    ud = rgb_to_cam02ucs(simulate_cvd(rgb, "deuteranopia"))
    out["deut_retention"] = float(np.linalg.norm(np.diff(ud, axis=0), axis=-1).sum() / dE.sum())
    out["grey_retention"] = float(np.abs(np.diff(J)).sum() / dE.sum())
    return out


def count_reversals(y, thresh=3.0):
    """Number of direction changes of a profile, ignoring wiggles smaller than `thresh` (zig-zag filter)."""
    direction, ext, rev = 0, y[0], 0
    for v in y[1:]:
        if direction == 0:
            if v > ext + thresh:
                direction, ext = 1, v
            elif v < ext - thresh:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif v < ext - thresh:
                rev, direction, ext = rev + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v > ext + thresh:
                rev, direction, ext = rev + 1, 1, v
    return int(rev)


# ============================================================================ registration
def register_all(overwrite=True):
    for s in SCHEMES.values():
        if s.lib:
            continue
        cm = s.cmap()
        for c, suffix in ((cm, ""), (cm.reversed(), "_r")):
            nm = PREFIX + s.name + suffix
            c.name = nm
            try:
                mpl.colormaps.register(c, name=nm, force=overwrite)
            except Exception:
                pass
    for k, p in PAIRINGS.items():
        cm = split_cmap(p["neg"], p["pos"], name=PREFIX + "split." + k)
        try:
            mpl.colormaps.register(cm, name=PREFIX + "split." + k, force=overwrite)
        except Exception:
            pass


def get(name, n=256):
    """Colormap by scheme name, 'split.<pairing>', or any registered matplotlib name."""
    if name in SCHEMES:
        return SCHEMES[name].cmap(n)
    if name.startswith("split."):
        p = PAIRINGS[name[6:]]
        return split_cmap(p["neg"], p["pos"], name=PREFIX + name)
    return mpl.colormaps[name]


register_all()

if __name__ == "__main__":
    print(len(SCHEMES), "schemes;", len(PAIRINGS), "split pairings")
    for s in SCHEMES.values():
        m = metrics(s)
        print(s.name, s.kind, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k not in ("name", "kind", "L")})
