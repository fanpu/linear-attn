"""One visual system for the double-descent post: static plates, animations and widgets share these tokens.

Light plates sit on the post's paper colour. Hero pieces use an ink-violet "night" surface where the fit glows.
Semantic roles (validated with the dataviz validator, light surface #fcfbf8: CVD worst adjacent dE 14.3):
    BIAS  = blue      (what the model can't express)
    VAR   = crimson   (what the noise does to it)  -- the villain of the post
    GOLD  = threshold marker (p = n), always paired with a label
Dark variants are deliberately brighter than the dashboard lightness band because they glow on near-black.
Sequential families (SNR, lambda, width) use one-hue ramps; heatmaps use cmcrameri lajolla/batlow-like maps.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

PAPER = "#fcfbf8"
INK = "#1f1d24"
MUTED = "#6e6a75"
RULE = "#e7e2dc"

NIGHT = "#0e0b12"
NIGHT_INK = "#f2ede6"
NIGHT_MUTED = "#8f8898"
NIGHT_RULE = "#2a2431"

BIAS = "#2f6db5"
VAR = "#d1495b"
GOLD = "#e09f3e"
BIAS_D = "#6aa8f0"
VAR_D = "#ff6b7d"
GOLD_D = "#ffc45e"
DATA_D = "#fff6e8"

# one-hue ramps (light -> dark) for ordered families
def ramp(hex_dark, n, lo=0.28, hi=1.0, paper=PAPER):
    c = np.array(to_rgb(hex_dark)); p = np.array(to_rgb(paper))
    return [tuple(p + (c - p) * t) for t in np.linspace(lo, hi, n)]

INDIGO = "#3b2f8f"
TEAL = "#146c6c"

# "heat" for the hero: calm violet -> hot coral -> white-gold (semantic heat, used with a labelled scale)
HEAT = LinearSegmentedColormap.from_list("dd_heat", ["#5b6cf0", "#9d5cf0", "#ff5d8f", "#ff9a5a", "#ffe6a8"])

SANS = "Nimbus Sans"
for f in font_manager.findSystemFonts():
    if "NimbusSans" in f or "Nimbus Sans" in f:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass


def use(theme="light"):
    dark = theme == "dark"
    bg, ink, muted, rule = (NIGHT, NIGHT_INK, NIGHT_MUTED, NIGHT_RULE) if dark else (PAPER, INK, MUTED, RULE)
    mpl.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "font.family": "sans-serif", "font.sans-serif": [SANS, "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
        "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": rule, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "axes.grid": False, "grid.color": rule, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "mathtext.fontset": "custom",
        "mathtext.rm": SANS, "mathtext.it": f"{SANS}:italic", "mathtext.bf": f"{SANS}:bold",
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlecolor": ink,
    })
    return dict(bg=bg, ink=ink, muted=muted, rule=rule, dark=dark,
                bias=BIAS_D if dark else BIAS, var=VAR_D if dark else VAR, gold=GOLD_D if dark else GOLD)


def save(fig, path, dpi=200, **kw):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25, **kw)
    plt.close(fig)
    print(path)
