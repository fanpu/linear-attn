"""Visual system for the lazy/rich + muP post.

Two regimes, two temperatures: LAZY is cold blue (frozen features), RICH is ember orange (features on the move).
Hero pieces sit on a deep ink "night" surface where the two classes glow amber and cyan (intentionally brighter
than the dashboard lightness band). Width ramps are single-hue violet, depth ramps single-hue teal.
Palettes checked with the dataviz skill's validator (light categorical + ordinal ramps pass).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

PAPER = "#fcfbf8"
INK = "#1d1d1f"
MUTED = "#6b6b70"
RULE = "#e6e2da"

LAZY = "#2f6db5"
RICH = "#e0561f"

NIGHT = "#070912"
NIGHT_INK = "#ecebe6"
NIGHT_MUTED = "#8a8d9c"
NIGHT_RULE = "#22263a"
POS = "#ffb454"  # class +1 on dark
NEG = "#58c4f5"  # class -1 on dark

WIDTH_RAMP = ["#ad92e0", "#8561cf", "#5e37b0", "#361a7d"]  # 128, 256, 512, 1024
DEPTH_RAMP = ["#6fb8a7", "#3f9e8a", "#1d7564", "#0a4a3f"]  # 2, 4, 8, 16

SANS = "Ubuntu Sans"
for f in font_manager.findSystemFonts():
    if "UbuntuSans" in f.replace(" ", "") or "Ubuntu" in f:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass


def use(theme="light"):
    dark = theme == "dark"
    bg, ink, muted, rule = (NIGHT, NIGHT_INK, NIGHT_MUTED, NIGHT_RULE) if dark else (PAPER, INK, MUTED, RULE)
    mpl.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "font.family": "sans-serif", "font.sans-serif": [SANS, "Ubuntu", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
        "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": rule, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.color": rule, "grid.linewidth": 0.7,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "mathtext.fontset": "dejavusans",
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlecolor": ink,
    })
    return dict(bg=bg, ink=ink, muted=muted, rule=rule)


def save(fig, path, dpi=200, pad=0.25):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(path)
