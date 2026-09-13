"""One visual system for every figure in the post.

Grammar:  theory / closed-form curves are thin INK lines;  measurements are COLOURED markers or lines.
Colour = entity, fixed across the post (validated all-pairs with the dataviz palette checker on #fcfbf8):
    linear attention / LSA  blue    #2a78d6
    softmax attention       orange  #eb6834
    DeltaNet                aqua    #1baf7a   (sub-3:1 contrast -> always direct-labelled)
    Gated DeltaNet          violet  #4a3aa7
Classical algorithms are neutrals: ridge/RLS = ink, GD = warm gray, LMS = light warm gray.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

PAPER = "#fcfbf8"
INK = "#1d1d1f"
MUTED = "#6b6b70"
RULE = "#e6e2da"
C = dict(linear="#2a78d6", softmax="#eb6834", delta="#1baf7a", gdelta="#4a3aa7",
         ridge=INK, gd="#8f887c", lms="#b9b2a6", dmmse="#1d1d1f")
NAMES = dict(linear="Linear attention", softmax="Softmax attention", delta="DeltaNet", gdelta="Gated DeltaNet")

# dark hero system
NIGHT = "#0b0d12"
NIGHT_INK = "#ece8df"
NIGHT_MUTED = "#8b8f99"
NIGHT_RULE = "#23262e"
GLOW = "#7fd4ff"


def use_light():
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "font.family": "Nimbus Sans", "font.size": 11, "mathtext.fontset": "stixsans",
        "axes.edgecolor": "#c9c4ba", "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": RULE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def use_dark():
    mpl.rcParams.update({
        "figure.facecolor": NIGHT, "axes.facecolor": NIGHT, "savefig.facecolor": NIGHT,
        "font.family": "Nimbus Sans", "font.size": 11, "mathtext.fontset": "stixsans", "text.color": NIGHT_INK,
        "axes.edgecolor": NIGHT_RULE, "axes.labelcolor": NIGHT_MUTED, "axes.titlecolor": NIGHT_INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": NIGHT_RULE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": NIGHT_MUTED, "ytick.color": NIGHT_MUTED, "xtick.labelcolor": NIGHT_MUTED, "ytick.labelcolor": NIGHT_MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.solid_capstyle": "round",
    })


def ramp(hex_color, n, lo=0.35, hi=1.0):
    """n lightness steps of one hue, light -> full (for depth L=1..n of one model family)."""
    c = np.array(to_rgb(hex_color)); w = np.array(to_rgb(PAPER))
    return [tuple(w + (c - w) * t) for t in np.linspace(lo, hi, n)]


def diverging_light():
    return LinearSegmentedColormap.from_list("bwr_paper", ["#184f95", "#3987e5", "#cde2fb", "#f3f1ec", "#fbd3c4", "#e34948", "#9c2224"])


def diverging_dark():
    """Zero = the night background, so structure literally emerges from darkness."""
    return LinearSegmentedColormap.from_list("night_div", ["#9fe3ff", "#2e8fd6", "#123a5c", NIGHT, "#5c2413", "#e0632f", "#ffd08a"])


def label_end(ax, x, y, text, color, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, va="center",
                fontsize=kw.pop("fontsize", 10), fontweight=kw.pop("fontweight", "bold"), **kw)
