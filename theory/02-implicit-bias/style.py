"""One visual system for the post: light explanatory plates + dark cinematic hero pieces.

Semantic colour is fixed across every figure and widget (validated with the dataviz validator):
  L2  / plain GD / kernel regime  -> BLUE
  Linf / sign GD / Adam           -> ORANGE
  L1  / rich regime / sparse      -> AQUA
  spectral / Muon                 -> VIOLET
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- light plates
PAPER = "#fcfbf8"
INK = "#1d1d1f"
INK2 = "#52514e"
MUTED = "#8b8984"
GRID = "#e8e5de"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"

# ---- dark hero
NIGHT = "#0b0d12"
NIGHT2 = "#151923"
STAR = "#e9e6df"     # primary ink on dark
DIM = "#8e929c"      # secondary ink on dark
FAINT = "#2a2f3a"    # hairlines on dark
AMBER = "#cc7a38"    # + class
TEAL = "#1fa597"     # - class
LILAC = "#8b7ae6"    # the max-margin target

# alpha sweep ramp: dark teal (rich, L1) -> light blue (kernel, L2); monotone lightness
ALPHA_CMAP = LinearSegmentedColormap.from_list("rich_kernel", ["#0d4f3c", "#17866a", "#2a8fb0", "#5a9ee6", "#a9c8f2"])
# depth ramp (ordinal, violet family)
DEPTH_COLORS = {1: "#9c92e0", 2: "#6a5bc8", 3: "#3b2d8f", 4: "#221a5c"}

SANS = "Ubuntu Sans"
MONO = "Ubuntu Sans Mono"


def light():
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "font.family": SANS, "font.size": 11, "mathtext.fontset": "stixsans",
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12.5, "axes.titleweight": "medium",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False, "savefig.dpi": 200,
    })


def dark():
    mpl.rcParams.update({
        "figure.facecolor": NIGHT, "axes.facecolor": NIGHT, "savefig.facecolor": NIGHT,
        "font.family": SANS, "font.size": 11, "mathtext.fontset": "stixsans",
        "text.color": STAR, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
        "axes.edgecolor": FAINT, "axes.linewidth": 0.8, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titlelocation": "left",
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False, "savefig.dpi": 200,
    })


def glow_line(ax, x, y, color, lw=2.0, layers=6, spread=5.0, alpha=0.07, **kw):
    """Soft neon glow: a stack of wide translucent strokes under a crisp core line."""
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, lw=lw + spread * i, alpha=alpha, solid_capstyle="round", **kw)
    return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", **kw)


def label_end(ax, x, y, text, color=INK2, dx=4, dy=0, ha="left", va="center", size=10, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color, fontsize=size, **kw)
