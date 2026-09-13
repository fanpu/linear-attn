"""One visual system for every static figure, animation and widget in this post.

Light plates sit on the post's paper colour; hero pieces use a deep blue-black "night" surface.
Palettes were checked with the dataviz skill's validator (CVD separation passes; the dark
palettes are intentionally brighter than the dashboard lightness band because they glow on near-black).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

PAPER = "#fcfbf8"
INK = "#1d1d1f"
MUTED = "#6b6b70"
RULE = "#e6e2da"
ACCENT = "#b5452b"

NIGHT = "#0c0f17"
NIGHT_INK = "#ecebe6"
NIGHT_MUTED = "#8b8d98"
NIGHT_RULE = "#252a36"

# Modes, ordered strongest (learned first) -> weakest (learned last).
MODES_LIGHT = ["#e0952a", "#d2553a", "#a23f84", "#4a45a8", "#1a7fa6"]
MODES_DARK = ["#ffc15e", "#ff7a59", "#e16ab8", "#8a86f5", "#4fc3d9"]

# Item categories (birds, fish, trees, flowers).
CATS = ["bird", "fish", "tree", "flower"]
CAT_LIGHT = dict(zip(CATS, ["#e08a1e", "#2f7fd6", "#23845a", "#c94f96"]))
CAT_DARK = dict(zip(CATS, ["#ffb347", "#4f9df7", "#2e9e6e", "#e573b5"]))

THEORY_LIGHT = INK
THEORY_DARK = "#ffffff"

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
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.color": rule, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "mathtext.fontset": "custom",
        "mathtext.rm": SANS, "mathtext.it": f"{SANS}:italic", "mathtext.bf": f"{SANS}:bold",
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlecolor": ink,
    })
    return dict(bg=bg, ink=ink, muted=muted, rule=rule, modes=MODES_DARK if dark else MODES_LIGHT,
                cats=CAT_DARK if dark else CAT_LIGHT, theory=THEORY_DARK if dark else THEORY_LIGHT)


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(path)
