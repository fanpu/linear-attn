"""Shared visual idioms: log paper (plotter ink), dark observatory, two-ink riso, Spectral."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

STYLES = {
    # K&E-style log paper: cream stock, orange-brown ruling, one blue-black ink
    "paper": dict(bg="#f3eddf", ink="#1c2633", grid_major="#c98a62", grid_minor="#e3c3a8",
                  text="#1c2633", accent="#1c2633", font="DejaVu Serif"),
    "dark": dict(bg="#0a0c11", ink="#e8e2d0", grid_major="#2a3040", grid_minor="#161a24",
                 text="#cfc8b4", accent="#f2b134", font="DejaVu Sans"),
    # two spot inks: Federal Blue + Fluorescent Pink on off-white newsprint
    "riso": dict(bg="#f4f0e6", ink="#3255a4", grid_major="#b9c3dc", grid_minor="#dde2ee",
                 text="#3255a4", accent="#ff48b0", font="DejaVu Sans Mono"),
    "spectral": dict(bg="#fbfaf6", ink="#2b2b2b", grid_major="#d8d4cc", grid_minor="#ece9e2",
                     text="#2b2b2b", accent="#9e0142", font="DejaVu Sans"),
}


def dcolor(style, d, dims):
    """Colour for a manifold dimension d (declared aesthetic: ordinal ramp over the d values)."""
    t = (np.log(d) - np.log(min(dims))) / (np.log(max(dims)) - np.log(min(dims)))
    if style == "paper":
        return STYLES["paper"]["ink"]
    if style == "dark":
        return plt.get_cmap("magma")(0.35 + 0.6 * t)
    if style == "riso":
        # mix of the two inks, pink share grows with d (overprint look)
        b, p = np.array(to_rgb("#3255a4")), np.array(to_rgb("#ff48b0"))
        return tuple((1 - t) * b + t * p)
    if style == "spectral":
        return plt.get_cmap("Spectral_r")(0.02 + 0.96 * t)


def setup(style, figsize, dpi=200):
    S = STYLES[style]
    plt.rcParams.update({"font.family": S["font"], "text.color": S["text"], "axes.labelcolor": S["text"],
                         "xtick.color": S["text"], "ytick.color": S["text"], "axes.edgecolor": S["text"],
                         "mathtext.fontset": "dejavuserif" if style == "paper" else "dejavusans"})
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=S["bg"])
    return fig, S


def log_paper(ax, style, xlim, ylim):
    """Rule an axis like printed log-log paper: decade lines heavy, 2..9 lines light."""
    S = STYLES[style]
    ax.set_facecolor(S["bg"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    for axis, lim, line in ((0, xlim, ax.axvline), (1, ylim, ax.axhline)):
        for e in range(int(np.floor(np.log10(lim[0]))), int(np.ceil(np.log10(lim[1]))) + 1):
            for m in range(1, 10):
                v = m * 10.0 ** e
                if lim[0] <= v <= lim[1]:
                    line(v, color=S["grid_major"] if m == 1 else S["grid_minor"],
                         lw=0.9 if m == 1 else 0.45, zorder=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(S["grid_major"] if style != "dark" else S["grid_major"]); sp.set_linewidth(1.0)
    ax.tick_params(which="both", length=0, labelsize=8)


def riso_misregister(img, dx=3, dy=-2):
    """Offset the pink plate of a rendered RGB image to fake misregistration (declared)."""
    img = img.astype(float)
    bg = np.array(to_rgb(STYLES["riso"]["bg"])) * 255
    pink = np.array(to_rgb("#ff48b0")) * 255
    # pixels closer to pink than blue belong to the pink plate
    blue = np.array(to_rgb("#3255a4")) * 255
    dp = np.linalg.norm(img - pink, axis=-1); db = np.linalg.norm(img - blue, axis=-1)
    mask = (dp < db) & (np.linalg.norm(img - bg, axis=-1) > 40)
    out = img.copy()
    out[mask] = bg
    sh = np.roll(np.roll(mask, dy, 0), dx, 1)
    src = np.roll(np.roll(img, dy, 0), dx, 1)
    # multiply blend (ink overprint)
    out[sh] = out[sh] * src[sh] / 255.0
    return out.clip(0, 255).astype(np.uint8)


def save(fig, path, riso=False):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    im = np.array(Image.open(buf).convert("RGB"))
    if riso:
        im = riso_misregister(im)
    Image.fromarray(im).save(path, optimize=True)
    return im.shape
