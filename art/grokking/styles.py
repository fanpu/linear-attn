"""Shared drawing vocabulary for all grokking renders.  Pure presentation: every number drawn comes
from cache/analysis.npz.  All choices here are declared aesthetic choices (see README)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import colorcet as cc
from cmcrameri import cm as ccm

P = 113

STYLES = {
    # dark ground, edges coloured by residue a with a CYCLIC map (residues are cyclic), additive look
    "nocturne": dict(bg="#07080c", ink="#e8e6df", muted="#8a8a92", grid="#23252d", font="DejaVu Sans",
                     cyclic=cc.cm["CET_C6"], seq=ccm.lajolla_r if hasattr(ccm, "lajolla_r") else "magma",
                     spec_cmap="magma", lw=0.55, alpha=0.9, train="#9bd0ff", test="#ffb36b"),
    # Molnar/Mohr plotter sheet: one ink on paper white
    "plotter": dict(bg="#f5f2ea", ink="#16161a", muted="#6d6a64", grid="#d9d4c8", font="DejaVu Sans Mono",
                    cyclic=None, spec_cmap=None, lw=0.45, alpha=1.0, train="#16161a", test="#16161a"),
    # two-spot risograph: fluorescent pink + blue, multiply blend, deliberate misregistration
    "riso": dict(bg="#f3eee4", ink="#1d3f8f", ink2="#ff4f9a", muted="#7a7468", grid="#ddd5c6", font="DejaVu Sans",
                 cyclic=None, spec_cmap=None, lw=0.6, alpha=1.0, train="#1d3f8f", test="#ff4f9a"),
    # observatory archive plate: sepia on aged card, serif captions, plate frame
    "plate": dict(bg="#e8dec6", ink="#3a2c1e", muted="#7b6a55", grid="#cdbf9f", font="DejaVu Serif",
                  cyclic=None, spec_cmap=None, lw=0.5, alpha=1.0, train="#3a2c1e", test="#9a4a22"),
}


def sepia_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sepia", ["#e8dec6", "#b89b72", "#6b4e2f", "#2a1d10"])


def paper_cmap(ink="#16161a", bg="#f5f2ea"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("paper", [bg, ink])


def star_segments(z, closed=True):
    """z (P,2) -> segments a -> a+1 (the {113/k} star polygon when z is the k-circle)."""
    nxt = np.roll(z, -1, axis=0)
    seg = np.stack([z, nxt], 1)
    return seg if closed else seg[:-1]


def normalize(z):
    """Declared: centre and scale each frame to unit RMS radius (weight decay changes the overall scale)."""
    z = z - z.mean(0)
    return z / np.sqrt((z ** 2).sum(1).mean())


def draw_star(ax, z, style, lw=None, alpha=None, color=None, lim=1.75, residue_colors=True, zorder=2):
    st = STYLES[style]
    seg = star_segments(z)
    lw = st["lw"] if lw is None else lw
    alpha = st["alpha"] if alpha is None else alpha
    if style == "nocturne" and residue_colors and color is None:
        cols = st["cyclic"](np.arange(P) / P)
        lc = LineCollection(seg, colors=cols, linewidths=lw, alpha=alpha, capstyle="round", zorder=zorder)
    else:
        lc = LineCollection(seg, colors=color or st["ink"], linewidths=lw, alpha=alpha, capstyle="round", zorder=zorder)
    ax.add_collection(lc)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    return lc


def fig_canvas(w_px, h_px, style, dpi=200):
    st = STYLES[style]
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor=st["bg"])
    plt.rcParams["font.family"] = st["font"]
    return fig


def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return buf


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def riso_composite(layers, inks, bg, offsets=None, grain=0.0, seed=0):
    """layers: list of (H,W) coverage arrays in [0,1]; inks: hex; multiply blend on paper.
    offsets: per-layer (dy, dx) pixel misregistration (declared aesthetic)."""
    H, W = layers[0].shape
    out = np.ones((H, W, 3)) * hex2rgb(bg)
    rng = np.random.default_rng(seed)
    for i, (cov, ink) in enumerate(zip(layers, inks)):
        if offsets is not None:
            dy, dx = offsets[i]
            cov = np.roll(np.roll(cov, dy, 0), dx, 1)
        if grain > 0:
            cov = np.clip(cov * (1 - grain * rng.random((H, W))), 0, 1)
        c = hex2rgb(ink)
        out *= (1 - cov[..., None] * (1 - c))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def coverage_from_fig(fig):
    """Render a white-background, black-ink figure and return ink coverage in [0,1]."""
    arr = fig_to_array(fig).astype(np.float32) / 255.0
    return 1.0 - arr.mean(-1)


def save_png(arr_or_fig, path, style=None):
    from PIL import Image
    if isinstance(arr_or_fig, np.ndarray):
        Image.fromarray(arr_or_fig).save(path, optimize=True)
    else:
        arr_or_fig.savefig(path, facecolor=arr_or_fig.get_facecolor(), dpi=arr_or_fig.dpi)
        plt.close(arr_or_fig)
