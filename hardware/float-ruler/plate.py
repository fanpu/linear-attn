"""Shared styling helpers for float-ruler renders (all aesthetic, declared)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cmcrameri.cm  # noqa: E402,F401  (registers cmc.* colormaps)
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).parent
CACHE = HERE / "cache"
GALLERY = HERE / "gallery"
GALLERY.mkdir(exist_ok=True)

SERIF = "P052"      # Palatino-like, for the type-specimen idiom
SERIF2 = "C059"     # Century-Schoolbook-like
MONO = "Nimbus Mono PS"
SANS = "Nimbus Sans"

STACK = "GB10 · torch 2.14.0+cu130 · exact decode of every bit pattern · 2026-09"

STYLES = {
    "engraved": dict(bg="#f3eee2", ink="#1b1a17", ink2="#8a2b1d", faint="#b9b0a0", sub="#8a2b1d"),
    "observatory": dict(bg="#07080c", ink="#e8e4d8", ink2="#f0b44c", faint="#3a3d48", sub="#6fb7d9"),
    "riso": dict(bg="#f4efe6", ink="#ff4f8b", ink2="#1f5fbf", faint="#c9c1b3", sub="#1f5fbf"),
}


def trailing_zeros(m: np.ndarray, M: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.int64)
    tz = np.full(m.shape, M, dtype=np.int64)
    nz = m != 0
    mm = m[nz]
    t = np.zeros(mm.shape, dtype=np.int64)
    while True:
        even = (mm & 1) == 0
        if not even.any():
            break
        t += even
        mm = np.where(even, mm >> 1, mm)
    tz[nz] = np.minimum(t, M)
    return tz


def paper_grain(fig, rgb=(0, 0, 0), alpha=0.035, seed=0):
    """Overlay faint paper grain (declared aesthetic)."""
    w, h = fig.canvas.get_width_height()
    rng = np.random.default_rng(seed)
    g = rng.random((h // 4, w // 4))
    ax = fig.add_axes([0, 0, 1, 1], zorder=-5)
    ax.imshow(g, cmap="gray", alpha=alpha, aspect="auto", interpolation="bilinear")
    ax.axis("off")


def save(fig, name, **kw):
    p = GALLERY / name
    fig.savefig(p, facecolor=fig.get_facecolor(), **kw)
    plt.close(fig)
    print("wrote", p)
    return p


def misregister_rgb(layers, bg, offsets):
    """Composite spot-colour layers (each HxW coverage in [0,1]) multiplicatively on paper.
    layers: list of (coverage, hex colour). offsets: per-layer (dy, dx) pixel shifts."""
    bg = np.array(matplotlib.colors.to_rgb(bg))
    out = np.ones(layers[0][0].shape + (3,)) * bg
    for (cov, col), (dy, dx) in zip(layers, offsets):
        cov = np.roll(np.roll(cov, dy, axis=0), dx, axis=1)
        c = np.array(matplotlib.colors.to_rgb(col))
        out *= 1 - cov[..., None] * (1 - c)
    return np.clip(out, 0, 1)
