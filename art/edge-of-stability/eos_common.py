"""Shared helpers for rendering (cache -> gallery). No GPU."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as PAL  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
GAL = os.path.join(HERE, "gallery")
os.makedirs(GAL, exist_ok=True)

INK = "#1b1b1f"
PAPER = "#f3efe6"
NIGHT = "#0b0c10"
SERIF = "C059"
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": SERIF, "savefig.facecolor": "none", "path.simplify": False})


def load(name):
    d = dict(np.load(os.path.join(CACHE, name), allow_pickle=False))
    d["meta"] = json.loads(str(d["meta"]))
    T = int(d["steps_done"])
    for key, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == d["loss"].shape[0] and key != "invs":
            d[key] = v[:T]
    return d


def ffill(a):
    """Forward-fill NaNs along axis 0 (eigenvalues between refreshes)."""
    a = a.copy()
    for t in range(1, len(a)):
        m = ~np.isfinite(a[t])
        a[t][m] = a[t - 1][m]
    return a


def envelope(x, w=10):
    """Running max of |x| over a centred window (declared smoothing for ribbon widths)."""
    from scipy.ndimage import maximum_filter1d
    y = np.nan_to_num(np.abs(x))
    return maximum_filter1d(y, size=2 * w + 1, mode="nearest")


def save(fig, name, dpi=200, **kw):
    path = os.path.join(GAL, name)
    fig.savefig(path, dpi=dpi, **kw)
    plt.close(fig)
    print("wrote", path)
    return path


def spectral_split(v, near_boundary="small", pairing="sd_spectral", **kw):
    """Signed field -> RGB via per-side rank normalisation (declared aesthetic mapping)."""
    return PAL.render_split(v, pairing, near_boundary=near_boundary, **kw)
