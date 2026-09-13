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
    N0 = d["loss"].shape[0]
    for key, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N0 and key != "invs":
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


def pca_braid(d, m, win=64, half=10):
    """Windowed-PCA oscillation coordinate from the count-sketch (swap-free cross-check of x_t).

    r_t = sketch(theta_t - theta_0) minus its centred (2*half+1)-step moving average; in windows of
    `win` steps (hop win/2) the top principal direction p_w of r is found, sign-aligned to the previous
    window, and c_t = <r_t, p_w> is blended across overlapping windows with triangular weights.
    Returns c (T,), cos (per window |cos| between p_w and the mean sketched u1 in that window), centres.
    """
    from scipy.ndimage import uniform_filter1d
    s = d["sketch"][:, m].astype(np.float64)
    r = s - uniform_filter1d(s, size=2 * half + 1, axis=0, mode="nearest")
    T = len(r)
    hop = win // 2
    c = np.zeros(T)
    wsum = np.zeros(T)
    prev = None
    coss, cents = [], []
    tri = 1 - np.abs(np.linspace(-1, 1, win))
    u1s = d["u1_sketch"][:, m].astype(np.float64)
    for a in range(0, T - win + 1, hop):
        R = r[a:a + win]
        _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        p = Vt[0]
        if prev is not None and p @ prev < 0:
            p = -p
        prev = p
        c[a:a + win] += tri * (R @ p)
        wsum[a:a + win] += tri
        u = u1s[a:a + win]
        u = u * np.sign(u @ p)[:, None]
        um = u.mean(0)
        coss.append(abs(um @ p) / (np.linalg.norm(um) + 1e-30))
        cents.append(a + win // 2)
    c = np.where(wsum > 0, c / np.maximum(wsum, 1e-12), np.nan)
    return c, np.array(coss), np.array(cents)


COORD = os.environ.get("EOS_COORD", "pca")  # 'pca' (windowed PCA, default) or 'u1' (current top eigenvector)


def braid(d, m, kind=None, ts=40):
    """Oscillation coordinate for model m: windowed-PCA (swap-free) or along the current u1."""
    kind = kind or COORD
    if kind == "pca" and "sketch" in d:
        v = pca_braid(d, m)[0]
    else:
        v = d["x"][:, m].astype(float).copy()
    v[:ts] = np.nan
    return v
