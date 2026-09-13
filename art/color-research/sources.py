"""Read-only loaders for real arrays computed by other projects (never written to).

Each loader returns a dict with the array(s), a one-line description of what is measured, and the
semantics needed to colour it (sign convention, which magnitude is near the boundary).
"""
import numpy as np

ART = "/home/fzeng/ml/research/art"
HW = "/home/fzeng/ml/research/hardware"


def trainability(which="zoomA"):
    """Sohl-Dickstein convergence measure M on a (eta0, eta1) learning-rate grid.
    M < 0 converged (|M| = sum of normalised losses), M > 0 diverged (|M| = sum of inverse losses);
    the slowest runs (largest |M|) sit at the fractal boundary -> near_boundary='large'."""
    f = {"zoomA": f"{ART}/trainability-fractal/cache/zoom_zoomA/kf_000.npz",
         "dip": f"{ART}/trainability-fractal/cache/windows/dip_B_tanh.npz",
         "explore2": f"{ART}/trainability-fractal/cache/zoom_explore/kf_002.npz",
         "liu": f"{ART}/trainability-fractal/cache/windows/liu_eps05_2048.npz",
         "liu_zoom": f"{ART}/trainability-fractal/cache/zoom_liu/kf_010.npz"}[which]
    d = np.load(f)
    M = d["measure"][::-1]  # row 0 = top
    return dict(x=M, near_boundary="large", desc=f"trainability measure ({which}), {M.shape[0]}^2, "
                "M<0 converged / M>0 diverged, 500 GD steps (trainability-fractal)")


def basin_signed(tag="z1T_4096", stride=2):
    """GD basins of 1/2(1 - x^2 y^2)^2 (Zhu et al. deg-4, eta=0.2): converged pixels carry the step at
    which loss < 1e-12 (negative), diverged pixels the smooth escape value nu (positive). Both are
    largest next to the boundary -> near_boundary='large'."""
    d = np.load(f"{ART}/gd-bifurcation/cache/basin_zhu4_{tag}.npz")
    s, tev = d["status"][::stride, ::stride], d["tev"][::stride, ::stride].astype(float)
    x = np.where(s == 1, -tev, np.nan)
    if "nu" in d.files:
        x = np.where(s == 2, np.nan_to_num(d["nu"][::stride, ::stride], nan=1.0), x)
    else:
        x = np.where(s == 2, tev, x)
    return dict(x=x[::-1], near_boundary="large", status=s[::-1],
                desc=f"GD basin map zhu4 {tag}: -(steps to converge) | +(escape value)")


def lyapunov(pattern="AABAB"):
    """Lyapunov exponent of GD on a 2-parameter alternating-learning-rate plane (gd-bifurcation).
    lambda < 0 ordered, lambda > 0 chaotic; boundary at lambda = 0 -> near_boundary='small'."""
    d = np.load(f"{ART}/gd-bifurcation/cache/lyapplane_{pattern}_2048.npz")
    return dict(x=d["lam"].astype(float)[::-1], near_boundary="small",
                desc=f"Lyapunov plane {pattern}, 2048^2, burn 1500 / record 3000 steps")


def bifurcation_density(page="p05"):
    """Histogram counts of GD iterates vs learning rate (bifurcation atlas page)."""
    d = np.load(f"{ART}/gd-bifurcation/cache/atlas/{page}.npz")
    C = d["C"].astype(float)
    return dict(x=C[::-1], desc=f"bifurcation density atlas {page} (period-{int(d['period'])} window), counts")


def spectrogram(name="int3"):
    """dB magnitude spectrogram (time x frequency) of a 30 s chirp quantized to 3 bits, 48 kHz,
    NFFT 4096 (hardware/dither). Returned as frequency (rows, high at top) x time (cols)."""
    S = np.load(f"{HW}/dither/cache/spec_{name}.npy").astype(np.float32)
    return dict(x=S.T[::-1], desc=f"spectrogram spec_{name}.npy, dB re full-scale sine, {S.shape}")


def sigma_delta_idle(level=3):
    """Spectrum of a 1-bit sigma-delta modulator vs DC input (golden-ratio zoom level), dB."""
    d = np.load(f"{HW}/dither/cache/sdzoom_deep.npz")
    return dict(x=d[f"gold_{level}"].astype(np.float32), desc=f"sigma-delta idle-tone zoom gold_{level}, dB")


def random_field(key="heaviside_L3"):
    """Output of a random deep network restricted to a sphere patch (depth-roughness tiles)."""
    d = np.load(f"{ART}/depth-roughness/cache/tiles_2048.npz")
    return dict(x=d[key].astype(float), near_boundary="small", desc=f"random-network field {key} on a sphere patch, 2048^2")


def gradient_direction(key="heaviside_L2", smooth=2.0):
    """Orientation of the gradient of a random-network field: genuinely cyclic in [0, 2pi)."""
    from scipy import ndimage
    f = random_field(key)["x"]
    f = ndimage.gaussian_filter(f, smooth)
    gy, gx = np.gradient(f)
    th = np.arctan2(gy, gx)
    mag = np.hypot(gx, gy)
    return dict(x=(th + np.pi) / (2 * np.pi), mag=mag, desc=f"gradient direction of {key} (cyclic), |grad| for shading")


def basin_classes(tag="wide_4096", stride=2):
    """Categorical basin map: 0 = diverged, 1..4 = converged to the minimum branch with
    sign pattern (sign fx, sign fy) in (-,-), (-,+), (+,-), (+,+). zhu4 model, eta=0.2."""
    d = np.load(f"{ART}/gd-bifurcation/cache/basin_zhu4_{tag}.npz")
    s = d["status"][::stride, ::stride]
    fx = d["fx"][::stride, ::stride]
    fy = d["fy"][::stride, ::stride]
    c = np.zeros(s.shape, np.int8)
    conv = s == 1
    c[conv] = 1 + 2 * (fx[conv] > 0) + (fy[conv] > 0)
    tev = d["tev"][::stride, ::stride].astype(float)
    return dict(x=c[::-1], tev=tev[::-1], ncls=5, desc=f"zhu4 {tag} basin classes (diverged + 4 minimum branches)",
                labels=["diverged", "x<0,y<0", "x<0,y>0", "x>0,y<0", "x>0,y>0"])
