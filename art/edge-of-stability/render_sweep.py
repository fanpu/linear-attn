"""Learning-rate sweep plates (render step).

usage: python render_sweep.py <cache.npz> <tag>

Rows = step sizes (2/eta ascending, top to bottom), columns = GD steps.
  edge_*   colour = (lambda_1 - 2/eta) / (2/eta), split at 0. Spectral split: each side rank-
           normalised separately (pooled over all rows) onto one half of Spectral; below the edge
           purple->blue->green->pale, above the edge deep red->orange->pale; the dark ends meet at 0.
  phase_*  colour = demodulated oscillation coordinate (-1)^t x_t, split at 0 and rank-normalised per
           row (declared: amplitudes differ by orders of magnitude between rows). A seam running
           down a band is a phase slip of the period-2 oscillation, i.e. a crossing in the braid.
           Pixels where the amplitude is below 1e-5 (no oscillation, float32 floor) are ground.
  score_*  one stave per step size: even/odd-step strands of x_t, each stave scaled to its own
           99.7th percentile (declared per-stave gain).
Eigenvalues are refreshed every E steps (see meta); between refreshes they are held.
"""
import sys

from PIL import Image

from eos_common import *  # noqa: F401,F403


def band_image(field_rgb, row_h, gap, bg):
    R, C, _ = field_rgb.shape
    H = R * row_h + (R + 1) * gap
    img = np.empty((H, C, 3))
    img[:] = np.array(PAL.hex2rgb(bg))
    for r in range(R):
        y = gap + r * (row_h + gap)
        img[y:y + row_h] = field_rgb[r][None]
    return img


def to_png(img, name, width=None):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    p = os.path.join(GAL, name)
    im.save(p)
    print("wrote", p)


def fields(d, t0=0, amin=1e-5):
    invs = d["invs"]
    lam = ffill(d["evals"][:, :, 0].astype(float))
    alive = np.isfinite(d["loss"]).T  # (M, T)
    edge = ((lam - invs[None]) / invs[None]).T
    edge[~alive] = np.nan
    x = np.stack([braid(d, r) for r in range(len(d["invs"]))])
    T = x.shape[1]
    s = (-1.0) ** np.arange(T)
    xd = x * s[None]
    amp = np.sqrt(0.5 * (x ** 2 + np.roll(x, -1, 1) ** 2))
    xd[~(amp > amin)] = np.nan
    xd[~alive] = np.nan
    return edge[:, t0:], xd[:, t0:], invs


def main(f, tag):
    d = load(f)
    edge, xd, invs = fields(d)
    E = d["meta"]["eig_every"]
    M, T = edge.shape
    W = 4000
    # --- edge plate (spectral + a second pairing)
    for pairing, bg in [("sd_spectral", "#0d0d12"), ("hubble_sho", "#0a0b10")]:
        rgb = spectral_split(edge, near_boundary="small", pairing=pairing, nan_color=bg)
        colw = max(1, W // T)
        rgb = np.repeat(rgb, colw, axis=1) if colw > 1 else rgb
        img = band_image(rgb, row_h=max(1, int(rgb.shape[1] * 0.45 / M)), gap=max(2, rgb.shape[1] // 700), bg=bg)
        to_png(img, f"sweep_{tag}_edge_{pairing}.png", width=W)
    # --- phase plate, per-row normalisation
    rgb = np.zeros(xd.shape + (3,))
    for r in range(M):
        rgb[r] = spectral_split(xd[r][None], near_boundary="small", nan_color="#0d0d12")[0]
    img = band_image(rgb, row_h=max(1, int(T * 0.45 / M)), gap=max(2, T // 700), bg="#0d0d12")
    to_png(img, f"sweep_{tag}_phase_spectral.png", width=W)
    # --- score plates
    for style in ["night", "paper"]:
        bg = NIGHT if style == "night" else PAPER
        ce, co = ("#f2a541", "#58b4c4") if style == "night" else (INK, "#c0392b")
        fg = "#9a968a" if style == "night" else "#6b675e"
        fig = plt.figure(figsize=(16, 1.0 * M + 1.2), facecolor=bg)
        h = (1 - 0.08) / M
        for r in range(M):
            ax = fig.add_axes([0.06, 1 - 0.03 - (r + 1) * h, 0.92, h * 0.92], facecolor=bg)
            x = braid(d, r)
            t = np.arange(len(x))
            ok = np.isfinite(x)
            if ok.sum() < 10:
                ax.axis("off")
                ax.text(0, 0.5, f"2/η = {invs[r]:.0f}   diverged at step {d['diverged_at'][r]}",
                        transform=ax.transAxes, color=fg, fontsize=8, family=MONO, va="center")
                continue
            A = np.percentile(np.abs(x[ok & (t > 30)]), 99.7) * 1.1 + 1e-12
            ev = ok & (t % 2 == 0)
            od = ok & (t % 2 == 1)
            ax.plot(t[ev], x[ev], color=ce, lw=0.35)
            ax.plot(t[od], x[od], color=co, lw=0.35)
            ax.set_ylim(-A, A)
            ax.set_xlim(0, len(x) - 1)
            ax.axis("off")
            ax.text(-0.005, 0.5, f"2/η = {invs[r]:.0f}", transform=ax.transAxes, ha="right",
                    va="center", color=fg, fontsize=8, family=MONO)
        fig.text(0.06, 0.012, f"x_t along the current top Hessian eigenvector, even / odd steps; per-stave gain; "
                 f"steps 0–{T - 1}; eigenvectors refreshed every {E} steps", color=fg, fontsize=8, family=MONO)
        save(fig, f"sweep_{tag}_score_{style}.png", dpi=250, facecolor=bg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
