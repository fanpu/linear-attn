"""Spectral pieces: Before/After diptych of ||DFT_a W_E||, the spectrogram over training (time x frequency),
a long horizontal 'score' strip, and a 12-voice score (one row per seed).  Output: gallery/spectral_*.png"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from styles import STYLES, fig_canvas, fig_to_array, riso_composite, save_png, hex2rgb
from phases import phases

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
A = dict(np.load("cache/analysis.npz"))
s = args.seed
tag = f"seed{int(A['init_seeds'][s])}"
steps = A["emb_steps"]; spec = A["spec_E"]            # (T,S,57)
ph = phases(A, s)
K = int(A["nkeys"][s]); keys = sorted(A["keys"][s][:K])
KF = np.arange(1, 57)

# uniform time grid (every 20 steps); the dense early checkpoints are subsampled onto it (declared)
grid = np.arange(0, steps[-1] + 1, 20)
gi = np.searchsorted(steps, grid)


def share(sp):
    """Declared normalisation: each column is the share of ||W_E||^2 carried by frequency k (k=1..56)."""
    e = sp[..., 1:] ** 2
    return e / e.sum(-1, keepdims=True)


def cmap_for(style):
    st = STYLES[style]
    if style == "nocturne":
        return plt.get_cmap("magma")
    if style == "plate":
        return LinearSegmentedColormap.from_list("sep", [st["bg"], "#b89b72", "#6b4e2f", "#1f150b"])
    return LinearSegmentedColormap.from_list("ink", [st["bg"], st["ink"]])


# ------------------------------------------------------------------ diptych
def diptych(style):
    st = STYLES[style]
    t_before = int(np.searchsorted(steps, ph["t_mem"]))     # the moment memorisation completes
    cols = [(t_before, f"before · step {int(steps[t_before])}", "memorised, test accuracy at chance"),
            (len(steps) - 1, f"after · step {int(steps[-1])}", "grokked, test accuracy 100%")]
    fig = fig_canvas(1800, 3000, style)
    for c, (ti, head, sub) in enumerate(cols):
        sh = share(spec[ti, s])
        ax = fig.add_axes([0.08 + c * 0.47, 0.06, 0.38, 0.8]); ax.set_facecolor(st["bg"])
        v = sh / sh.max() if c == 0 else sh / share(spec[-1, s]).max()
        # frequency runs top (k=1) to bottom (k=56); each frequency is a horizontal 'spectral line'
        if style == "nocturne":
            img = cmap_for(style)(np.sqrt(sh / share(spec[-1, s]).max()))[:, None, :3]
            ax.imshow(np.repeat(img, 40, 1), aspect="auto", extent=[0, 1, 56.5, 0.5], interpolation="nearest")
        elif style == "plate":
            # absorption-spectrum idiom: darker = more energy, drawn as a strip
            img = cmap_for(style)(np.sqrt(sh / share(spec[-1, s]).max()))[:, None, :3]
            ax.imshow(np.repeat(img, 40, 1), aspect="auto", extent=[0, 1, 56.5, 0.5], interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_color(st["ink"]); sp.set_linewidth(0.8)
        else:
            L = sh / share(spec[-1, s]).max()
            ax.hlines(KF, 0, np.clip(L, 0, 1), color=st["ink"], lw=2.2)
            ax.set_xlim(0, 1.02)
        ax.set_ylim(56.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([1, 14, 28, 42, 56] if c == 0 else [])
        ax.tick_params(colors=st["muted"], labelsize=11)
        if style != "plate":
            for sp in ax.spines.values():
                sp.set_visible(False)
        for k in keys:
            if c == 1:
                ax.text(1.04, k, f"k={k}", transform=ax.get_yaxis_transform(), va="center", fontsize=11, color=st["muted"])
        fig.text(0.08 + c * 0.47 + 0.19, 0.93, head, ha="center", fontsize=17, color=st["ink"],
                 style="italic" if style == "plate" else "normal")
        fig.text(0.08 + c * 0.47 + 0.19, 0.9, sub, ha="center", fontsize=11, color=st["muted"])
    fig.text(0.04, 0.46, "frequency k  (token a ↦ cos, sin of 2πka/113)", rotation=90, va="center", fontsize=12, color=st["muted"])
    fig.text(0.5, 0.025, f"share of ‖W_E‖² at each frequency, {tag}; both panels on the same scale (sqrt tone map)" if style != "plotter"
             else f"line length = share of ‖W_E‖² at frequency k (same scale), {tag}", ha="center", fontsize=11, color=st["muted"])
    return fig


# ------------------------------------------------------------------ spectrogram
def spectrogram(style, W=3600, Hh=1500, seeds=None, strip=False):
    st = STYLES[style]
    seeds = [s] if seeds is None else seeds
    n = len(seeds)
    fig = fig_canvas(W, Hh, style)
    top = 0.86 if not strip else 0.9
    bot = 0.12 if not strip else 0.06
    hh = (top - bot) / n
    for r, ss in enumerate(seeds):
        sh = share(spec[gi, ss])                                    # (G,56)
        ax = fig.add_axes([0.06, top - (r + 1) * hh + 0.004, 0.86, hh - 0.008]); ax.set_facecolor(st["bg"])
        if style == "riso":
            ax.remove()
            continue
        v = np.sqrt(sh / np.quantile(sh[-200:], 0.999))              # declared sqrt tone map, saturate at final peak
        ax.imshow(cmap_for(style)(np.clip(v.T, 0, 1)), aspect="auto", interpolation="nearest",
                  extent=[grid[0], grid[-1], 56.5, 0.5])
        ax.set_yticks([] if n > 1 else [1, 14, 28, 42, 56]); ax.tick_params(colors=st["muted"], labelsize=10)
        if r < n - 1:
            ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if n > 1:
            kk = sorted(A["keys"][ss][:A["nkeys"][ss]])
            ax.text(1.005, 0.5, f"seed {int(A['init_seeds'][ss])}" + ("" if A["data_seeds"][ss] == 598 else "*") + "\n" + " ".join(map(str, kk)),
                    transform=ax.transAxes, va="center", fontsize=9, color=st["muted"])
        else:
            for name, t in [("memorised", ph["t_mem"]), ("circuit", ph["t_circ"]), ("cleanup", ph["t_clean"]), ("100% test", ph["t_done"])]:
                if t > 0:
                    ax.axvline(t, color=st["muted"] if style != "nocturne" else "#ffffff", lw=0.6, ls=":", alpha=0.7)
                    ax.text(t, 0.3, " " + name, color=st["muted"], fontsize=10, va="bottom")
            for k in keys:
                ax.text(1.005, k, f"k={k}", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=st["muted"])
    if style == "riso":
        return riso_spectrogram(W, Hh, seeds, top, bot, hh)
    if not strip:
        fig.text(0.06, 0.93, "Spectrogram of the embedding during training" if n == 1 else "Twelve seeds, twelve scores",
                 fontsize=20, color=st["ink"], style="italic" if style == "plate" else "normal")
        fig.text(0.06, 0.05, "time (training step) →        rows: frequency k = 1…56        tone: share of ‖W_E‖² at k (sqrt)",
                 fontsize=11, color=st["muted"])
    return fig


def riso_spectrogram(W, Hh, seeds, top, bot, hh):
    """Two inks: key-frequency rows in pink, all other rows in blue (declared separation), misregistered."""
    st = STYLES["riso"]
    Hpx = Hh; lay = [np.zeros((Hpx, W)), np.zeros((Hpx, W))]
    x0, x1 = int(0.06 * W), int(0.92 * W)
    for r, ss in enumerate(seeds):
        sh = share(spec[gi, ss])
        v = np.clip(np.sqrt(sh / np.quantile(sh[-200:], 0.999)), 0, 1)
        kk = set(A["keys"][ss][:A["nkeys"][ss]].tolist())
        ytop = int((1 - (top - r * hh)) * Hpx) + 2; ybot = int((1 - (top - (r + 1) * hh)) * Hpx) - 2
        from PIL import Image
        for layer in range(2):
            mask = np.array([(k in kk) == (layer == 1) for k in range(1, 57)])
            img = (v * mask[None]).T                                      # (56, G)
            im = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((x1 - x0, ybot - ytop), Image.NEAREST)) / 255.0
            lay[layer][ytop:ybot, x0:x1] = im
    out = riso_composite(lay, [st["ink"], st["ink2"]], st["bg"], offsets=[(0, 0), (4, -6)], grain=0.15)
    return out


for style in STYLES:
    if style != "riso":
        save_png(diptych(style), f"gallery/spectral_diptych_{tag}_{style}.png")
    save_png(spectrogram(style), f"gallery/spectrogram_{tag}_{style}.png")
    save_png(spectrogram(style, W=6000, Hh=900, strip=True), f"gallery/spectral_score_strip_{tag}_{style}.png")
    save_png(spectrogram(style, W=3600, Hh=3000, seeds=list(range(len(A["init_seeds"])))), f"gallery/spectrogram_12seeds_{style}.png")
    print("wrote", style, flush=True)
