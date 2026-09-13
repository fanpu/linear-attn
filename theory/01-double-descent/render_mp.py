"""Marchenko-Pastur animation: as gamma = p/n passes 1 the spectrum's lower edge hits zero and the variance explodes.

One Gaussian matrix X (n x p_max); frame k uses its first p columns, so the spectrum evolves continuously.
    python render_mp.py            -> figures/mp_edge.mp4, figures/mp_edge_poster.png
    python render_mp.py --still G  -> scratch/mp_still.png
"""
import sys, pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

import dd_core as C
import style as S

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
N = 800
FPS = 30
W_PX, H_PX, DPI = 1800, 900, 120
SIG2 = 0.25


def gamma_path(frames=150):
    # log-uniform in a warped coordinate that lingers around gamma = 1
    u = np.linspace(-1, 1, frames)
    w = np.sign(u) * np.abs(u) ** 1.7
    return 10 ** w


def spectra(gammas, seed=0):
    rng = np.random.default_rng(seed)
    pmax = int(np.ceil(gammas.max() * N))
    X = rng.standard_normal((N, pmax))
    out = []
    for g in gammas:
        p = max(1, int(round(g * N)))
        if p == N:
            p = N + 1 if g > 1 else N - 1
        Xp = X[:, :p]
        ev = np.linalg.eigvalsh(Xp @ Xp.T / N) if p >= N else np.linalg.eigvalsh(Xp.T @ Xp / N)
        ev = np.clip(ev, 1e-12, None)
        out.append((p, ev))
    return out


class MP:
    def __init__(self):
        S.use("light")
        self.fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
        f = self.fig
        self.ah = f.add_axes([0.06, 0.12, 0.52, 0.66])
        self.al = f.add_axes([0.67, 0.53, 0.30, 0.25])
        self.av = f.add_axes([0.67, 0.12, 0.30, 0.25])
        f.text(0.06, 0.95, "The spectrum behind the peak", fontsize=21, fontweight="bold", va="top", color=S.INK)
        f.text(0.06, 0.895, "eigenvalues of $X^\\top X/n$ for a Gaussian matrix with n = 800 rows, as the number of columns p grows",
               fontsize=13, color=S.MUTED, va="top")
        self.bins = np.geomspace(1e-7, 40, 110)
        self.lc = np.sqrt(self.bins[1:] * self.bins[:-1])
        self.gtext = f.text(0.58, 0.95, "", fontsize=24, fontweight="bold", ha="right", va="top", color=S.INK)
        self.ztext = self.ah.text(0.02, 0.96, "", transform=self.ah.transAxes, fontsize=12.5, color=S.MUTED, va="top")

    def setup_side(self, gammas, specs):
        g = np.array([s[0] for s in specs]) / N
        lmin = np.array([s[1].min() for s in specs])
        var = np.array([SIG2 / N * np.sum(1 / s[1]) for s in specs])
        gg = np.concatenate([np.geomspace(0.1, 0.999, 300), np.geomspace(1.001, 10, 300)])
        for ax in (self.al, self.av):
            ax.set_xscale("log"); ax.set_xlim(0.1, 10)
            ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
            ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"]); ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axvline(1, color=S.GOLD, lw=1.2, zorder=0)
            ax.set_yscale("log"); ax.tick_params(labelsize=11)
        edge = (1 - np.sqrt(gg)) ** 2
        self.al.plot(gg[gg < 1], edge[gg < 1], color=S.INK, lw=1.4); self.al.plot(gg[gg > 1], edge[gg > 1], color=S.INK, lw=1.4)
        self.al.plot(g, lmin, color=S.BIAS, lw=0, marker="o", ms=2.6, alpha=0.5)
        self.al.set_ylim(1e-6, 3); self.al.set_title("smallest non-zero eigenvalue", fontsize=13.5, pad=6)
        self.al.yaxis.set_major_locator(FixedLocator([1e-6, 1e-4, 1e-2, 1])); self.al.yaxis.set_minor_locator(NullLocator())
        self.al.text(0.12, 3e-5, "line: (1 − √γ)²", fontsize=11.5, color=S.INK)
        th = C.ridgeless_risk(gg, 1.0, SIG2)[2]
        self.av.plot(gg[gg < 1], th[gg < 1], color=S.INK, lw=1.4); self.av.plot(gg[gg > 1], th[gg > 1], color=S.INK, lw=1.4)
        self.av.plot(g, var, color=S.VAR, lw=0, marker="o", ms=2.6, alpha=0.5)
        self.av.set_ylim(3e-3, 3e3); self.av.set_title("variance = σ² · (1/n) · Σ 1/λ", fontsize=13.5, pad=6)
        self.av.yaxis.set_major_locator(FixedLocator([1e-2, 1, 1e2])); self.av.yaxis.set_minor_locator(NullLocator())
        self.av.set_xlabel("γ = p / n", fontsize=12.5)
        self.av.text(0.12, 60, "line: Hastie et al. (SNR 4)", fontsize=11.5, color=S.INK)
        self.dl = self.al.scatter([], [], s=70, color=S.BIAS, ec=S.PAPER, lw=1.6, zorder=6)
        self.dv = self.av.scatter([], [], s=70, color=S.VAR, ec=S.PAPER, lw=1.6, zorder=6)
        self.g, self.lmin, self.var = g, lmin, var

        ah = self.ah
        ah.set_xscale("log"); ah.set_xlim(1e-7, 40)
        ah.xaxis.set_major_locator(FixedLocator([1e-6, 1e-4, 1e-2, 1, 10]))
        ah.set_xticklabels(["$10^{-6}$", "$10^{-4}$", "0.01", "1", "10"]); ah.xaxis.set_minor_formatter(NullFormatter())
        ah.set_ylim(0, 1.25); ah.yaxis.set_major_locator(NullLocator()); ah.spines["left"].set_visible(False)
        ah.set_xlabel("eigenvalue λ  (log scale)", fontsize=12.5)
        ah.tick_params(labelsize=11.5)
        self.bars = ah.bar(self.lc, np.zeros_like(self.lc), width=np.diff(self.bins) * 0.86, color=S.INDIGO, alpha=0.55,
                           align="center", lw=0)
        self.theory_line, = ah.plot([], [], color=S.INK, lw=1.8)
        self.edge_line = ah.axvline(1, color=S.VAR, lw=1.4, ls=(0, (3, 2)))
        self.edge_text = ah.text(1, 1.18, "", color=S.INK, fontsize=12, ha="left", va="top")
        ah.text(0.02, 0.905, "bars: measured   ·   line: Marchenko–Pastur law", transform=ah.transAxes,
                ha="left", va="top", fontsize=12, color=S.MUTED)

    def draw(self, k, spec):
        p, ev = spec
        g = p / N
        h, _ = np.histogram(ev, self.bins)
        dens = h / len(ev) / np.diff(np.log10(self.bins))       # per decade
        for b, v in zip(self.bars, dens):
            b.set_height(v)
        s = np.geomspace(1e-7, 40, 1500)
        f = C.mp_density(s, g) * (g if g > 1 else 1.0)          # normalized over the non-zero eigenvalues
        self.theory_line.set_data(s, f * s * np.log(10))
        top = max(1.25, 1.4 * max(dens.max(), (f * s * np.log(10)).max()))
        self.ah.set_ylim(0, top)
        a = (1 - np.sqrt(g)) ** 2
        self.edge_line.set_xdata([max(a, 1.2e-7)] * 2)
        right = a < 3e-4
        self.edge_text.set_position((max(a, 1.2e-7) * (1.4 if right else 0.7), 0.6 * top))
        self.edge_text.set_ha("left" if right else "right")
        self.edge_text.set_text(f"lower edge (1 − √γ)² = {a:.2g}")
        self.gtext.set_text(f"p = {p}   γ = {g:.2f}")
        if p < N:
            self.ztext.set_text(f"{p} eigenvalues, all non-zero")
        else:
            self.ztext.set_text(f"{N} non-zero eigenvalues shown  (+ {p - N} exact zeros: directions the data never sees)")
        self.dl.set_offsets([[g, ev.min()]])
        self.dv.set_offsets([[g, SIG2 / N * np.sum(1 / ev)]])

    def rgb(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]


def main():
    gammas = gamma_path(170)
    specs = spectra(gammas)
    m = MP(); m.setup_side(gammas, specs)
    if "--still" in sys.argv:
        G = float(sys.argv[sys.argv.index("--still") + 1])
        k = int(np.argmin(np.abs(gammas - G)))
        m.draw(k, specs[k]); out = HERE / "scratch" / "mp_still.png"
        m.fig.savefig(out, dpi=DPI * 0.6); print(out, specs[k][0]); return
    import imageio_ffmpeg
    out = FIG / "mp_edge.mp4"
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    order = list(range(len(specs))) + [len(specs) - 1] * 20 + list(range(len(specs) - 1, -1, -1)) + [0] * 20
    for idx, k in enumerate(order):
        m.draw(k, specs[k])
        proc.stdin.write(m.rgb().tobytes())
    proc.stdin.close(); proc.wait()
    k = int(np.argmin(np.abs(gammas - 0.93)))
    m.draw(k, specs[k]); m.fig.savefig(FIG / "mp_edge_poster.png", dpi=DPI)
    print(out)


if __name__ == "__main__":
    main()
