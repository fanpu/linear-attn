"""Hero animation: min-norm random-features fits as the number of features sweeps through n = 20.

    python render_hero.py            -> figures/hero.mp4 + figures/hero_poster.png
    python render_hero.py --still P  -> scratch/hero_still_P.png (for iteration)
"""
import sys, pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import style as S

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
FPS = 30
W_PX, H_PX, DPI = 1920, 1080, 120
YL = 2.6


def dwell(P):
    if P <= 10:
        return 5
    if P <= 16:
        return 9
    if P <= 26:
        return 16
    if P <= 60:
        return 5
    return 3


def smooth(t):
    return t * t * (3 - 2 * t)


def build_frames(Ps):
    fr = []
    for i in range(len(Ps) - 1):
        d = dwell(Ps[i])
        for k in range(d):
            fr.append((i, smooth(k / d)))
    last = len(Ps) - 1
    fr += [(last, 0.0)] * int(2.2 * FPS)
    return fr


class Hero:
    def __init__(self):
        z = np.load(HERE / "cache" / "hero.npz")
        self.z = z
        self.Ps = z["Ps"]; self.xs = z["xs"]; self.fits = np.clip(z["fits"], -8, 8)
        self.R = z["R"]; self.SM = z["SM"]
        self.med = np.median(self.R, 1); self.q1, self.q3 = np.percentile(self.R, [25, 75], axis=1)
        self.smed = np.median(self.SM, 1)
        self.n = len(z["x"])
        c = S.use("dark")
        self.c = c
        self.fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
        fig = self.fig
        self.ax = fig.add_axes([0.055, 0.395, 0.90, 0.46])
        self.ar = fig.add_axes([0.055, 0.075, 0.42, 0.22])
        self.asv = fig.add_axes([0.535, 0.075, 0.42, 0.22])
        self.norm = Normalize(np.log10(0.006), np.log10(20.0), clip=True)
        self._static()

    def _static(self):
        ax, ar, asv, z, c = self.ax, self.ar, self.asv, self.z, self.c
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-YL, YL)
        for s in ["left", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_major_locator(NullLocator()); ax.yaxis.set_major_locator(NullLocator())
        ax.axhline(0, color=S.NIGHT_RULE, lw=0.8, zorder=0)
        ax.plot(self.xs, z["fs"], color=S.NIGHT_MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1, alpha=0.9)
        # data points with a soft halo
        for s_, a_ in [(420, 0.05), (180, 0.10)]:
            ax.scatter(z["x"], z["y"], s=s_, color=S.DATA_D, alpha=a_, lw=0, zorder=8)
        ax.scatter(z["x"], z["y"], s=46, color=S.DATA_D, lw=1.6, edgecolors=S.NIGHT, zorder=9)
        self.fig.text(0.055, 0.955, "Fitting 20 noisy points with more and more random features",
                      fontsize=21, color=S.NIGHT_INK, fontweight="bold", va="top")
        self.fig.text(0.055, 0.912, "minimum-norm least squares on p random Fourier features  ·  dashed: the true function  ·  colour = test error",
                      fontsize=13, color=S.NIGHT_MUTED, va="top")
        self.ptext = self.fig.text(0.945, 0.955, "", fontsize=30, color=S.NIGHT_INK, ha="right", va="top", fontweight="bold")
        self.rtext = self.fig.text(0.945, 0.893, "", fontsize=16, color=S.NIGHT_INK, ha="right", va="top")
        # glow line collections (several widths)
        self.glow = []
        for lw, a in [(14, 0.05), (8, 0.10), (4.5, 0.22), (2.2, 1.0)]:
            lc = LineCollection([], linewidths=lw, alpha=a, capstyle="round", joinstyle="round", zorder=5)
            ax.add_collection(lc); self.glow.append(lc)
        self.ghosts = [ax.plot([], [], lw=1.0, zorder=3)[0] for _ in range(5)]

        Ps = self.Ps
        for a_ in (ar, asv):
            a_.set_xscale("log"); a_.set_xlim(1, Ps[-1] * 1.05)
            a_.xaxis.set_major_locator(FixedLocator([1, 10, 20, 100, 1000, 5000]))
            a_.set_xticklabels(["1", "10", "20", "100", "1000", "5000"])
            a_.xaxis.set_minor_formatter(NullFormatter())
            a_.axvline(self.n, color=S.GOLD_D, lw=1.2, alpha=0.8, zorder=0)
            a_.set_xlabel("number of features p", fontsize=13)
            a_.tick_params(labelsize=12.5)
        ar.set_yscale("log"); ar.set_ylim(3e-3, 300)
        ar.fill_between(Ps, self.q1, self.q3, color="#6b5b9a", alpha=0.35, lw=0)
        pts = np.column_stack([np.log10(Ps), np.log10(self.med)])
        segs = np.stack([np.column_stack([Ps[:-1], self.med[:-1]]), np.column_stack([Ps[1:], self.med[1:]])], 1)
        lc = LineCollection(segs, cmap=S.HEAT, norm=self.norm, linewidths=2.2, zorder=3)
        lc.set_array(np.log10(0.5 * (self.med[:-1] + self.med[1:])))
        ar.add_collection(lc)
        ar.set_title("test error (median of 200 feature draws, band = middle 50%)", fontsize=14.5, color=S.NIGHT_INK,
                     fontweight="normal", loc="left", pad=8)
        ar.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
        ar.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
        ar.text(self.n * 1.08, 150, "p = n", color=S.GOLD_D, fontsize=13, va="center")
        asv.text(self.n * 1.08, 0.3, "p = n", color=S.GOLD_D, fontsize=13, va="center")
        asv.set_yscale("log"); asv.set_ylim(3e-13, 3)
        asv.plot(Ps, self.smed, color=S.NIGHT_INK, lw=1.8, alpha=0.9, zorder=3)
        asv.set_title("smallest singular value of the 20 × p feature matrix", fontsize=14.5, color=S.NIGHT_INK,
                      fontweight="normal", loc="left", pad=8)
        asv.yaxis.set_major_locator(FixedLocator([1e-12, 1e-8, 1e-4, 1]))
        asv.set_yticklabels(["1e-12", "1e-8", "1e-4", "1"])
        asv.yaxis.set_minor_locator(NullLocator())
        ar.yaxis.set_minor_locator(NullLocator())
        self.dots = []
        for a_ in (ar, asv):
            halo = a_.scatter([], [], s=260, color=S.DATA_D, alpha=0.18, lw=0, zorder=5)
            dot = a_.scatter([], [], s=55, color=S.DATA_D, lw=0, zorder=6)
            self.dots.append((halo, dot))

    def draw(self, i, t, trail):
        Ps = self.Ps
        j = min(i + 1, len(Ps) - 1)
        f = (1 - t) * self.fits[i] + t * self.fits[j]
        lp = (1 - t) * np.log(Ps[i]) + t * np.log(Ps[j])
        lr = (1 - t) * np.log10(self.med[i]) + t * np.log10(self.med[j])
        rs = (1 - t) * np.log10(self.z["risk_shown"][i]) + t * np.log10(self.z["risk_shown"][j])
        ls = (1 - t) * np.log10(self.smed[i]) + t * np.log10(self.smed[j])
        col = S.HEAT(self.norm(rs))
        fc = np.clip(f, -YL * 1.6, YL * 1.6)
        seg = np.stack([np.column_stack([self.xs[:-1], fc[:-1]]), np.column_stack([self.xs[1:], fc[1:]])], 1)
        for lc in self.glow:
            lc.set_segments(seg); lc.set_color(col)
        for g, (k, a) in zip(self.ghosts, trail):
            g.set_data(self.xs, np.clip(self.fits[k], -YL * 1.6, YL * 1.6)); g.set_color(S.HEAT(self.norm(np.log10(self.z["risk_shown"][k]))))
            g.set_alpha(a)
        for g in self.ghosts[len(trail):]:
            g.set_data([], [])
        P_show = Ps[j] if t >= 0.5 else Ps[i]
        self.ptext.set_text(f"p = {P_show:,}")
        if P_show < self.n - 2:
            reg = "fewer features than points: a smooth compromise"
        elif P_show <= self.n + 4:
            reg = "p ≈ n: forced to thread every noisy point"
        elif P_show < 200:
            reg = "more features than points: many exact fits, pick the smallest"
        else:
            reg = "p ≫ n: the smallest exact fit is calm again"
        self.rtext.set_text(reg)
        x = np.exp(lp)
        for (halo, dot), yv in zip(self.dots, [10 ** lr, 10 ** ls]):
            halo.set_offsets([[x, yv]]); dot.set_offsets([[x, yv]])

    def frame_rgb(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]


def trail_for(frames, idx):
    i = frames[idx][0]
    return [(k, 0.16 * (1 - m / 5)) for m, k in enumerate(range(i - 1, max(i - 6, -1), -1))]


def main():
    h = Hero()
    Ps = h.Ps
    if "--still" in sys.argv:
        P = int(sys.argv[sys.argv.index("--still") + 1])
        i = int(np.searchsorted(Ps, P))
        h.draw(i, 0.0, [(k, 0.16 * (1 - m / 5)) for m, k in enumerate(range(i - 1, max(i - 6, -1), -1))])
        out = HERE / "scratch" / f"hero_still_{P}.png"; out.parent.mkdir(exist_ok=True)
        h.fig.savefig(out, dpi=DPI * 0.6); print(out)
        return
    frames = build_frames(Ps)
    # loop: crossfade back to the first frame
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = FIG / "hero.mp4"
    proc = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "slow",
                             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    h.draw(0, 0.0, []); first = h.frame_rgb().astype(np.float32).copy()
    last = None
    for idx, (i, t) in enumerate(frames):
        h.draw(i, t, trail_for(frames, idx))
        rgb = h.frame_rgb()
        proc.stdin.write(rgb.tobytes()); last = rgb.astype(np.float32).copy()
    for k in range(int(0.8 * FPS)):
        a = smooth((k + 1) / (0.8 * FPS))
        proc.stdin.write(((1 - a) * last + a * first).astype(np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    i22 = int(np.searchsorted(Ps, 22))
    h.draw(i22, 0.0, [(k, 0.16 * (1 - m / 5)) for m, k in enumerate(range(i22 - 1, max(i22 - 6, -1), -1))])
    h.fig.savefig(FIG / "hero_poster.png", dpi=DPI)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main()
