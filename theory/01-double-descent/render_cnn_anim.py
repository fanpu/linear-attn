"""Animation: the model-wise test-error curve of the CNN family as training proceeds (dark).  -> figures/cnn_peak.mp4"""
import sys, pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import cnn_data as cd
import style as S
from render_cnn import analysis, EPS

HERE = pathlib.Path(__file__).resolve().parent
FPS, W_PX, H_PX, DPI = 30, 1800, 1000, 120


def main():
    d = cd.load()
    ks, E = d["k"], d["epoch"]
    te, tr, kpk, kem = analysis(d)
    S.use("dark")
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    a1 = fig.add_axes([0.08, 0.40, 0.86, 0.45]); a2 = fig.add_axes([0.08, 0.09, 0.86, 0.23])
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_xlim(0.9, 72)
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64])); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.xaxis.set_minor_locator(NullLocator()); ax.tick_params(labelsize=12.5)
    a1.set_xticklabels([])
    a1.set_ylim(0.33, 0.92); a1.set_ylabel("test error", fontsize=13.5)
    a2.set_ylim(-0.03, 0.95); a2.set_ylabel("train error", fontsize=13.5); a2.set_xlabel("CNN width k", fontsize=13.5)
    a2.axhline(EPS, color=S.NIGHT_MUTED, lw=1, ls=(0, (1, 3)))
    a2.text(0.95, EPS + 0.03, "10%", color=S.NIGHT_MUTED, fontsize=11.5)
    a2.axhline(0.2, color=S.NIGHT_RULE, lw=1)
    fig.text(0.08, 0.955, "Deep double descent, unfolding in training time", fontsize=22, fontweight="bold", va="top", color=S.NIGHT_INK)
    fig.text(0.08, 0.905, "14 CNNs of width k trained side by side on 10k CIFAR-10 images, 20% of labels wrong  ·  gold: where train error crosses 10% (the EMC = n prediction)",
             fontsize=13, va="top", color=S.NIGHT_MUTED)
    etext = fig.text(0.94, 0.955, "", fontsize=30, fontweight="bold", ha="right", va="top", color=S.NIGHT_INK)
    ghosts = [a1.plot([], [], lw=1.2)[0] for _ in range(6)]
    glow1 = [a1.plot([], [], color=S.VAR_D, lw=lw, alpha=al, solid_capstyle="round")[0] for lw, al in [(10, 0.07), (5, 0.18), (2.4, 1)]]
    dots1, = a1.plot([], [], "o", color=S.VAR_D, ms=6, mec=S.NIGHT, mew=1.2)
    glow2 = [a2.plot([], [], color=S.BIAS_D, lw=lw, alpha=al)[0] for lw, al in [(8, 0.08), (2.2, 1)]]
    dots2, = a2.plot([], [], "o", color=S.BIAS_D, ms=5, mec=S.NIGHT, mew=1.0)
    vl1 = a1.axvline(1, color=S.GOLD_D, lw=1.6, ls=(0, (4, 3))); vl2 = a2.axvline(1, color=S.GOLD_D, lw=1.6, ls=(0, (4, 3)))
    star, = a1.plot([], [], "*", ms=20, color=S.DATA_D, mec=S.NIGHT, mew=1.0, zorder=8)
    lab = a1.text(0, 0, "", color=S.NIGHT_INK, fontsize=12.5, zorder=9)

    lE = np.log(E)
    T = np.linspace(lE[0], lE[-1], 330)
    frames = list(T) + [lE[-1]] * 60
    import imageio_ffmpeg
    out = HERE / "figures" / "cnn_peak.mp4"
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    interp = lambda M, t: np.array([np.interp(t, lE, row) for row in M])
    hist = []
    for fi, t in enumerate(frames):
        cur_te, cur_tr = interp(te, t), interp(tr, t)
        for g in glow1: g.set_data(ks, cur_te)
        dots1.set_data(ks, cur_te)
        for g in glow2: g.set_data(ks, cur_tr)
        dots2.set_data(ks, cur_tr)
        if fi % 30 == 0:
            hist.append(cur_te.copy())
        for gi, g in enumerate(ghosts):
            if gi < len(hist) - 1:
                h = hist[-(gi + 2)]
                g.set_data(ks, h); g.set_color(S.VAR_D); g.set_alpha(0.22 * (1 - gi / 6))
        ke = cd.crossing(ks, cur_tr, EPS)
        for v in (vl1, vl2):
            v.set_xdata([ke, ke] if np.isfinite(ke) else [1e-3, 1e-3])
        j = int(np.argmin(np.abs(lE - t)))
        if np.isfinite(kpk[j]):
            yk = np.interp(np.log(kpk[j]), np.log(ks), cur_te)
            star.set_data([kpk[j]], [yk]); lab.set_position((kpk[j] * 1.1, yk + 0.025)); lab.set_text("peak")
        else:
            star.set_data([], []); lab.set_text("")
        etext.set_text(f"epoch {np.exp(t):.0f}")
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
        if fi == len(T) - 1:
            fig.savefig(HERE / "figures" / "cnn_peak_poster.png", dpi=DPI)
    proc.stdin.close(); proc.wait()
    print(out)


if __name__ == "__main__":
    main()
