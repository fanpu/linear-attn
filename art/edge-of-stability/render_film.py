"""Braid-drawing film (render step).

usage: python render_film.py <cache.npz> <model> <tag> [steps_per_frame] [t_start]

1920x1080, 30 fps, silent. The pen advances `steps_per_frame` GD steps per frame.
Top: 2/eta hairline and lambda_1..3 strokes drawn up to the pen. Middle: the whole braid so far
(even / odd strands, one global scale). Bottom: a magnifier showing the last 160 steps at a large
scale, so each individual GD step of the period-2 oscillation is visible as one zig or zag.
All quantities measured; the magnifier's vertical scale follows the running 99th percentile of |x|
over the window (declared auto-gain, the gain is printed).
Writes MP4 (H.264, yuv420p) and a GIF (every 2nd frame, 720 px, palettegen/paletteuse).
"""
import subprocess
import sys

from eos_common import *  # noqa: F401,F403


def main(f, m, tag, spf=5, ts=40):
    d = load(f)
    inv = float(d["invs"][m])
    x = braid(d, m)
    T = np.where(np.isfinite(x))[0].max() + 1
    lam = ffill(d["evals"][:, m].astype(float))[:T]
    x = np.nan_to_num(x[:T])
    x[:ts] = 0
    t = np.arange(T)
    A = np.percentile(np.abs(x), 99.7) * 1.15
    bg, fg = NIGHT, "#9a968a"
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=bg)
    at = fig.add_axes([0.05, 0.70, 0.90, 0.24], facecolor=bg)
    ab = fig.add_axes([0.05, 0.42, 0.90, 0.26], facecolor=bg)
    am = fig.add_axes([0.05, 0.06, 0.90, 0.30], facecolor=bg)
    for a in (at, ab, am):
        a.axis("off")
    at.set_xlim(0, T)
    ab.set_xlim(0, T)
    lo = np.nanmin(lam[ts:, 2])
    at.set_ylim(lo - 0.05 * (inv - lo), max(inv + 0.3 * (inv - lo), np.nanmax(lam[:, 0]) + 2))
    ab.set_ylim(-A, A)
    at.axhline(inv, color="#e8e2d0", lw=0.8)
    at.text(0, inv, " 2/η", color="#e8e2d0", fontsize=12, family=MONO, va="bottom")
    l1, = at.plot([], [], color="#f4ead2", lw=1.2)
    l2, = at.plot([], [], color="#6d6a62", lw=0.7)
    l3, = at.plot([], [], color="#6d6a62", lw=0.7)
    be, = ab.plot([], [], color="#f2a541", lw=0.6)
    bo, = ab.plot([], [], color="#58b4c4", lw=0.6)
    pen = ab.axvline(0, color="#e8e2d0", lw=0.5, alpha=0.4)
    zz, = am.plot([], [], color="#8d8a80", lw=1.0)
    me, = am.plot([], [], "o-", color="#f2a541", lw=2.0, ms=4)
    mo, = am.plot([], [], "o-", color="#58b4c4", lw=2.0, ms=4)
    win = 160
    am.set_xlim(-win, 0)
    txt = fig.text(0.05, 0.955, "", color=fg, fontsize=14, family=MONO)
    gtxt = fig.text(0.95, 0.37, "", color=fg, fontsize=11, family=MONO, ha="right")
    fig.text(0.05, 0.02, f"fc-tanh on CIFAR-10 (5000), full-batch GD, MSE, η = 2/{inv:.0f}.  "
             "Top: 2/η and top-3 Hessian eigenvalues.  Middle: " + coord_label(short=True) + ", even/odd steps.  "
             "Bottom: last 160 steps magnified.", color=fg, fontsize=11, family=MONO)
    frames = os.path.join(CACHE, "frames", tag)
    os.makedirs(frames, exist_ok=True)
    ends = list(range(ts + spf, T, spf)) + [T] * 60  # hold the last frame 2 s
    for i, e in enumerate(ends):
        tt = t[:e]
        l1.set_data(tt, lam[:e, 0])
        l2.set_data(tt, lam[:e, 1])
        l3.set_data(tt, lam[:e, 2])
        ev = tt[tt % 2 == 0]
        od = tt[tt % 2 == 1]
        be.set_data(ev, x[ev])
        bo.set_data(od, x[od])
        pen.set_xdata([e, e])
        s0 = max(ts, e - win)
        w = np.arange(s0, e)
        g = np.percentile(np.abs(x[w]), 99) * 1.25 + 1e-9 if len(w) else 1
        am.set_ylim(-g, g)
        zz.set_data(w - e, x[w])
        me.set_data(w[w % 2 == 0] - e, x[w[w % 2 == 0]])
        mo.set_data(w[w % 2 == 1] - e, x[w[w % 2 == 1]])
        txt.set_text(f"step {e - 1:5d}    λ₁ = {lam[e - 1, 0]:6.1f}    2/η = {inv:.0f}    loss = {d['loss'][e - 1, m]:.4f}")
        gtxt.set_text(f"magnifier ±{g:.1e}")
        fig.savefig(os.path.join(frames, f"{i:05d}.png"), dpi=100, facecolor=bg)
    plt.close(fig)
    mp4 = os.path.join(GAL, f"film_{tag}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(frames, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", mp4], check=True)
    gif = os.path.join(GAL, f"film_{tag}.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    gif], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], *(int(v) for v in sys.argv[4:]))
