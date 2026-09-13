"""Hero animation: logistic regression on 2D separable data, a log-time clock running to t = 1e100.

Reads cache/soudry_hero.npz, writes figures/hero.mp4, figures/hero.gif, figures/hero_still.png.
    python render_hero.py [--frames N] [--preview i,j,k]
"""
import argparse, os, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=420)
ap.add_argument("--preview", type=str, default="")
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

D = np.load("cache/soudry_hero.npz")
X, y, wh, wt, t_all, W_all = D["X"], D["y"], D["w_hat"], D["w_tilde"], D["t"], D["W"]
S_idx = D["S"]
S.dark()

lt_all = np.log10(np.maximum(t_all, 1e-12))
ok = t_all > 0
lt_all, W_all = lt_all[ok], W_all[ok]


def w_at(lt):
    return np.array([np.interp(lt, lt_all, W_all[:, j]) for j in range(2)])


def ang_deg(w, v):
    c = w @ v / np.linalg.norm(w) / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ---- time warp: slow through the visible rotation, then the clock runs away
LT0, LT1, LT_END = -2.0, 4.0, 100.0


def warp(u):
    if u < 0.42:
        return LT0 + (LT1 - LT0) * (u / 0.42)
    v = (u - 0.42) / 0.58
    return LT1 + (LT_END - LT1) * v ** 2.0


n_main = args.frames
n_hold = int(1.6 * args.fps)
lts = np.array([warp(i / (n_main - 1)) for i in range(n_main)] + [LT_END] * n_hold)

# curves for the side panels
lt_curve = np.linspace(LT0, LT_END, 1500)
ang_curve = np.array([ang_deg(w_at(l), wh) for l in lt_curve])
norm_curve = np.array([np.linalg.norm(w_at(l)) for l in lt_curve])
lt_th = np.linspace(1.0, LT_END, 600)
ang_th = np.array([ang_deg(wh * np.log(10 ** l) + wt, wh) for l in lt_th])

# ---- field
L = 4.6
g = np.linspace(-L, L, 460)
GX, GY = np.meshgrid(g, g)
P = np.stack([GX, GY], -1)
from matplotlib.colors import to_rgb
TEAL_RGB, AMBER_RGB, NIGHT_RGB, STAR_RGB = (np.array(to_rgb(c)) for c in (S.TEAL, S.AMBER, S.NIGHT, S.STAR))


def fmt_t(lt):
    if lt < 0:
        return f"{10 ** lt:.2f}", ""
    if lt < 5:
        return f"{int(round(10 ** lt)):,}", ""
    return "10", f"{lt:.0f}"


COMPARE = [(0, ""), (9.5, "a GPU-year of steps is about 10¹⁰"), (17.6, "the universe is about 10¹⁸ seconds old"),
           (80, "there are about 10⁸⁰ atoms in the universe")]


def draw(i, fig):
    lt = lts[i]
    w = w_at(lt)
    nw = np.linalg.norm(w)
    ang = ang_deg(w, wh)
    fig.clf()
    # ---------------- main plane
    ax = fig.add_axes([0.035, 0.06, 0.50625, 0.9])
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.axis("off")
    logits = P @ w
    prob = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    tint = (1 - prob)[..., None] * TEAL_RGB + prob[..., None] * AMBER_RGB
    img = NIGHT_RGB * (1 - 0.2) + 0.2 * tint
    glow = np.exp(-(logits / 1.6) ** 2)[..., None]
    img = img * (1 - 0.55 * glow) + 0.55 * glow * (0.55 * STAR_RGB + 0.45 * tint)
    ax.imshow(np.clip(img, 0, 1), extent=(-L, L, -L, L), origin="lower", interpolation="bilinear")
    # iso-probability contours (they squeeze as ||w|| grows)
    for p_lvl, a in [(0.73, 0.35), (0.95, 0.22), (0.995, 0.14)]:
        z = np.log(p_lvl / (1 - p_lvl))
        for sgn in (1, -1):
            _line(ax, w, sgn * z, color=S.STAR, lw=0.6, alpha=a)
    # max-margin target: corridor + dashed boundary
    u = wh / np.linalg.norm(wh)
    gam = 1 / np.linalg.norm(wh)
    perp = np.array([-u[1], u[0]])
    s = np.array([-2 * L, 2 * L])
    corner = [gam * u + s[0] * perp, gam * u + s[1] * perp, -gam * u + s[1] * perp, -gam * u + s[0] * perp]
    ax.fill([c[0] for c in corner], [c[1] for c in corner], color=S.LILAC, alpha=0.07, lw=0)
    for off in (gam, -gam):
        pts = off * u[None] + s[:, None] * perp[None]
        ax.plot(pts[:, 0], pts[:, 1], color=S.LILAC, lw=0.8, alpha=0.45)
    pts = s[:, None] * perp[None]
    ax.plot(pts[:, 0], pts[:, 1], color=S.LILAC, lw=1.4, ls=(0, (5, 4)), alpha=0.95)
    # current boundary with glow
    uw = w / nw
    pw = np.array([-uw[1], uw[0]])
    pts = s[:, None] * pw[None]
    S.glow_line(ax, pts[:, 0], pts[:, 1], S.STAR, lw=2.0, layers=7, spread=3.2, alpha=0.06)
    # the weight vector itself (arrow from origin, length ~ log ||w||)
    alen = min(0.55 + 0.33 * np.log1p(nw), 3.6)
    ax.annotate("", xy=uw * alen, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=S.STAR, lw=1.6))
    # data
    pos, neg = y > 0, y < 0
    ax.scatter(X[pos, 0], X[pos, 1], s=58, color=S.AMBER, edgecolor=S.NIGHT, linewidth=1.6, zorder=5)
    ax.scatter(X[neg, 0], X[neg, 1], s=52, marker="D", color=S.TEAL, edgecolor=S.NIGHT, linewidth=1.6, zorder=5)
    ax.scatter(X[S_idx, 0], X[S_idx, 1], s=250, facecolor="none", edgecolor=S.LILAC, linewidth=1.3, zorder=6)
    # plane annotations
    ax.plot([-L + 0.2, -L + 0.8], [L - 0.35, L - 0.35], color=S.STAR, lw=2)
    ax.text(-L + 0.95, L - 0.35, "decision boundary of logistic regression trained by GD", color=S.STAR, fontsize=11, va="center")
    ax.plot([-L + 0.2, -L + 0.8], [L - 0.75, L - 0.75], color=S.LILAC, lw=1.4, ls=(0, (5, 4)))
    ax.text(-L + 0.95, L - 0.75, "max-margin (hard SVM) boundary, and its margin", color=S.LILAC, fontsize=11, va="center")
    ax.text(L - 0.15, -L + 0.2, "ringed: support vectors", color=S.DIM, fontsize=9.5, ha="right")

    # ---------------- clock
    xr = 0.585
    fig.text(xr, 0.93, "TRAINING TIME", color=S.DIM, fontsize=10.5, fontweight="medium")
    base, ex = fmt_t(lt)
    tt = fig.text(xr, 0.83, "t = " + base, color=S.STAR, fontsize=40, family=S.MONO, va="baseline")
    if ex:
        fig.canvas.draw() if i == 0 else None
        r = tt.get_window_extent(renderer=fig.canvas.get_renderer())
        x1 = fig.transFigure.inverted().transform((r.x1, r.y1))[0]
        fig.text(x1 + 0.003, 0.865, ex, color=S.STAR, fontsize=23, family=S.MONO, va="baseline")
    note = ""
    for thr, txt in COMPARE:
        if lt >= thr:
            note = txt
    fig.text(xr, 0.785, note, color=S.DIM, fontsize=11, style="italic")
    fig.text(0.965, 0.83, f"{ang:.4f}°" if ang < 1 else f"{ang:.2f}°", color=S.STAR, fontsize=26, family=S.MONO, ha="right", va="baseline")
    fig.text(0.965, 0.93, "ANGLE TO MAX-MARGIN", color=S.DIM, fontsize=10.5, ha="right", fontweight="medium")
    fig.text(0.965, 0.785, rf"$\|w\|$ = {nw:,.1f}", color=S.DIM, fontsize=12, family=S.MONO, ha="right")

    # ---------------- angle panel
    a1 = fig.add_axes([xr + 0.035, 0.43, 0.345, 0.28])
    a1.set_yscale("log"); a1.minorticks_off()
    a1.set_xlim(LT0, LT_END); a1.set_ylim(0.12, 45)
    a1.plot(lt_th, ang_th, color=S.LILAC, lw=1.1, ls=(0, (4, 3)), alpha=0.9)
    m = lt_curve <= lt
    a1.plot(lt_curve[m], ang_curve[m], color=S.STAR, lw=2)
    a1.scatter([lt], [ang], s=50, color=S.STAR, edgecolor=S.NIGHT, linewidth=2, zorder=5)
    a1.scatter([lt], [ang], s=260, color=S.STAR, alpha=0.12, lw=0, zorder=4)
    a1.set_yticks([0.3, 1, 3, 10, 30]); a1.set_yticklabels(["0.3°", "1°", "3°", "10°", "30°"])
    a1.set_xticks([0, 25, 50, 75, 100]); a1.set_xticklabels(["1", "10²⁵", "10⁵⁰", "10⁷⁵", "10¹⁰⁰"])
    for yv in [0.3, 1, 3, 10, 30]:
        a1.axhline(yv, color=S.FAINT, lw=0.6, zorder=0)
    a1.spines["left"].set_visible(False); a1.spines["bottom"].set_color(S.FAINT)
    a1.set_title("angle between the GD and max-margin directions", color=S.STAR, fontsize=11.5, loc="left", pad=8)
    a1.text(55, 0.6, r"theory: $w(t) \approx \hat w \log t + \tilde w$", color=S.LILAC, fontsize=10)
    # ---------------- norm panel
    a2 = fig.add_axes([xr + 0.035, 0.08, 0.345, 0.24])
    a2.set_xlim(LT0, LT_END)
    a2.set_ylim(0, norm_curve.max() * 1.08)
    a2.plot(lt_th, np.linalg.norm(wh) * np.log(10 ** lt_th), color=S.LILAC, lw=1.1, ls=(0, (4, 3)), alpha=0.9)
    a2.plot(lt_curve[m], norm_curve[m], color=S.STAR, lw=2)
    a2.scatter([lt], [nw], s=50, color=S.STAR, edgecolor=S.NIGHT, linewidth=2, zorder=5)
    a2.set_xticks([0, 25, 50, 75, 100]); a2.set_xticklabels(["1", "10²⁵", "10⁵⁰", "10⁷⁵", "10¹⁰⁰"])
    a2.set_yticks([0, 100, 200]); a2.spines["left"].set_visible(False); a2.spines["bottom"].set_color(S.FAINT)
    for yv in [100, 200]:
        a2.axhline(yv, color=S.FAINT, lw=0.6, zorder=0)
    a2.set_title(r"the weights never stop growing: $\|w\| \approx \|\hat w\|\,\log t$", color=S.STAR, fontsize=11.5, loc="left", pad=8)
    a2.set_xlabel("t  (log scale)", color=S.DIM, fontsize=10)


def _line(ax, w, level, **kw):
    # points x with w.x = level
    nw = np.linalg.norm(w)
    u = w / nw
    p = np.array([-u[1], u[0]])
    s = np.array([-2 * L, 2 * L])
    pts = (level / nw) * u[None] + s[:, None] * p[None]
    ax.plot(pts[:, 0], pts[:, 1], **kw)


if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 9), dpi=100)
    if args.preview:
        for i in [int(k) for k in args.preview.split(",")]:
            draw(i, fig)
            fig.savefig(f"_preview/hero_frame_{i:04d}.png", dpi=100)
            print("_preview/hero_frame", i, lts[i])
    else:
        fdir = "cache/hero_frames"
        shutil.rmtree(fdir, ignore_errors=True); os.makedirs(fdir)
        for i in range(len(lts)):
            draw(i, fig)
            fig.savefig(f"{fdir}/f{i:05d}.png", dpi=100)
        # still (2x) from a late-ish frame
        draw(int(n_main * 0.55), fig)
        fig.savefig("figures/hero_still.png", dpi=200)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{fdir}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart",
                        "figures/hero.mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{fdir}/f%05d.png",
                        "-vf", "fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=sierra2_4a",
                        "figures/hero.gif"], check=True)
        print("wrote hero.mp4 / hero.gif")
