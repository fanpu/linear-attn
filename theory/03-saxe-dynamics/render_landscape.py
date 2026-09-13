"""The scalar toy: f(x) = b*a*x fitting y = s*x.  E(a, b) = 1/2 (s - ab)^2.

Static plate: landscape + gradient-flow streamlines + conserved hyperbolas + time-stamped trajectories,
and the product ab(t) for different init scales.   Animation: a swarm of small inits escaping the saddle.

  .venv/bin/python 03-saxe-dynamics/render_landscape.py [--anim]
"""
import argparse
import multiprocessing as mp
import pathlib
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import saxe_core as sc
import style

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
S = 1.0
LIM = 2.0


def flow(a0, b0, t_end, dt=1e-3, s=S):
    """RK4 gradient flow a' = b(s-ab), b' = a(s-ab). a0, b0 arrays."""
    a, b = np.array(a0, float), np.array(b0, float)
    n = int(t_end / dt)
    out = np.empty((n + 1, 2) + a.shape)
    out[0] = a, b

    def f(a, b):
        e = s - a * b
        return b * e, a * e

    for i in range(n):
        k1 = f(a, b); k2 = f(a + dt / 2 * k1[0], b + dt / 2 * k1[1])
        k3 = f(a + dt / 2 * k2[0], b + dt / 2 * k2[1]); k4 = f(a + dt * k3[0], b + dt * k3[1])
        a = a + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        b = b + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out[i + 1] = a, b
    return out


def background(ax, P, cmap, dark=False):
    g = np.linspace(-LIM, LIM, 600)
    A, B = np.meshgrid(g, g)
    E = 0.5 * (S - A * B) ** 2
    Z = np.log10(E + 1e-3)
    if dark:
        Z = -Z
        ax.imshow(Z, extent=(-LIM, LIM, -LIM, LIM), origin="lower", cmap=cmap, vmin=-1.2, vmax=4.5,
                  interpolation="bilinear", zorder=0)
    else:
        ax.imshow(Z, extent=(-LIM, LIM, -LIM, LIM), origin="lower", cmap=cmap, vmin=-3, vmax=1.0,
                  interpolation="bilinear", zorder=0)
    # conserved quantity a^2 - b^2 = c
    for c in np.linspace(-3, 3, 13):
        if dark:
            break
        if abs(c) < 1e-9:
            ax.plot(g, g, color=P["muted"], lw=0.6, alpha=0.35, zorder=1)
            ax.plot(g, -g, color=P["muted"], lw=0.6, alpha=0.35, zorder=1)
            continue
        ax.contour(A, B, A ** 2 - B ** 2, levels=[c], colors=[P["muted"]], linewidths=0.5, alpha=0.3, zorder=1)
    # hyperbola of global minima ab = s
    x = np.linspace(S / LIM, LIM, 300)
    for sign in (1, -1):
        ax.plot(sign * x, sign * S / x, color=style.ACCENT if not dark else "#ffc15e", lw=3.0, zorder=3,
                solid_capstyle="round")


def static():
    P = style.use("light")
    fig = plt.figure(figsize=(13.5, 6.2))
    ax = fig.add_axes([0.03, 0.08, 0.44, 0.84])
    from matplotlib.colors import LinearSegmentedColormap
    base = cmc.lapaz_r(np.linspace(0.0, 0.62, 256))
    cmap = LinearSegmentedColormap.from_list("lapaz_light", base)
    background(ax, P, cmap)
    g = np.linspace(-LIM, LIM, 40)
    A, B = np.meshgrid(g, g)
    Ea = -(B * (S - A * B)); Eb = -(A * (S - A * B))
    ax.streamplot(A, B, -Ea * 0 + B * (S - A * B), A * (S - A * B), color="#2b3350", linewidth=0.45, density=1.25,
                  arrowsize=0.6, zorder=2)
    # time-stamped trajectories from different init scales
    inits = [(0.55, -0.42, ""), (-0.06, 0.035, ""), (0.004, -0.0022, ""), (1.75, 1.55, "")]
    cols = [P["modes"][0], P["modes"][1], P["modes"][3], P["modes"][2]]
    trajs = []
    for (a0, b0, lab), c in zip(inits, cols):
        tr = flow([a0], [b0], 14.0, dt=1e-3)[:, :, 0]
        trajs.append(tr)
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=2.2, zorder=4)
        idx = np.arange(0, len(tr), 400)
        ax.scatter(tr[idx, 0], tr[idx, 1], s=18, color=c, edgecolor="white", lw=0.8, zorder=5)
    ax.scatter([0], [0], s=150, marker="o", facecolor="none", edgecolor=P["ink"], lw=1.2, zorder=6)
    ax.annotate("saddle at the origin\n(the zero network):\ndots pile up here", (0.02, -0.02), (0.35, -1.05),
                fontsize=10.5, color=P["ink"], arrowprops=dict(arrowstyle="-", color=P["ink"], lw=0.8), zorder=7)
    ax.text(0.62, 1.8, "global minima  $ab = s$", fontsize=11, color=P["ink"], zorder=7)
    ax.text(-1.9, -1.9, "dots: equal steps of time (Δt = 0.4)", fontsize=10, color=P["ink"], va="bottom", zorder=7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xlabel("first-layer weight  $a$"); ax.set_ylabel("second-layer weight  $b$")
    ax.set_title("Loss landscape of  $\\hat y = b\\,a\\,x$  fitting  $y = s\\,x$", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax2 = fig.add_axes([0.57, 0.155, 0.40, 0.69])
    t = np.arange(len(trajs[0])) * 1e-3
    for tr, (a0, b0, lab), c in zip(trajs[:3], inits[:3], cols[:3]):
        u0 = a0 * b0
        ax2.plot(t, tr[:, 0] * tr[:, 1], color=c, lw=2.4)
        ax2.plot(t[::500], sc.sigmoid_mode(t[::500], S, u0) * 0 + np.nan)
        th = sc.first_crossing(t, tr[:, 0] * tr[:, 1], 0.5)
        ax2.annotate(f"$|a_0|\\approx{abs(a0):g}$", (th, 0.5), (th + 0.35, 0.36), fontsize=10.5, color=P["ink"],
                     arrowprops=dict(arrowstyle="-", color=P["muted"], lw=0.7))
    ax2.set_xlim(0, 10); ax2.set_ylim(-0.03, 1.1)
    ax2.axhline(S, color=P["rule"], lw=1, zorder=0)
    ax2.set_xlabel("training time  $t$"); ax2.set_ylabel("network map  $ab$")
    ax2.set_title("Smaller init  →  longer wait at the saddle", pad=10)
    ax2.text(9.9, 0.1, "every 10× smaller init adds the same\ndelay:  $\\Delta t = \\ln(10)/s$",
             ha="right", fontsize=10.5, color=P["muted"])
    style.save(fig, FIG / "landscape.png", dpi=150)


# ------------------------------------------------------------------------------------------------ animation
N_PART = 1400
RNG = np.random.default_rng(4)
A0 = RNG.uniform(-LIM, LIM, N_PART)
B0 = RNG.uniform(-LIM, LIM, N_PART)
LOGR = None  # filled with log escape time
T_ANIM = 7.0
DT = 2e-3
TRAJ = None
COLV = None


def _frame(args):
    fi, ti, out, fade = args
    P = style.use("dark")
    fig = plt.figure(figsize=(9, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    background(ax, P, cmc.oslo, dark=True)
    tail = 60
    i0 = max(0, ti - tail)
    seg = TRAJ[i0: ti + 1: 3]  # (k, 2, N)
    cols = plt.get_cmap(cmc.lipari)(0.25 + 0.72 * COLV)
    if len(seg) > 1:
        segs = np.stack([seg[:-1], seg[1:]], axis=1)  # (k-1, 2, 2, N)
        segs = np.transpose(segs, (3, 0, 1, 2)).reshape(-1, 2, 2)
        k = seg.shape[0] - 1
        al = np.tile(np.linspace(0, 0.75, k), N_PART)
        rgba = np.repeat(cols, k, axis=0); rgba[:, 3] = al
        ax.add_collection(LineCollection(segs, colors=rgba, linewidths=1.1, zorder=4))
    ax.scatter(TRAJ[ti, 0], TRAJ[ti, 1], s=7, c=cols, lw=0, zorder=5)
    ax.scatter([0], [0], s=80, facecolor="none", edgecolor="white", lw=1.2, zorder=6)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.axis("off")
    box = dict(boxstyle="round,pad=0.45", fc=style.NIGHT, ec="none", alpha=0.72)
    ax.text(-1.9, 1.9, f"t = {ti * DT:3.1f}", color=style.NIGHT_INK, fontsize=15, va="top", family="monospace", bbox=box, zorder=9)
    ax.text(-1.9, -1.92, "1,400 initialisations of  ŷ = b·a·x,  gradient flow on  E = ½(s − ab)²\n"
            "bright curves: the minima ab = s   ·   colour: time to converge (lighter = slower)",
            color=style.NIGHT_INK, fontsize=11.5, va="bottom", linespacing=1.5, bbox=box, zorder=9)
    if fade < 1:
        ax.add_patch(plt.Rectangle((-LIM, -LIM), 2 * LIM, 2 * LIM, color=style.NIGHT, alpha=1 - fade, zorder=20))
    fig.savefig(out, dpi=120, facecolor=style.NIGHT)
    plt.close(fig)


def anim():
    global TRAJ
    global COLV
    TRAJ = flow(A0, B0, T_ANIM + 6, dt=DT)
    err = np.abs(TRAJ[:, 0] * TRAJ[:, 1] - S)  # (n, N)
    t_conv = np.array([np.argmax(err[:, j] < 0.05 * S) * DT if np.any(err[:, j] < 0.05 * S) else T_ANIM + 6
                       for j in range(N_PART)])
    COLV = np.clip(np.log(t_conv + 0.05) / np.log(T_ANIM + 6), 0, 1)
    COLV = (COLV - COLV.min()) / (COLV.max() - COLV.min())
    TRAJ = TRAJ[: int(T_ANIM / DT) + 1]
    fr = HERE / "cache/frames_landscape"
    if fr.exists():
        shutil.rmtree(fr)
    fr.mkdir(parents=True)
    fps = 30
    n = int(10 * fps)
    tis = [int(round((k / (n - 1)) * (len(TRAJ) - 1))) for k in range(n)] + [len(TRAJ) - 1] * int(1.0 * fps)
    fades = [min(1.0, (k + 1) / 8) for k in range(n)] + [1.0] * (len(tis) - n)
    tis += [len(TRAJ) - 1] * 12
    fades += [1 - (k + 1) / 12 for k in range(12)]
    jobs = [(k, ti, fr / f"f{k:04d}.png", fd) for k, (ti, fd) in enumerate(zip(tis, fades))]
    with mp.Pool(4) as pool:
        pool.map(_frame, jobs, chunksize=4)
    out = FIG / "landscape_swarm.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fr / "f%04d.png"),
                    "-vf", "scale=900:900", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "slow",
                    "-movflags", "+faststart", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anim", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    if a.anim:
        anim()
    else:
        static()
