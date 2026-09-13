"""Rotating tetrahedron film (C=4): the centred train class means (exactly 3-D) morph through every
checkpoint while the camera turns, then the final state makes a full turn.

  python film_tetra.py c4 [--style brass|plotter] [--fpc 4] [--res 1080] [--workers 4]
Declared: linear tween between consecutive checkpoints; per-checkpoint global scale; fixed dust
exposure taken from the epoch-60 checkpoint; dust = train subset + test samples, class-coloured.
"""
import argparse, os, subprocess
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import nclib as N, artlib as A, render_tetra as T

G = {}
DUST = np.array([A.rgb(c) for c in ["#e8b04a", "#d9694a", "#6fb3c9", "#a7c96a"]])
EXT = (-1.05, 1.05, -1.05, 1.05)


def init(tag):
    meta, mets, ck = N.load_run(tag)
    G.update(ck=ck, meta=meta, y=np.r_[N.labels(meta)[1], N.labels(meta)[0]], mets={round(m["epoch"], 4): m for m in mets})


def state(i):
    if i not in G:
        z = np.load(G["ck"][i][1])
        al = N.Aligner(z["mu"], z["muG"])
        G[i] = (G["ck"][i][0], al.coords_means, np.r_[al(z["h_test"]), al(z["h_train"])], al.residual)
        for k in [k for k in G if isinstance(k, int) and k < i - 2]:
            G.pop(k)
    return G[i]


def dust_img(X, R, px, norm=None):
    xy, _ = T.project3(X, R)
    img = np.zeros((px, px, 3)); s = px / 1400
    for j in range(4):
        img += A.glow(A.splat(xy[G["y"] == j], EXT, px), (0.8 * s, 3 * s, 12 * s), (1, .3, .1))[..., None] * DUST[j]
    q = norm or np.percentile(img.max(-1), 99.7)
    return A.tonemap(img / q, 2.0), q


def frame(args):
    f, i, t, az, out, res, norm, style = args
    e0, V0, X0, r0 = state(i)
    e1, V1, X1, r1 = state(min(i + 1, len(G["ck"]) - 1))
    L = lambda a, b: (1 - t) * a + t * b
    V, X, e, r = L(V0, V1), L(X0, X1), L(e0, e1), L(r0, r1)
    st = T.STYLES[style]; R = T.rot(az, -20)
    fig = A.canvas(res, res, st["bg"])
    ax = A.panel(fig, [0, 0.06, 0.94, 0.94], EXT)
    ax.set_position([0.03, 0.06, 0.94, 0.94])
    if st["dust"]:
        img, _ = dust_img(X, R, int(res * 0.94), norm)
        ax.imshow(A.rgb(st["bg"]) + (1 - A.rgb(st["bg"])) * img, extent=EXT, zorder=1, interpolation="lanczos")
    F, _ = N.fourier_etf(4)
    T.draw_tetra(ax, V, R, st, (6.0 if st.get("tube") else 1.4) * res / 1080, ghost=F)
    Mc = V; nr = np.linalg.norm(Mc, axis=1); cos = (Mc @ Mc.T) / np.outer(nr, nr)
    angs = np.degrees(np.arccos(np.clip([cos[a, b] for a, b in T.EDGES], -1, 1)))
    lab = f"epoch {e:6.2f}" if e >= 1 else f"iteration {round(e * 390.6):3d}"
    fig.text(0.5, 0.022, f"{lab}    max |angle − 109.47°| {np.abs(angs - T.IDEAL_DEG).max():5.2f}°    ETF misfit {r:.3f}",
             color=st["text"], fontsize=6.5 * res / 1080, ha="center", family="monospace")
    fig.savefig(os.path.join(out, f"f{f:05d}.png"), dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--style", default="brass"); ap.add_argument("--fpc", type=int, default=4)
    ap.add_argument("--res", type=int, default=1080); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--turn", type=int, default=360); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    init(a.tag); ck = G["ck"]
    i60 = int(np.argmin([abs(e - 60) for e, _ in ck]))
    _, _, X, _ = state(i60)
    norm = dust_img(X, T.rot(20, -20), int(a.res * 0.94))[1]
    out = os.path.join(N.HERE, "scratch", f"filmtetra_{a.tag}_{a.style}"); os.makedirs(out, exist_ok=True)
    jobs, f, az = [], 0, 20.0
    for i in range(len(ck) - 1):
        for s in range(a.fpc):
            jobs.append((f, i, s / a.fpc, az, out, a.res, norm, a.style)); f += 1; az += 0.5
    for s in range(a.turn):
        jobs.append((f, len(ck) - 1, 0.0, az, out, a.res, norm, a.style)); f += 1; az += 360 / a.turn
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)]
    with Pool(a.workers, initializer=init, initargs=(a.tag,)) as pool:
        for n, _ in enumerate(pool.imap(frame, jobs, chunksize=8)):
            if n % 100 == 0:
                print(n, "/", len(jobs), flush=True)
    if a.limit:
        raise SystemExit
    base = os.path.join(A.GAL, f"film_tetra_{a.tag}_{a.style}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{out}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", base + ".mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{out}/f%05d.png", "-vf",
                    "fps=15,scale=540:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a",
                    base + ".gif"], check=True)
    print(base + ".mp4", base + ".gif")
