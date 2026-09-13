"""Nebula-condensing film: C=10 features in the four {10/k} Fourier planes over all checkpoints.

  python film_stars.py c10 [--fpc 8] [--res 1080] [--workers 4]
Each checkpoint is aligned independently onto the SAME ideal frame (class order fixed), so no flips.
Declared: linear per-sample tween between consecutive checkpoints (fpc frames each; checkpoint
spacing is roughly logarithmic in epoch so time is not linear); per-frame global scale (feature
norm grows ~100x); fixed exposure per plane taken from the epoch-10 checkpoint.
"""
import argparse, os, subprocess, types
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import nclib as N, artlib as A, render_stars as RS

G = {}


def proj(i):
    if i not in G:
        e, p = G["ck"][i]
        z = np.load(p)
        al = N.Aligner(z["mu"], z["muG"])
        G[i] = (e, al.coords_means, al(z["h_train"]), al(z["h_test"]), al.residual)
        if len(G) > 8:
            G.pop(min(k for k in G if isinstance(k, int)))
    return G[i]


def init(tag):
    meta, mets, ck = N.load_run(tag)
    G.update(ck=ck, meta=meta, mets={round(m["epoch"], 4): m for m in mets})


def frame(args):
    f, i, t, out, res, norm = args
    e0, m0, tr0, te0, r0 = proj(i)
    e1, m1, tr1, te1, r1 = proj(min(i + 1, len(G["ck"]) - 1))
    lerp = lambda a, b: (1 - t) * a + t * b
    C = m0.shape[0]
    F, planes = N.fourier_etf(C)
    al = types.SimpleNamespace(C=C, F=F, planes=planes, coords_means=lerp(m0, m1))
    ytr, yte = N.labels(G["meta"])
    d = dict(al=al, Xtr=lerp(tr0, tr1), Xte=lerp(te0, te1), ytr=ytr, yte=yte, norm=norm)
    fig = RS.render(d, "night", [1, 2, 3, 4], res)
    e = lerp(e0, e1)
    near = G["mets"].get(round(e0 if t < 0.5 else e1, 4), {})
    txt = f"epoch {e:7.2f}    ETF misfit {lerp(r0, r1):.3f}"
    if near:
        txt += f"    NC1 train {near['nc1']:.3g}  test {near['nc1_test']:.3g}"
    fig.text(0.5, 0.004, txt, color="#9a978c", fontsize=6.2 * res / 1080, ha="center", va="bottom", family="monospace")
    for (x, y), k in zip([(0.03, 0.965), (0.52, 0.965), (0.03, 0.475), (0.52, 0.475)], [1, 2, 3, 4]):
        fig.text(x, y, f"{{10/{k}}}", color="#5c5b60", fontsize=6 * res / 1080, ha="left", va="top", family="monospace")
    fig.savefig(os.path.join(out, f"f{f:05d}.png"), dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--fpc", type=int, default=8); ap.add_argument("--res", type=int, default=1080)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--hold", type=int, default=60)
    a = ap.parse_args()
    init(a.tag)
    ck = G["ck"]
    # fixed exposure: auto-norm of the checkpoint nearest epoch 10
    i10 = int(np.argmin([abs(e - 10) for e, _ in ck]))
    e, m, tr, te, r = proj(i10)
    F, planes = N.fourier_etf(len(m))
    d = dict(al=types.SimpleNamespace(C=len(m), F=F, planes=planes, coords_means=m), Xtr=tr, Xte=te,
             ytr=N.labels(G["meta"])[0], yte=N.labels(G["meta"])[1])
    plt.close(RS.render(d, "night", [1, 2, 3, 4], a.res))
    norm = float(np.median(list(d["norm_used"].values())))
    out = os.path.join(N.HERE, "scratch", f"film_{a.tag}"); os.makedirs(out, exist_ok=True)
    jobs, f = [], 0
    for i in range(len(ck) - 1):
        for s in range(a.fpc):
            jobs.append((f, i, s / a.fpc, out, a.res, norm)); f += 1
    for s in range(a.hold):
        jobs.append((f, len(ck) - 1, 0.0, out, a.res, norm)); f += 1
    with Pool(a.workers, initializer=init, initargs=(a.tag,)) as pool:
        for n, _ in enumerate(pool.imap_unordered(frame, jobs, chunksize=4)):
            if n % 100 == 0:
                print(n, "/", len(jobs), flush=True)
    mp4 = os.path.join(A.GAL, f"film_stars_{a.tag}_night.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{out}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow", mp4], check=True)
    gif = os.path.join(A.GAL, f"film_stars_{a.tag}_night.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{out}/f%05d.png", "-vf",
                    "fps=8,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    gif], check=True)
    print(mp4, gif, "exposure norm", norm)
