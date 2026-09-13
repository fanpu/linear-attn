"""Extra pieces from the cache (no GPU):
  python render_extra.py stl        --tag g101        # STL heightfields + preview diptych
  python render_extra.py ridge      --tag g101        # Joy-Division ridgeline of surface rows
  python render_extra.py sweep      --tag g101        # raking-light sweep film (MP4 + GIF)
  python render_extra.py zoom                         # roughness zoom test (plate + json)
  python render_extra.py pca                          # PCA trajectory trail on contour sheets
  python render_extra.py ckfilm                       # 1-D slices through training (plate + film)
  python render_extra.py lines                        # 1-D slice plate for all 4 models + subset check
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
from render_maps import (load, height, upsample, hillshade, loss_levels, frame, LO, HI, CHANCE, PRETTY, TRAIN,
                         PAPER, INK, RED, FONT, SURF, GAL, ORDER)

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIR = ["resnet56", "resnet56_noshort"]


def lsurf(model, tag, ep="final"):
    d = np.load(os.path.join(SURF, f"{model}_{ep}_{tag}.npz"))
    return d


def encode(frames_dir, out_mp4, fps=30, gif_w=540):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{frames_dir}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4], check=True)
    gif = out_mp4.replace(".mp4", ".gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4, "-vf",
                    f"fps=15,scale={gif_w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    gif], check=True)
    print("wrote", out_mp4, gif, f"{os.path.getsize(gif) / 1e6:.1f} MB")


# ----------------------------------------------------------------------------- STL
def write_stl(path, Z, size_mm=150.0, relief_mm=40.0, base_mm=6.0):
    n = Z.shape[0]
    z = (Z - np.log10(LO)) / (np.log10(HI) - np.log10(LO)) * relief_mm + base_mm
    g = np.linspace(0, size_mm, n)
    X, Y = np.meshgrid(g, g)
    V = np.stack([X, Y, z], -1)
    tris = []
    a, b, c, d = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]
    tris += [np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)]
    B = V.copy(); B[..., 2] = 0
    a, b, c, d = B[:-1, :-1], B[:-1, 1:], B[1:, 1:], B[1:, :-1]
    tris += [np.stack([a, c, b], -2).reshape(-1, 3, 3), np.stack([a, d, c], -2).reshape(-1, 3, 3)]
    for top, bot in [(V[0], B[0]), (V[-1], B[-1]), (V[:, 0], B[:, 0]), (V[:, -1], B[:, -1])]:
        t0, t1, b0, b1 = top[:-1], top[1:], bot[:-1], bot[1:]
        tris += [np.stack([t0, b0, b1], 1), np.stack([t0, b1, t1], 1)]
    T = np.concatenate(tris).astype(np.float32)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, T
    with open(path, "wb") as f:
        f.write(b"loss-landscape heightfield".ljust(80, b" ")); f.write(np.uint32(len(T)).tobytes()); f.write(rec.tobytes())
    print("wrote", path, len(T), "triangles")


def stl(tag):
    from matplotlib.colors import LightSource
    fig = plt.figure(figsize=(16, 8.4), facecolor="#141312")
    for i, m in enumerate(PAIR):
        d = lsurf(m, tag)
        X, Y, Z = upsample(d["xs"], d["ys"], height(d["loss"]), 241)
        os.makedirs(os.path.join(GAL, "stl"), exist_ok=True)
        write_stl(os.path.join(GAL, "stl", f"{m}_{tag}.stl"), Z)
        ax = fig.add_axes([0.0 + i * 0.5, 0.08, 0.5, 0.86], projection="3d", facecolor="#141312")
        ls = LightSource(azdeg=300, altdeg=28)
        zz = (Z - np.log10(LO)) / (np.log10(HI) - np.log10(LO)) * 0.27
        rgb = ls.shade(zz, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", ["#b9b2a6", "#b9b2a6"]),
                       vert_exag=1.2, blend_mode="soft", dx=X[1] - X[0], dy=Y[1] - Y[0])
        ax.plot_surface(X, Y, zz, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.view_init(elev=38, azim=-62); ax.set_box_aspect((1, 1, 0.3)); ax.set_axis_off()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 0.27)
        fig.text(0.25 + i * 0.5, 0.07, PRETTY[m], ha="center", color="#d8cdb6", fontsize=15, family=FONT)
    fig.text(0.5, 0.025, "STL heightfield preview (single material, light from NW). Height = log10 training loss, "
             "shared scale [0.08, 150]; 150 mm tiles, 40 mm relief. A 2-D slice, not the landscape.",
             ha="center", color="#9d9384", fontsize=10, family=FONT)
    out = os.path.join(GAL, f"stl_preview_{tag}.png"); fig.savefig(out, dpi=160, facecolor="#141312"); plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------- ridgeline
def ridge(tag, dark=True):
    for m in PAIR:
        d = lsurf(m, tag)
        Z = height(d["loss"])
        xs = np.linspace(-1, 1, 400)
        Zx = ndimage.zoom(Z, (1, 400 / Z.shape[1]), order=3, mode="nearest")
        rows = np.linspace(0, Z.shape[0] - 1, 61).astype(int)
        bg, fg = ("#0c0b0a", "#eee4cf") if dark else (PAPER, INK)
        fig = plt.figure(figsize=(8, 10), facecolor=bg)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor=bg)
        amp = 0.9
        for k, r in enumerate(rows[::-1]):
            base = k * 0.1
            yv = base + amp * (Zx[r] - np.log10(LO)) / (np.log10(HI) - np.log10(LO))
            ax.fill_between(xs, base - 1, yv, color=bg, zorder=k)
            ax.plot(xs, yv, color=fg, lw=0.8, zorder=k + 0.5)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 0.1 * len(rows) + amp); ax.set_axis_off()
        fig.text(0.5, 0.93, PRETTY[m].upper(), ha="center", color=fg, fontsize=16, family=FONT)
        fig.text(0.5, 0.06, f"61 parallel 1-D slices (rows β = const of the {Z.shape[0]}² filter-normalized surface); "
                 "height = log10 training loss, shared scale", ha="center", color=fg, fontsize=7.5, family=FONT)
        out = os.path.join(GAL, f"ridge_{'dark' if dark else 'paper'}_{m}_{tag}.png")
        fig.savefig(out, dpi=260, facecolor=bg); plt.close(fig); print("wrote", out)


# ----------------------------------------------------------------------------- light sweep film
def sweep(tag, nframes=360):
    import imageio.v3 as iio
    fd = os.path.join(ROOT, "cache", "frames_sweep"); os.makedirs(fd, exist_ok=True)
    Zs = []
    for m in PAIR:
        d = lsurf(m, tag)
        X, Y, Z = upsample(d["xs"], d["ys"], height(d["loss"]), 860)
        Zs.append((Z, X[1] - X[0]))
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", 26); small = ImageFont.truetype("DejaVuSerif.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    for f in range(nframes):
        az = 300 + 360.0 * f / nframes
        canvas = np.full((1080, 1920, 3), 10, np.uint8)
        for i, (Z, dx) in enumerate(Zs):
            hs = hillshade(Z, dx, az=az, alt=22, exag=0.12)[::-1]
            rgb = (np.array([0.04, 0.035, 0.03]) + hs[..., None] ** 1.6 * np.array([0.95, 0.88, 0.76]))
            im = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            x0 = 80 + i * 900; canvas[90:90 + im.shape[0], x0:x0 + im.shape[1]] = im
        img = Image.fromarray(canvas); dr = ImageDraw.Draw(img)
        for i, m in enumerate(PAIR):
            dr.text((80 + i * 900 + 430, 1000), PRETTY[m], fill=(216, 205, 182), font=font, anchor="mm")
        dr.text((960, 45), "raking light circling two filter-normalized loss slices  ·  log10 loss, shared scale",
                fill=(150, 140, 125), font=small, anchor="mm")
        img.save(f"{fd}/{f:05d}.png")
    encode(fd, os.path.join(GAL, f"sweep_{tag}.mp4"), fps=30, gif_w=720)


# ----------------------------------------------------------------------------- zoom test
def roughness(h):
    """Relative roughness of a 1-D profile: RMS residual after a cubic fit / (max-min) of the profile,
    plus the count of strict local extrema of the residual-free profile (sign changes of the slope
    whose neighbouring |diff| exceed 1e-4 of the window range)."""
    t = np.linspace(-1, 1, len(h))
    res = h - np.polyval(np.polyfit(t, h, 3), t)
    rng = h.max() - h.min()
    dh = np.diff(h); thr = 1e-4 * max(rng, 1e-12)
    sig = np.sign(dh[np.abs(dh) > thr])
    return float(np.sqrt(np.mean(res ** 2)) / max(rng, 1e-12)), int(np.sum(sig[1:] != sig[:-1])), float(rng)


def zoom():
    tags = ["zoom0", "zoom1", "zoom2", "zoom3"]
    half = [0.5, 0.05, 0.005, 0.0005]
    table = {}
    fig, axs = plt.subplots(len(tags), 2, figsize=(11, 10), facecolor=PAPER)
    for j, m in enumerate(PAIR):
        table[m] = []
        for i, t in enumerate(tags):
            p = os.path.join(SURF, f"{m}_final_{t}.npz")
            ax = axs[i, j]; ax.set_facecolor(PAPER)
            if not os.path.exists(p):
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center"); continue
            d = np.load(p); L = d["loss"]; a = d["xs"]
            h = np.log10(L)
            r, ext, rng = roughness(h)
            table[m].append(dict(half_width=half[i], rel_rough=r, extrema=ext, log10_range=rng,
                                 dtype=json.loads(str(d["meta"]))["dtype"]))
            ax.plot(a, L, color=INK, lw=0.8)
            ax.set_title(f"{PRETTY[m]}  ·  α ∈ 0.5 ± {half[i]:g}  ·  rough {r:.2e}, extrema {ext}", fontsize=8,
                         family=FONT, color=INK)
            ax.tick_params(labelsize=6, colors=INK)
            for s in ax.spines.values():
                s.set_color(INK); s.set_linewidth(0.6)
    fig.suptitle("Zoom test along β = 0 around α = 0.5: does the no-shortcut roughness persist at finer spacing?\n"
                 "roughness = RMS residual after cubic fit / window range of log10 loss (201 points per window; "
                 "two finest windows in float64)", fontsize=9, family=FONT, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(GAL, "zoom_test.png"); fig.savefig(out, dpi=220, facecolor=PAPER); plt.close(fig)
    json.dump(table, open(os.path.join(ROOT, "cache", "zoom_test.json"), "w"), indent=1)
    print(json.dumps(table, indent=1)); print("wrote", out)


# ----------------------------------------------------------------------------- PCA trail
def pca():
    for m, tag, dname in [("resnet56", "pca41", "resnet56_pca"), ("resnet56_noshort", "pcaf10_41", "resnet56_noshort_pca_from10")]:
        p = os.path.join(SURF, f"{m}_final_{tag}.npz")
        if not os.path.exists(p):
            print("missing", p); continue
        d = np.load(p); D = torch.load(os.path.join(ROOT, "cache", "dirs", f"{dname}.pt"), weights_only=False)
        xs, ys = d["xs"], d["ys"]
        X, Y, Z = upsample(xs, ys, height(d["loss"]), 500)
        fig = plt.figure(figsize=(10, 8.2), facecolor=PAPER)
        ax = fig.add_axes([0.07, 0.15, 0.86, 0.76], facecolor=PAPER)
        minor, index = loss_levels(10)
        ax.contour(X, Y, Z, levels=minor, colors=INK, linewidths=0.25)
        cs = ax.contour(X, Y, Z, levels=index, colors=INK, linewidths=0.8)
        ax.clabel(cs, fmt=lambda v: f"{10 ** v:g}", fontsize=6)
        c, eps = D["coords"], D["epochs"]
        ax.plot(c[:, 0], c[:, 1], color=RED, lw=1.4, marker="o", ms=3)
        for (u, v), e in zip(c, eps):
            if e in (0, 1, 2, 5, 10, 15, 20, 30, 40):
                ax.annotate(f"ep {e}", (u, v), xytext=(4, 4), textcoords="offset points", fontsize=7, color=RED, family=FONT)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_aspect("equal")
        ax.tick_params(labelsize=7, colors=INK)
        ex = D["explained"]
        ax.set_title(f"{PRETTY[m]}: training trail on the PCA plane of its own checkpoints", fontsize=13, family=FONT, color=INK)
        fig.text(0.5, 0.07, f"Directions = top-2 PCs of (w_epoch − w_final) over epochs {eps[0]}–{eps[-1]} "
                 f"(explained variance {ex[0]:.1%}, {ex[1]:.1%}); unit-norm, not filter-normalized. "
                 "Heights: log10 train loss on 1000 images at the final BN statistics.", ha="center", fontsize=7.5, family=FONT, color=INK)
        fig.text(0.5, 0.045, "The trail lies in the plane only approximately (residual variance is off-plane); "
                 "the surface is evaluated with final-epoch BN running statistics.", ha="center", fontsize=7.5, family=FONT, color=INK)
        out = os.path.join(GAL, f"pca_trail_{m}.png"); fig.savefig(out, dpi=240, facecolor=PAPER); plt.close(fig)
        print("wrote", out)


# ----------------------------------------------------------------------------- checkpoint slices
def ckfilm(nf_between=12):
    eps = [1, 2, 3, 4, 5, 6, 8, 10, 13, 15, 16, 20, 25, 30, 35, 40]
    data = {}
    for m in PAIR:
        rows = []
        for e in eps:
            p = os.path.join(SURF, f"{m}_ep{e:03d}_ckline.npz")
            if os.path.exists(p):
                d = np.load(p); rows.append((e, d["xs"], np.log10(np.clip(d["loss"], LO, HI))))
        data[m] = rows
    have = min(len(v) for v in data.values())
    if have < 2:
        print("not enough checkpoint slices yet", {k: len(v) for k, v in data.items()}); return
    # plate: ridgeline over epochs
    fig, axs = plt.subplots(1, 2, figsize=(12, 9), facecolor="#0c0b0a")
    for ax, m in zip(axs, PAIR):
        ax.set_facecolor("#0c0b0a")
        for k, (e, a, h) in enumerate(data[m][:have]):
            base = (have - 1 - k) * 0.35
            y = base + (h - np.log10(LO)) / np.log10(HI / LO) * 2.2
            ax.fill_between(a, base - 3, y, color="#0c0b0a", zorder=k)
            ax.plot(a, y, color="#efe3c8", lw=0.9, zorder=k + 0.5)
            ax.text(-1.08, base + 0.05, f"ep {e}", color="#9d9384", fontsize=7, ha="right", family=FONT)
        ax.set_axis_off(); ax.set_title(PRETTY[m], color="#efe3c8", fontsize=14, family=FONT)
    fig.text(0.5, 0.03, "1-D filter-normalized slice (seed-1 direction, re-normalized to each checkpoint) through the weights "
             "at each saved epoch; top = epoch 1. Height = log10 train loss.", ha="center", color="#9d9384", fontsize=8, family=FONT)
    out = os.path.join(GAL, "ckpt_ridge.png"); fig.savefig(out, dpi=220, facecolor="#0c0b0a"); plt.close(fig); print("wrote", out)
    # film: morph between successive epochs (linear interpolation in log-loss between measured slices: declared)
    fd = os.path.join(ROOT, "cache", "frames_ck"); os.makedirs(fd, exist_ok=True)
    for f in glob.glob(f"{fd}/*.png"):
        os.remove(f)
    fi = 0
    for k in range(have - 1):
        for s in np.linspace(0, 1, nf_between, endpoint=False):
            fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100, facecolor="#0c0b0a")
            for ax, m in zip(axs, PAIR):
                ax.set_facecolor("#0c0b0a")
                for g in range(max(0, k - 6), k + 1):
                    ax.plot(data[m][g][1], data[m][g][2], color="#efe3c8", lw=0.6, alpha=0.08 + 0.05 * (g - k + 6))
                e0, a, h0 = data[m][k]; e1, _, h1 = data[m][k + 1]
                ax.plot(a, (1 - s) * h0 + s * h1, color="#f2c46d", lw=2.2)
                ax.set_ylim(np.log10(LO), np.log10(HI)); ax.set_xlim(-1, 1)
                ax.set_title(PRETTY[m], color="#efe3c8", fontsize=20, family=FONT)
                ax.tick_params(colors="#6d655a", labelsize=10)
                for sp in ax.spines.values():
                    sp.set_color("#3a352f")
                ax.set_yticks(np.log10([0.1, 1, 10, 100])); ax.set_yticklabels(["0.1", "1", "10", "100"])
            e0, e1 = data[PAIR[0]][k][0], data[PAIR[0]][k + 1][0]
            fig.text(0.5, 0.94, f"epoch {e0 + s * (e1 - e0):4.1f}", ha="center", color="#f2c46d", fontsize=22, family=FONT)
            fig.text(0.5, 0.03, "train loss (log) along one filter-normalized direction through the current weights; "
                     "frames between saved epochs are linear blends (declared)", ha="center", color="#8d8374", fontsize=12, family=FONT)
            fig.savefig(f"{fd}/{fi:05d}.png", facecolor="#0c0b0a"); plt.close(fig); fi += 1
    encode(fd, os.path.join(GAL, "ckpt_slices.mp4"), fps=24, gif_w=720)


def lines():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PAPER); ax.set_facecolor(PAPER)
    sty = {"resnet20": ("#6b5d4f", "-"), "resnet20_noshort": ("#6b5d4f", "--"), "resnet56": (INK, "-"), "resnet56_noshort": (RED, "-")}
    for m in ORDER:
        p = os.path.join(SURF, f"{m}_final_line.npz")
        if os.path.exists(p):
            d = np.load(p); ax.semilogy(d["xs"], d["loss"], color=sty[m][0], ls=sty[m][1], lw=1.1, label=PRETTY[m])
    for n, ls in [(1000, ":"), (5000, "-.")]:
        p = os.path.join(SURF, f"resnet56_noshort_final_line_n{n}.npz")
        if os.path.exists(p):
            d = np.load(p); ax.semilogy(d["xs"], d["loss"], color="#2f6aa3", ls=ls, lw=0.9, label=f"ResNet-56 no-short, n={n} (subset check)")
    ax.axhline(CHANCE, color="#999", lw=0.6); ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("α (filter-normalized units, seed-1 direction)", family=FONT); ax.set_ylabel("train loss", family=FONT)
    out = os.path.join(GAL, "slices_1d.png"); fig.savefig(out, dpi=220, facecolor=PAPER); plt.close(fig); print("wrote", out)
    a = [f"resnet56_noshort_final_line_n{n}.npz" for n in (1000, 5000)]
    if all(os.path.exists(os.path.join(SURF, x)) for x in a):
        l1, l5 = [np.load(os.path.join(SURF, x))["loss"] for x in a]
        print("subset check: median |log10 ratio| n1000 vs n5000 =", float(np.median(np.abs(np.log10(l1 / l5)))),
              "max", float(np.max(np.abs(np.log10(l1 / l5)))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--tag", default="g101")
    a = ap.parse_args()
    {"stl": lambda: stl(a.tag), "ridge": lambda: (ridge(a.tag, True), ridge(a.tag, False)), "sweep": lambda: sweep(a.tag),
     "zoom": zoom, "pca": pca, "ckfilm": ckfilm, "lines": lines}[a.what]()
