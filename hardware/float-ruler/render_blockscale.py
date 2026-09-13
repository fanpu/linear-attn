"""Render the block-scale pieces from cache/blockscale.npz.

    python render_blockscale.py fan      # staircase-fan stills (engraved, observatory, riso)
    python render_blockscale.py video    # sliding-comb MP4 + GIF

Exact: comb positions (scale x E2M1 values), scale codes, quantised values, relative MSE.
Aesthetic: colours, fonts, layout; the 32-value Laplace block and the gain sweep are illustrative inputs.
"""
import subprocess
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from plate import CACHE, GALLERY, MONO, SERIF, STACK, STYLES

D = dict(np.load(CACHE / "blockscale.npz"))
QPOS = D["E2M1"][1:]  # 0.5 ... 6


def fan(style):
    st = STYLES[style]
    fig = plt.figure(figsize=(16, 11), dpi=300, facecolor=st["bg"])
    am = D["fan_amax"]
    la = np.log2(am)
    riso = style == "riso"
    panels = [("MXFP4", D["fan_mx_scale"], "E8M0 block scale  2^(floor(log2 amax) - 2)", st["ink"])]
    panels.append(("NVFP4", D["fan_nv_scale"], "E4M3 block scale  round_E4M3(amax/6 * s_enc) / s_enc", st["ink2"] if riso else st["ink"]))
    if riso:
        axes = [fig.add_axes([0.08, 0.1, 0.86, 0.74])] * 2
    else:
        axes = [fig.add_axes([0.07, 0.1, 0.42, 0.74]), fig.add_axes([0.54, 0.1, 0.42, 0.74])]
    for ax, (name, scale, desc, col) in zip(axes, panels):
        ax.set_facecolor(st["bg"])
        for q in QPOS:
            y = np.log2(scale * q)
            lw = 2.2 if q in (1.0, 2.0, 4.0) else 1.1
            ax.plot(la, y, color=col, lw=lw, alpha=0.9 if not riso else 0.85, solid_capstyle="butt",
                    drawstyle="steps-post")
        ax.plot(la, la, color=st["ink"] if not riso else "#333333", lw=0.9, ls=(0, (4, 3)), alpha=0.8)
        if not riso or name == "MXFP4":
            ax.fill_between(la, la, np.log2(scale * 6), where=la > np.log2(scale * 6),
                            color=st["ink2"] if not riso else st["ink"], alpha=0.18, lw=0, step="post")
        ax.set_xlim(-2, 6)
        ax.set_ylim(-5, 7)
        for sp in ax.spines.values():
            sp.set_color(st["faint"])
        ax.tick_params(colors=st["ink"], labelsize=10)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_family(MONO)
        ax.set_xlabel("log2 block amax", family=MONO, color=st["ink"], fontsize=12)
        if not riso:
            ax.set_title(f"{name}\n{desc}", family=SERIF, color=st["ink"], fontsize=17, loc="left")
    axes[0].set_ylabel("log2 representable magnitude  (scale x {0.5, 1, 1.5, 2, 3, 4, 6})", family=MONO,
                       color=st["ink"], fontsize=12)
    if riso:
        axes[0].text(-1.8, 6.4, "MXFP4  (E8M0 power-of-two scale, block 32)", color=st["ink"], family=SERIF, fontsize=17)
        axes[0].text(-1.8, 5.8, "NVFP4  (E4M3 scale x per-tensor FP32 scale, block 16)", color=st["ink2"], family=SERIF, fontsize=17)
    fig.text(0.07, 0.95, "The comb and its scale", family=SERIF, fontsize=34, color=st["ink"])
    fig.text(0.07, 0.885, "Seven FP4 teeth ride on the block scale. A power-of-two scale moves the whole comb by octaves; "
                          "an E4M3 scale moves it in eighth-octave steps.\nDashed: the block's own max. "
                          "Shaded: magnitudes above the top tooth, clipped to 6X by MX Algorithm 1.",
             family=SERIF, fontsize=12.5, color=st["ink"], style="italic")
    fig.text(0.96, 0.02, STACK + f"  ·  NVFP4 s_enc = 6*448/amax_tensor, amax_tensor = 2^6", family=MONO, fontsize=8,
             color=st["ink"], ha="right")
    p = GALLERY / f"blockscale_fan_{style}.png"
    fig.savefig(p, facecolor=st["bg"])
    plt.close(fig)
    print("wrote", p)


def frame(i, style="observatory"):
    st = STYLES[style]
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=st["bg"])
    base, g = D["base"], D["gains"][i]
    v = base * g
    lo, hi = -5.0, 6.2
    ax = fig.add_axes([0.06, 0.34, 0.9, 0.5])
    ax.set_facecolor(st["bg"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.3, 4.1)
    ax.axis("off")
    c1, c2 = "#f0b44c", "#6fb7d9"
    blockcol = np.where(np.arange(32) < 16, 0, 1)
    colors = np.array([c1, c2])[blockcol]
    rows = [(2.75, "MXFP4  (one block of 32)", D["mx_scale"][i], D["mx_q"][i], np.arange(32),
             f"E8M0 code {int(D['mx_code'][i])}  ->  scale 2^{int(np.log2(D['mx_scale'][i]))}"),
            (1.4, "NVFP4  block 1 (16)", D["nv_scale"][i][0], D["nv_q"][i], np.arange(16),
             f"E4M3 code {D['nv_code'][i][0]:g}  ->  scale {D['nv_scale'][i][0]:.4f}"),
            (0.05, "NVFP4  block 2 (16)", D["nv_scale"][i][1], D["nv_q"][i], np.arange(16, 32),
             f"E4M3 code {D['nv_code'][i][1]:g}  ->  scale {D['nv_scale'][i][1]:.4f}")]
    for y, name, scale, q, idx, lab in rows:
        teeth = np.log2(scale * QPOS)
        top = 0.5
        segs = [[(x, y), (x, y + (top if t in (1.0, 2.0, 4.0) else 0.34))] for x, t in zip(teeth, QPOS)]
        ax.add_collection(LineCollection(segs, colors=st["ink"], linewidths=2.6))
        ax.plot([lo, hi], [y, y], color=st["faint"], lw=0.8)
        for x, t in zip(teeth, QPOS):
            if lo < x < hi:
                ax.text(x, y - 0.06, f"{t:g}", ha="center", va="top", family=MONO, fontsize=9, color="#8f94a3")
        yd = y + 0.95
        xd = np.log2(np.abs(v[idx]))
        for j, xj in zip(idx, xd):
            if q[j] != 0:
                ax.plot([xj, np.log2(abs(q[j]))], [yd - 0.05, y + top + 0.03], color=colors[j], lw=0.9, alpha=0.55)
            else:
                ax.plot([xj, xj], [yd - 0.05, y + 0.1], color=colors[j], lw=0.9, alpha=0.3, ls=":")
        ax.scatter(xd, np.full(len(idx), yd), s=34, c=list(colors[idx]), zorder=3, edgecolors="none")
        ax.text(lo, yd + 0.12, name, family=SERIF, fontsize=15, color=st["ink"])
        ax.text(hi, yd + 0.12, lab, family=MONO, fontsize=11, color=st["ink"], ha="right")
    # traces
    axt = fig.add_axes([0.06, 0.07, 0.42, 0.2])
    axm = fig.add_axes([0.54, 0.07, 0.42, 0.2])
    n = len(D["gains"])
    xs = np.arange(n)
    for a in (axt, axm):
        a.set_facecolor(st["bg"])
        for sp in a.spines.values():
            sp.set_color(st["faint"])
        a.tick_params(colors=st["ink"], labelsize=9)
        a.set_xlim(0, n)
    axt.plot(xs, np.log2(D["gains"] * np.abs(D["base"]).max()), color=st["faint"], lw=1, ls="--")
    axt.step(xs[: i + 1], np.log2(D["mx_scale"][: i + 1] * 6), color=st["ink"], lw=1.8, where="post")
    axt.step(xs[: i + 1], np.log2(D["nv_scale"][: i + 1, 0] * 6), color=c1, lw=1.2, where="post")
    axt.step(xs[: i + 1], np.log2(D["nv_scale"][: i + 1, 1] * 6), color=c2, lw=1.2, where="post")
    axt.set_ylim(-1.5, 5)
    axt.set_title("top tooth log2(6 x scale): MXFP4 (white) vs NVFP4 blocks; dashed = amax", family=MONO,
                  fontsize=10, color=st["ink"], loc="left")
    axm.semilogy(xs[: i + 1], D["mx_relmse"][: i + 1], color=st["ink"], lw=1.6)
    axm.semilogy(xs[: i + 1], D["nv_relmse"][: i + 1], color=c1, lw=1.6)
    axm.set_ylim(1e-3, 0.2)
    axm.set_title("relative MSE of the 32 values: MXFP4 (white) vs NVFP4 (amber)", family=MONO, fontsize=10,
                  color=st["ink"], loc="left")
    fig.text(0.06, 0.93, "Sliding the comb", family=SERIF, fontsize=30, color=st["ink"])
    fig.text(0.06, 0.885, "log axis: the scale translates the FP4 comb. E8M0 moves it only by whole octaves; "
                          "E4M3 moves it in eighth-octave steps.", family=SERIF, fontsize=14, style="italic", color=st["ink"])
    fig.text(0.96, 0.93, f"gain 2^{D['u'][i]:.2f}", family=MONO, fontsize=18, color=st["ink"], ha="right")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _job(i):
    from PIL import Image
    Image.fromarray(frame(i)).save(CACHE / "slide" / f"{i:05d}.png")
    return i


def video():
    (CACHE / "slide").mkdir(exist_ok=True)
    with Pool(4) as p:
        list(p.imap_unordered(_job, range(len(D["gains"])), chunksize=6))
    mp4 = GALLERY / "blockscale_slide.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(CACHE / "slide" / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True)
    gif = GALLERY / "blockscale_slide.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(CACHE / "slide" / "%05d.png"),
                    "-vf", "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=sierra2_4a",
                    str(gif)], check=True)
    print("wrote", mp4, gif)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fan"
    if what == "fan":
        for s in sys.argv[2:] or ["engraved", "observatory", "riso"]:
            fan(s)
    elif what == "frame":
        from PIL import Image
        Image.fromarray(frame(int(sys.argv[2]))).save("/tmp/slide_test.png")
    else:
        video()
