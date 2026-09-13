"""Render MNIST pieces from cache/mnist_*.npz (no recomputation).

  python render_mnist.py basin mosaic memo steps zoom
"""
import os
import sys

import numpy as np

from common import CACHE
from render_common import (DIGITS, NIGHT, PAPER, P, caption_strip, colorize, confidence_shade, flipud, grid_images, hx,
                           line_art, memo_palette, save, seam_shade)
from render_toy import frames_to_video, split_orientation  # noqa: F401

PAL = np.stack([hx(c) for c in DIGITS])


def load(tag):
    f = f"{CACHE}/mnist_{tag}.npz"
    if not os.path.exists(f):
        print("missing", f)
        return None
    return np.load(f)


def basin():
    z = load("hero")
    if z is None:
        return
    pr = z["probs"].astype(np.float32)
    lab, conf = flipud(pr.argmax(-1)), flipud(pr.max(-1))
    up = 6
    labu, confu = np.kron(lab, np.ones((up, up), int)), np.kron(conf, np.ones((up, up)))
    cap = ["Which digit: DDIM-50 of an MNIST DDPM from every noise vector on a norm-preserving great-sphere slice "
           "(radius 28, exp-map coords, +-pi/2 rad).",
           "Hue = classifier argmax (declared palette), brightness = classifier max-probability (measured)."]
    save(caption_strip(colorize(labu, PAL, confidence_shade((confu - 0.1) / 0.9, floor=0.08, gamma=2.0)), cap, ground=NIGHT,
                       ink="#d8d4c8", size=20), "mnist_basin_confidence.png", "mnist")
    save(caption_strip(line_art(labu, pal=PAL, tint_strength=0.5, width=1.4), cap[:1], size=20), "mnist_basin_paper.png", "mnist")
    # Spectral split: margin between top-2 classes, sign = which side of the 0|1-ranked boundary is not meaningful for
    # 10 classes, so use the Spectral map on log-odds margin as a sequential labelled variant.
    s = np.sort(pr, -1)
    margin = flipud(np.log(s[..., -1] + 1e-6) - np.log(s[..., -2] + 1e-6))
    img = P.mpl.colormaps["Spectral"](np.clip(margin / np.percentile(margin, 99), 0, 1))[..., :3]
    save(np.kron(img, np.ones((up, up, 1))), "mnist_margin_spectral_sequential.png", "mnist")


def mosaic(tag="hero", name="mnist_mosaic", stride=2, tint=0.35):
    """every stride-th pixel's generated digit, placed at its slice position; cell tinted by class hue"""
    z = load(tag)
    if z is None:
        return
    im = flipud(z["images"])[::stride, ::stride].astype(np.float32) / 255
    key = z["nn_idx"] if "nn_idx" in z.files and tag.startswith("memo") else z["probs"].argmax(-1)
    lab = flipud(key)[::stride, ::stride]
    pal = memo_palette(np.load(f"{CACHE}/mnist_memo_digits.npy")) if tag.startswith("memo") and os.path.exists(
        f"{CACHE}/mnist_memo_digits.npy") else PAL
    n = im.shape[0]
    out = np.zeros((n * 28, n * 28, 3))
    for i in range(n):
        for j in range(n):
            c = pal[lab[i, j] % len(pal)]
            out[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = im[i, j][..., None] * ((1 - tint) + tint * c) + (1 - im[i, j][..., None]) * 0.05
    save(out, f"{name}.png", "mnist")


def memo():
    zn, ze = load("memo_hero"), load("memo_exact")
    if zn is None:
        return
    from mnist import load_mnist
    ck = __import__("torch").load(f"{CACHE}/mnist_memo.pt", map_location="cpu")
    _, y = load_mnist(True)
    digits = y[ck["train_idx"]].numpy()
    np.save(f"{CACHE}/mnist_memo_digits.npy", digits)
    pal = memo_palette(digits)
    tiles = []
    for z, nm in [(zn, "network (U-Net, 40 images)"), (ze, "exact empirical score")]:
        if z is None:
            continue
        lab = flipud(z["nn_idx"]).astype(int)
        d12 = flipud(z["nn_d12"]).astype(np.float32)
        conf = 1 - d12[..., 0] / np.maximum(d12[..., 1], 1e-6)
        up = 5
        tiles.append(colorize(np.kron(lab, np.ones((up, up), int)), pal,
                              np.kron(confidence_shade(conf, floor=0.1), np.ones((up, up)))))
    g = grid_images(tiles, len(tiles), 16, NIGHT)
    cap = ["Memorisation cells: which of the 40 training images DDIM-50 returns, network (left) vs exact empirical score (right).",
           "Hue = digit class, lightness = instance (declared); brightness = 1 - d1/d2 nearest-neighbour margin (measured)."]
    save(caption_strip(g, cap, ground=NIGHT, ink="#d8d4c8", size=20), "memo_cells.png", "mnist")
    mosaic("memo_hero", "memo_mosaic", stride=2)


def steps():
    z = load("stepsweep")
    if z is None:
        return
    pr = z["probs"].astype(np.float32)
    up = 8

    def frame(i):
        lab = flipud(pr[i].argmax(-1))
        img = colorize(np.kron(lab, np.ones((up, up), int)), PAL, np.kron(confidence_shade(flipud(pr[i].max(-1))), np.ones((up, up))))
        return caption_strip(img, [f"DDIM steps = {int(z['steps'][i])}"], ground=NIGHT, ink="#d8d4c8", size=22)
    frames_to_video(lambda i: frame(min(i // 12, len(pr) - 1)), 12 * len(pr) + 24, "mnist_steps", "mnist")


def zoom():
    z = load("zoom")
    if z is None:
        return
    tiles = []
    for k in range(z["probs"].shape[0]):
        lab = flipud(z["probs"][k].astype(np.float32).argmax(-1))
        tiles.append(colorize(np.kron(lab, np.ones((4, 4), int)), PAL, np.kron(seam_shade(lab, 3, 0.3), np.ones((4, 4)))))
    save(grid_images(tiles, len(tiles), 12, NIGHT), "mnist_zoom_plate.png", "mnist")


if __name__ == "__main__":
    fns = dict(basin=basin, mosaic=mosaic, memo=memo, steps=steps, zoom=zoom)
    for a in sys.argv[1:]:
        fns[a]()
