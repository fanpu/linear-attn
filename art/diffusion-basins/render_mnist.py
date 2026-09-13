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


def upsample_probs(pr, up):
    """bilinear interpolation of measured class probabilities between grid samples (declared), then argmax"""
    from scipy import ndimage
    return np.clip(ndimage.zoom(pr.astype(np.float32), (up, up, 1), order=1, prefilter=False), 0, 1)


def riso(lab, density, ink1="#0078bf", ink2="#ff48b0", off=5, width=2.0):
    """two-spot risograph idiom: ink1 halftone-free density, ink2 boundary line misregistered by `off` px"""
    paper = hx("#f3eee2")
    from render_common import boundary_distance
    cov = np.clip(width + 0.5 - boundary_distance(lab), 0, 1)
    cov = np.roll(np.roll(cov, off, 0), -off, 1)
    t1 = 1 - np.clip(density, 0, 1)[..., None] * (1 - hx(ink1))
    t2 = 1 - cov[..., None] * (1 - hx(ink2))
    return paper * t1 * t2


def digit_legend(img, ground=NIGHT, ink="#d8d4c8", size=28):
    """append a swatch legend of the 10 digit hues (declared categorical palette)"""
    from PIL import Image, ImageDraw, ImageFont
    from render_common import FONT
    W = img.shape[1]
    h = int(size * 2.2)
    strip = Image.new("RGB", (W, h), tuple(int(255 * c) for c in hx(ground)))
    dr = ImageDraw.Draw(strip)
    font = ImageFont.truetype(FONT, size)
    step = min(W // 11, 160)
    x = 18
    dr.text((x, h // 2 - size // 2 - 2), "classifier label:", fill=tuple(int(255 * c) for c in hx(ink)), font=font)
    x += int(font.getlength("classifier label:")) + 24
    step = max(60, (W - x - 18) // 10)
    for d in range(10):
        dr.rectangle([x, h // 2 - size // 2, x + size, h // 2 + size // 2], fill=tuple(int(255 * c) for c in PAL[d]))
        dr.text((x + size + 8, h // 2 - size // 2 - 2), str(d), fill=tuple(int(255 * c) for c in hx(ink)), font=font)
        x += step
    return np.concatenate([img, np.asarray(strip) / 255.0], 0)


def basin():
    z = load("hero")
    if z is None:
        return
    up = 12
    pr = flipud(upsample_probs(z["probs"], up))
    lab, conf = pr.argmax(-1), pr.max(-1)
    cap = ["Which digit: DDIM-50 of an MNIST DDPM (1.28M-param U-Net) run from every noise vector on a norm-preserving "
           "great-sphere slice (radius 28, exp-map coords, +-pi/2 rad, 192x192 samples).",
           "Hue = classifier argmax (declared palette); brightness = classifier max-probability (measured), "
           "bilinearly interpolated between samples; dark seams = class boundaries."]
    shade = confidence_shade((conf - 0.1) / 0.9, floor=0.08, gamma=2.0) * seam_shade(lab, 3.0, 0.15)
    save(caption_strip(digit_legend(colorize(lab, PAL, shade)), cap, ground=NIGHT, ink="#d8d4c8", size=26), "mnist_basin_confidence.png", "mnist")
    save(caption_strip(line_art(lab, pal=PAL, tint_strength=0.35, width=1.8), cap[:1], size=26), "mnist_basin_paper.png", "mnist")
    s = np.sort(pr, -1)
    margin = np.log(s[..., -1] + 1e-6) - np.log(s[..., -2] + 1e-6)
    save(caption_strip(riso(lab, 0.85 * np.exp(-margin / 2.0)), [
        "Riso idiom: blue density = exp(-log-odds margin/2) between top-2 classes (measured; dense = undecided); "
        "pink = class boundaries, deliberately misregistered 5 px (aesthetic)."], ground="#f3eee2", ink="#2b2b2b", size=26),
        "mnist_basin_riso.png", "mnist")
    img = P.mpl.colormaps["Spectral"](np.clip(margin / np.percentile(margin, 99), 0, 1))[..., :3]
    save(caption_strip(img, ["Spectral (sequential, labelled variant): log-odds margin top-1 vs top-2 class; "
                             "dark red = boundary, violet = decisive."], ground=NIGHT, ink="#d8d4c8", size=26),
         "mnist_margin_spectral_sequential.png", "mnist")


def mosaic_boundary(tag="hero", win=24, name="mnist_mosaic_boundary", scale=3):
    """thumbnail mosaic of the win x win window with the most distinct classes (stride 1 = every sample)"""
    z = load(tag)
    if z is None:
        return
    lab_full = flipud(z["probs"].astype(np.float32).argmax(-1))
    best, bij = -1, (0, 0)
    for i in range(0, lab_full.shape[0] - win, 4):
        for j in range(0, lab_full.shape[1] - win, 4):
            w = lab_full[i:i + win, j:j + win]
            sc = len(np.unique(w)) * 1000 + (np.diff(w, axis=0) != 0).sum() + (np.diff(w, axis=1) != 0).sum()
            if sc > best:
                best, bij = sc, (i, j)
    i0, j0 = bij
    im = flipud(z["images"])[i0:i0 + win, j0:j0 + win].astype(np.float32) / 255
    lab = lab_full[i0:i0 + win, j0:j0 + win]
    out = np.zeros((win * 28, win * 28, 3))
    for i in range(win):
        for j in range(win):
            c = PAL[lab[i, j]]
            out[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = im[i, j][..., None] * (0.55 + 0.45 * c) + (1 - im[i, j][..., None]) * c * 0.10
    out = np.kron(out, np.ones((scale, scale, 1)))
    cap = [f"Across a boundary: the actual DDIM-50 samples for every one of the {win}x{win} adjacent noise vectors in the "
           f"window with the most classes (rows {i0}-{i0 + win}, cols {j0}-{j0 + win} of 192). Digits morph continuously; "
           "tint = classifier label (declared palette). "
           "Note the label flips between near-identical digits: the boundary is the classifier's opinion of a smoothly morphing sample."]
    save(caption_strip(digit_legend(out), cap, ground=NIGHT, ink="#d8d4c8", size=26), f"{name}.png", "mnist")
    print("boundary mosaic window", bij, np.unique(lab))


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
        up = 5
        oh = np.eye(40, dtype=np.float32)[z["nn_idx"].astype(int)]
        lab = flipud(upsample_probs(oh, up).argmax(-1))
        d12 = z["nn_d12"].astype(np.float32)
        conf = flipud(upsample_probs((1 - d12[..., 0] / np.maximum(d12[..., 1], 1e-6))[..., None], up)[..., 0])
        tiles.append(colorize(lab, pal, confidence_shade(conf, floor=0.1) * seam_shade(lab, 2.0, 0.2)))
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
    ws = [2 * c[2] for c in z["centres"]]
    cap = ["MNIST DDIM-50 zoom on the great-sphere slice, x4 per panel, each re-centred on a multi-class boundary point: "
           "window width " + ", ".join(f"{w:.2g}" for w in ws) + " rad (128x128 samples each).",
           "The boundary straightens into a single smooth curve between 2 classes by width 0.012 rad: no nested structure. "
           "Hue = classifier label (declared); dark seams = boundary."]
    save(caption_strip(digit_legend(grid_images(tiles, len(tiles), 12, NIGHT)), cap, ground=NIGHT, ink="#d8d4c8", size=24),
         "mnist_zoom_plate.png", "mnist")


if __name__ == "__main__":
    fns = dict(basin=basin, mosaic=mosaic, boundary=mosaic_boundary, memo=memo, steps=steps, zoom=zoom)
    for a in sys.argv[1:]:
        fns[a]()
