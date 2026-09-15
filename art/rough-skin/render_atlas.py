"""Slice atlas: 12 parallel cuts through the 256^3 cube, each drawn as a 2D coastline line plate in the plotter idiom
of art/depth-roughness (warm paper, near-black ink, URW C059 type; palette and font helpers imported from its
render_common.py, unmodified).

    render_atlas.py [--seed 7]

Declared: cuts are the planes z = const of the exp-map chart (only z = 0 is a totally geodesic great 2-sphere on S^3;
the others are equidistant surfaces, <= 3 % distorted); the coastline is T = u (u = median of the whole 256^3
volume), drawn along voxel edges between 4-neighbours of opposite sign: no interpolation, no smoothing
(spec §0.5); line weight 2 px at 4 px per voxel.
Outputs gallery/atlas_<act>_L<L>.png (12 cuts of one field) and gallery/atlas_depth.png (rows: Heaviside L = 1..4 and
ReLU L = 1; columns: 6 of the cuts).
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.append("/home/fzeng/ml/research/art/depth-roughness")   # appended so our common.py wins
from render_common import PAPER, INK, font      # source: art/depth-roughness/render_common.py
from common import *

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
fn = "cache/field_w4096_r256.npy" if seed == 7 else f"cache/field_w4096_r256_s{seed}.npy"
V = np.load(fn, mmap_mode="r")
K = 4                                   # px per voxel
R = 256
CUTS = np.linspace(8, 247, 12).round().astype(int)
zrad = grid_coords(256)


def coast_tile(sl, u):
    """[R*K, R*K] float ink mask of the voxel-edge coastline of sl > u (rows = y, cols = x after transpose)."""
    s = sl > u
    M = np.zeros((R * K, R * K), np.float32)
    dv = s[1:, :] != s[:-1, :]          # between rows r and r+1 -> horizontal line at y = (r+1)K
    dh = s[:, 1:] != s[:, :-1]          # between cols c and c+1 -> vertical line at x = (c+1)K
    rr, cc = np.nonzero(dv)
    for d in (-1, 0):
        y = (rr + 1) * K + d
        for t in range(K + 1):
            M[y, np.minimum(cc * K + t, R * K - 1)] = 1
    rr, cc = np.nonzero(dh)
    for d in (-1, 0):
        x = (cc + 1) * K + d
        for t in range(K + 1):
            M[np.minimum(rr * K + t, R * K - 1), x] = 1
    return M


def to_rgb(M):
    return (PAPER[None, None] * (1 - M[..., None]) + INK[None, None] * M[..., None])


def plate_one(ia, l):
    vol = np.asarray(V[ia, l]); u = median_level(vol)
    T = R * K; gap = 70; mx = 110; top = 230; bottom = 170
    W = 4 * T + 3 * gap + 2 * mx; H = top + 3 * T + 2 * gap + 2 * 60 + bottom
    img = np.ones((H, W, 3)) * PAPER
    for n, iz in enumerate(CUTS):
        r, c = divmod(n, 4)
        y0 = top + r * (T + gap + 60); x0 = mx + c * (T + gap)
        sl = vol[:, :, iz].T[::-1]       # rows = y (top = +y), cols = x
        img[y0:y0 + T, x0:x0 + T] = to_rgb(coast_tile(np.ascontiguousarray(sl), u))
        img[y0:y0 + T, [x0 - 1, x0 + T]] = INK * 0.5 + PAPER * 0.5
        img[[y0 - 1, y0 + T], x0:x0 + T] = INK * 0.5 + PAPER * 0.5
    im = Image.fromarray((img * 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    ink = tuple((INK * 255).astype(int))
    name = "Heaviside" if ia == 0 else "ReLU"
    theo = "3 − 2^−L = %.4g" % (3 - 2.0 ** -(l + 1)) if ia == 0 else "2 (Kac–Rice class)"
    dr.text((mx, 70), f"{name} network, depth L = {l + 1}, width 4096: twelve parallel cuts", font=font(86), fill=ink)
    dr.text((mx, 170), f"level set T = median through a 0.5 rad exp-map cube on S³ · 256² voxels per cut · theory {theo}",
            font=font(44, "italic"), fill=ink)
    for n, iz in enumerate(CUTS):
        r, c = divmod(n, 4)
        y0 = top + r * (T + gap + 60); x0 = mx + c * (T + gap)
        dr.text((x0, y0 + T + 14), f"z = {zrad[iz]:+.4f} rad", font=font(40, "mono"), fill=ink)
    dr.text((mx, H - 110), "Coastline drawn along voxel edges between opposite signs, no interpolation. "
            "Weights shared across depths and with the ReLU twin; seed %d." % seed, font=font(40, "italic"), fill=ink)
    return im


def plate_depth():
    rows = [(0, 0, "Heaviside L=1"), (0, 1, "Heaviside L=2"), (0, 2, "Heaviside L=3"), (0, 3, "Heaviside L=4"), (1, 0, "ReLU L=1")]
    cols = CUTS[::2]
    k = 3; T = R * k; gap = 40; left = 470; top = 220
    W = left + len(cols) * T + (len(cols) - 1) * gap + 80; H = top + len(rows) * T + (len(rows) - 1) * gap + 140
    img = np.ones((H, W, 3)) * PAPER
    global K
    K0, K = K, k
    for ri, (ia, l, lab) in enumerate(rows):
        vol = np.asarray(V[ia, l]); u = median_level(vol)
        for ci, iz in enumerate(cols):
            y0 = top + ri * (T + gap); x0 = left + ci * (T + gap)
            img[y0:y0 + T, x0:x0 + T] = to_rgb(coast_tile(np.ascontiguousarray(vol[:, :, iz].T[::-1]), u))
            img[y0:y0 + T, [x0 - 1, x0 + T]] = INK * 0.5 + PAPER * 0.5
            img[[y0 - 1, y0 + T], x0:x0 + T] = INK * 0.5 + PAPER * 0.5
    K = K0
    im = Image.fromarray((img * 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    ink = tuple((INK * 255).astype(int))
    dr.text((left, 50), "Rough skin: depth as the dial, one cut at a time", font=font(80), fill=ink)
    dr.text((left, 150), "width-4096 networks, one weight draw; level set T = median; 6 of the 12 parallel cuts; voxel edges, no interpolation",
            font=font(40, "italic"), fill=ink)
    for ri, (_, _, lab) in enumerate(rows):
        dr.text((50, top + ri * (T + gap) + T // 2 - 30), lab, font=font(50), fill=ink)
    for ci, iz in enumerate(cols):
        dr.text((left + ci * (T + gap), H - 110), f"z = {zrad[iz]:+.3f} rad", font=font(40, "mono"), fill=ink)
    return im


if __name__ == "__main__":
    sfx = "" if seed == 7 else f"_s{seed}"
    for ia, l in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]:
        out = f"gallery/atlas_{ACTS[ia]}_L{l + 1}{sfx}.png"
        plate_one(ia, l).save(out); print(out)
    plate_depth().save(f"gallery/atlas_depth{sfx}.png"); print(f"gallery/atlas_depth{sfx}.png")
