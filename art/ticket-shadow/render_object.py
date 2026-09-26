"""The physical object: a perforated sheet with one hole per surviving first-layer weight,
and a declared simulation of the shadow it casts when lit from behind.

Sheet (declared design):
  - 28 x 28 pixel cells on a CELL_MM pitch; each cell has 300 slots, one per hidden unit;
  - slots lie on a Vogel sunflower spiral (slot k at radius R*sqrt((k+.5)/300), angle k*137.508 deg);
  - hidden unit j occupies the same slot in every cell; slots are assigned in descending order of
    the unit's total surviving input degree (most-connected unit at the centre of every cell);
  - a hole is cut iff weight (pixel i -> unit j) survives. So the sheet IS the first-layer mask
    M1 (784 x 300), folded into image coordinates.

Backlit render (declared physical model, a simulation of an object, not a photograph):
  - a uniform disc source of diameter SRC_MM at distance A_M behind the sheet, wall B_M in front;
  - geometric optics: each hole projects to the wall at magnification (A+B)/A and paints the
    source's image, a disc of diameter SRC*B/A (pinhole-camera blur), weighted by hole area;
  - diffraction approximated by a Gaussian with the Airy-core FWHM 1.03*lambda*B/d (lambda 550 nm);
  - point-source falloff cos^3(theta) across the wall;
  - tone: wall irradiance E -> 1 - exp(-E/E0) with E0 set at the 99.5th percentile of the image
    (declared exposure), warm-white light on a near-black wall.
"""
import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from common import *

CELL_MM = 12.0
MARGIN_MM = 16.0
SLOT_R_MM = 5.5
HOLE_MM = 0.34
GOLDEN = np.deg2rad(137.50776)


def load_mask(cond, m, r):
    zz = np.load(f"{CACHE}/{cond}/round_{r:02d}.npz")
    M1 = np.unpackbits(zz["mask0"], axis=-1)[..., :300].astype(bool)[m]  # (784, 300) in net coords
    perm = zz["perms"][m]
    img = np.zeros_like(M1); img[perm] = M1
    return img, zz


def holes(M1):
    deg = M1.sum(0)
    order = np.argsort(-deg, kind="stable")        # unit order[k] -> slot k
    slot_of_unit = np.empty(300, int); slot_of_unit[order] = np.arange(300)
    k = np.arange(300)
    sr = SLOT_R_MM * np.sqrt((k + 0.5) / 300); sa = k * GOLDEN
    sx, sy = sr * np.cos(sa), sr * np.sin(sa)
    pts = []
    for i in range(784):
        row, col = divmod(i, 28)
        cx = MARGIN_MM + (col + 0.5) * CELL_MM; cy = MARGIN_MM + (row + 0.5) * CELL_MM
        js = np.nonzero(M1[i])[0]
        s = slot_of_unit[js]
        pts.append(np.stack([cx + sx[s], cy + sy[s]], 1))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def write_svg(P, path, title):
    size = 2 * MARGIN_MM + 28 * CELL_MM
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}mm" height="{size}mm" viewBox="0 0 {size} {size}">\n')
        f.write(f'<title>{title}</title>\n')
        f.write('<!-- red hairline: sheet outline (cut). black circles: perforations (cut). '
                f'{len(P)} holes of {HOLE_MM} mm diameter. Units: mm. -->\n')
        f.write(f'<rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="#ff0000" stroke-width="0.05"/>\n')
        f.write('<g fill="none" stroke="#000000" stroke-width="0.02">\n')
        for x, y in P:
            f.write(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{HOLE_MM / 2:.3f}"/>\n')
        f.write('</g>\n</svg>\n')


def sheet_png(P, px_per_mm, path, bg=CREAM, ink=INK):
    size = 2 * MARGIN_MM + 28 * CELL_MM
    S = int(size * px_per_mm)
    ss = 2
    im = Image.new("L", (S * ss, S * ss), 0)
    d = ImageDraw.Draw(im)
    r = HOLE_MM / 2 * px_per_mm * ss
    for x, y in P:
        X, Y = x * px_per_mm * ss, y * px_per_mm * ss
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=255)
    a = np.asarray(im.resize((S, S), Image.LANCZOS), float)[..., None] / 255
    rgb = np.array(bg, float) * (1 - a) + np.array(ink, float) * a
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return S


def wall_render(P, src_mm, a_m, b_m, px, path, light=(255, 238, 205), wall=NIGHT, extent_scale=1.12):
    size = 2 * MARGIN_MM + 28 * CELL_MM
    mag = (a_m + b_m) / a_m
    W_mm = size * mag * extent_scale
    mm_per_px = W_mm / px
    c = size / 2
    wx = (P[:, 0] - c) * mag + W_mm / 2; wy = (P[:, 1] - c) * mag + W_mm / 2
    E = np.zeros((px, px))
    ix = np.clip((wx / mm_per_px).astype(int), 0, px - 1); iy = np.clip((wy / mm_per_px).astype(int), 0, px - 1)
    np.add.at(E, (iy, ix), 1.0)
    # kernel: source image disc (diameter src*b/a) + hole image (diameter hole*mag) + diffraction gaussian
    dsrc = src_mm * b_m / a_m
    dhole = HOLE_MM * mag
    fwhm = 1.03 * 550e-6 * (b_m * 1000) / HOLE_MM
    rad = max(dsrc, dhole) / 2 + 3 * fwhm
    n = int(np.ceil(rad / mm_per_px)) * 2 + 1
    yy, xx = (np.mgrid[0:n, 0:n] - n // 2) * mm_per_px
    rr = np.hypot(xx, yy)
    disc = (rr <= max(dsrc, dhole) / 2).astype(float)
    if disc.sum() == 0:
        disc[n // 2, n // 2] = 1
    sig = fwhm / 2.355 / mm_per_px
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (sig * mm_per_px) ** 2 + 1e-12))
    K = fftconvolve(disc, g, mode="same"); K /= K.sum()
    E = fftconvolve(E, K, mode="same")
    Y, X = (np.mgrid[0:px, 0:px] + 0.5) * mm_per_px - W_mm / 2
    theta = np.arctan(np.hypot(X, Y) / ((a_m + b_m) * 1000))
    E *= np.cos(theta) ** 3
    E = np.maximum(E, 0)
    E0 = np.percentile(E, 99.5) + 1e-12
    t = 1 - np.exp(-E / E0 * 1.6)
    rgb = np.array(wall, float) * (1 - t[..., None]) + np.array(light, float) * t[..., None]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
    return dict(src_mm=src_mm, a_m=a_m, b_m=b_m, magnification=mag, wall_width_m=W_mm / 1000,
                source_blur_mm=dsrc, hole_image_mm=dhole, diffraction_fwhm_mm=fwhm, px=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="mnist_adam_norm_rw0")
    ap.add_argument("--seed_idx", type=int, default=0)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wall_px", type=int, default=3000)
    ap.add_argument("--sheet_px_per_mm", type=float, default=8.6)
    ap.add_argument("--no_sheet", action="store_true")
    a = ap.parse_args()
    z = np.load(f"{CACHE}/{a.cond}/round_00.npz")
    m = [i for i, md in enumerate(z["modes"]) if md == "imp"][a.seed_idx]
    M1, zz = load_mask(a.cond, m, a.round)
    P = holes(M1)
    seed = int(zz["seeds"][m])
    base = f"{GALLERY}/object_{a.cond}_s{seed}_r{a.round:02d}{a.tag}"
    title = (f"The Ticket's Shadow - {a.cond} seed {seed} round {a.round}: {len(P)} holes, "
             f"one per surviving input weight of LeNet-300-100")
    meta = dict(cond=a.cond, seed=seed, round=a.round, holes=int(len(P)),
                sheet_mm=2 * MARGIN_MM + 28 * CELL_MM, cell_mm=CELL_MM, hole_mm=HOLE_MM,
                test_acc_es=float(zz["es_test_acc"][m]), max_holes_per_cell=int(M1.sum(1).max()))
    write_svg(P, base + ".svg", title)
    if not a.no_sheet:
        sheet_png(P, a.sheet_px_per_mm, base + "_sheet.png")
    meta["walls"] = []
    for name, src, am, bm in [("sharp", 2.0, 0.6, 1.2), ("medium", 8.0, 0.6, 1.2), ("soft", 20.0, 0.6, 1.2)]:
        meta["walls"].append(dict(name=name, **wall_render(P, src, am, bm, a.wall_px, base + f"_wall_{name}.png")))
    json.dump(meta, open(base + ".json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
