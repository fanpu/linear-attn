"""Film (M3): depth dial L = 1 -> 4 on one draw (seed 7, the M2 hero), then a 360 deg turntable of the
L = 1 Heaviside cast. <= 1080 px, ~22 s at 24 fps, MP4 + GIF via r3d.write_film.

Dial (hard cuts only; the four depths are never interpolated into one another):
  L = 1, L = 2: exact 3D crisp-voxel casts, identical camera/level/lighting to gallery/cast_heaviside_L*_voxel.png
    (exterior, no cutaway). Caption prints the measured D: 3-draw mean +- sd, width 4096, 256^3, level = volume
    median, window-calibrated at k = 1 (the M2 report / M3 ruling): L1 2.426 +- 0.031, L2 2.774 +- 0.076, beside
    theory (3 - 2^-L).
  L = 3, L = 4: the 3D box-count estimator saturates at 256^3 (M1/M2: synthetic fields of exact D = 2.938 / 2.969
    already read 2.93 / 2.94, at or above the net's own raw 3D value, so calibration cannot be inverted) and the
    isosurface itself reads as undifferentiated foam from outside (M2 finding) -- so there is no D claim and no
    3D render. One of the 12 slice-atlas coastlines (same code as render_atlas.py's coast_tile, same field and
    level, enlarged) stands in, captioned "saturated".
Turntable: 360 deg about the vertical (z) axis, L = 1 Heaviside, exterior (no cutaway), the same field, level
    and light as the dial's L = 1 frame.

Declared: plaster, one raking light (0.45, -1, 0.75), hard shadow, 32-ray ambient occlusion -- form only, not
data (same as render_casts.py). The L = 3/4 coastline frames use the depth-roughness plotter idiom (paper /
ink) instead, a declared style change signalling that the dial is carried by the slice past L = 2.

python render_film.py [dial|turntable|all] [--size 1080] [--device cuda] [--test]
"""
import argparse, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d
from common import *
sys.path.append("/home/fzeng/ml/research/art/depth-roughness")
from render_common import PAPER, INK, font        # source: art/depth-roughness/render_common.py

CACHE = "cache"
GAL = "gallery"
FPS = 24
HOLD_S = 3.0                      # seconds per dial depth
N_TURN = 240                      # turntable frames, 360 deg / 240 = 1.5 deg/frame, 10 s at 24 fps
CAM = dict(radius=6.0, az_deg=-35, el_deg=32, ortho_height=3.35)
LIGHT = (0.45, -1.0, 0.75)
PLASTER = torch.tensor([0.95, 0.93, 0.89], dtype=torch.float64)
GROUND = 0.975
E = 127.5 * 2 / 256
LO, HI = (-E,) * 3, (E,) * 3

D_MEASURED = {                    # 3-draw mean +- sd, width 4096 256^3, window-calibrated k=1 (M2 report / M3 ruling)
    1: (2.426, 0.031),
    2: (2.774, 0.076),
}
THEORY = {L: 3.0 - 2.0 ** -L for L in (1, 2, 3, 4)}
THEORY_FRAC = {1: "1/2", 2: "1/4", 3: "1/8", 4: "1/16"}   # 2^-L, written as a fraction (matches the diptych caption)


def load_field(L, seed=7):
    """L is the depth (1-4); the cache array's second axis is 0-indexed by depth (l = L - 1)."""
    fn = f"{CACHE}/field_w4096_r256.npy" if seed == 7 else f"{CACHE}/field_w4096_r256_s{seed}.npy"
    V = np.load(fn, mmap_mode="r")
    f = np.asarray(V[0, L - 1]).astype(np.float32)   # ia=0 -> heaviside
    return f, median_level(f)


def shade(pos, nrm, occ):
    ao = r3d.ambient_occlusion(pos, nrm, occ, n_rays=32, radius=0.12, seed=1)
    sh = r3d.hard_shadow(pos, nrm, LIGHT, occ)
    lum = (0.30 + 0.70 * r3d.lambert(nrm, LIGHT, ambient=0.0) * sh) * (0.35 + 0.65 * ao)
    return PLASTER.to(pos.device).expand(len(pos), 3).clone() * lum[:, None].double()


def render_cast(f, u, az_deg, size, device):
    solid = torch.from_numpy(np.ascontiguousarray((f <= u).transpose(2, 1, 0))).to(device)
    cam = r3d.orbit((0, 0, 0), CAM["radius"], az_deg, CAM["el_deg"], width=size, height=size, ortho_height=CAM["ortho_height"])
    hit = r3d.render_voxels(solid.to(torch.int64), LO, HI, cam, solid=solid, device=device)
    m = hit["mask"]
    pos, nrm = hit["pos"][m], hit["normal"][m]
    occ = r3d.voxel_occluder(solid, LO, HI)
    img = torch.full((size, size, 3), GROUND, dtype=torch.float64, device=pos.device)
    img[m.to(pos.device)] = shade(pos, nrm, occ)
    return img.cpu().numpy()


def coast_tile(sl, u, K):
    """Voxel-edge coastline mask (see render_atlas.py), no interpolation."""
    R = sl.shape[0]
    s = sl > u
    M = np.zeros((R * K, R * K), np.float32)
    dv = s[1:, :] != s[:-1, :]
    dh = s[:, 1:] != s[:, :-1]
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


def render_coast_frame(f, u, size, iz=None, top=0.16, side=0.04, bottom=0.04):
    """Coastline tile, drawn below a blank top band reserved for the caption (paper background throughout,
    so the caption text never overlaps the coastline)."""
    R = f.shape[0]
    if iz is None:
        iz = int(np.round(np.linspace(8, 247, 12))[6])       # one of the 12 declared atlas cuts, near z = 0
    top_px, side_px, bot_px = int(top * size), int(side * size), int(bottom * size)
    K = max(1, min((size - 2 * side_px) // R, (size - top_px - bot_px) // R))
    sl = np.ascontiguousarray(f[:, :, iz].T[::-1])
    M = coast_tile(sl, u, K)
    T = M.shape[0]
    img = np.ones((size, size, 3)) * PAPER
    y0 = top_px + (size - top_px - bot_px - T) // 2
    x0 = (size - T) // 2
    img[y0:y0 + T, x0:x0 + T] = PAPER * (1 - M[..., None]) + INK * M[..., None]
    return img, iz


def fit_font(dr, text, kind, fs_max, fs_min, max_w):
    """Largest font size in [fs_min, fs_max] (step 1) for which `text` is <= max_w px wide."""
    for fs in range(fs_max, fs_min - 1, -1):
        f = font(fs, kind)
        if dr.textlength(text, font=f) <= max_w:
            return f
    return font(fs_min, kind)


def caption(img_np, lines, paper_bg):
    im = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    ink = tuple((INK * 255).astype(int)) if paper_bg else (40, 38, 44)
    size = im.size[0]
    margin = size // 40
    max_w = size - 2 * margin
    fs_max1, fs_max2 = max(16, size // 30), max(12, size // 42)
    y = margin
    for i, ln in enumerate(lines):
        kind = "serif" if i == 0 else "italic"
        f = fit_font(dr, ln, kind, fs_max1 if i == 0 else fs_max2, 10, max_w)
        dr.text((margin, y), ln, font=f, fill=ink)
        y += int(f.size * 1.5)
    return im


def dial(size, device, test=False):
    out = f"{CACHE}/frames_dial" + ("_test" if test else "")
    os.makedirs(out, exist_ok=True)
    hold = 4 if test else int(round(HOLD_S * FPS))
    fr = 0
    for L in (1, 2, 3, 4):
        f, u = load_field(L)
        theo = THEORY[L]; frac = THEORY_FRAC[L]
        if L in D_MEASURED:
            m, s = D_MEASURED[L]
            lines = [f"depth L = {L}    measured D = {m:.3f} ± {s:.3f}  (theory 3 − {frac} = {theo:.3f})",
                     "width-4096 Heaviside, 3-draw mean ± sd, calibrated 3D box counting, 256³"]
            img = render_cast(f, u, CAM["az_deg"], size, device)
            im = caption(img, lines, paper_bg=False)
        else:
            lines = [f"depth L = {L}    saturated, no measured D  (theory 3 − {frac} = {theo:.3f})",
                     "3D box counting saturates above D ≈ 2.97 at 256³; a slice-atlas coastline stands in"]
            img, iz = render_coast_frame(f, u, size)
            im = caption(img, lines, paper_bg=True)
        path0 = f"{out}/{fr:05d}.png"
        im.save(path0)
        for k in range(1, hold):
            os.link(path0, f"{out}/{fr + k:05d}.png") if not os.path.exists(f"{out}/{fr + k:05d}.png") else None
        fr += hold
        print("dial", L, "done", flush=True)
    return out, fr


def turntable(size, device, test=False):
    out = f"{CACHE}/frames_turntable" + ("_test" if test else "")
    os.makedirs(out, exist_ok=True)
    n = 6 if test else N_TURN
    f, u = load_field(1)
    m, s = D_MEASURED[1]
    lines = [f"turntable: Heaviside L = 1    D = {m:.3f} ± {s:.3f}  (theory 2.5)",
             "same field, level and light as the dial's first frame"]
    t0 = time.time()
    for i in range(n):
        path = f"{out}/{i:05d}.png"
        if os.path.exists(path):
            continue
        az = CAM["az_deg"] + 360.0 * i / n
        img = render_cast(f, u, az, size, device)
        caption(img, lines, paper_bg=False).save(path)
        if i % 20 == 0:
            print("turntable", i, "/", n, f"{time.time()-t0:.1f}s", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    return out, n


def stitch(dial_out, n_dial, turn_out, n_turn, size, test=False):
    """Concatenate dial then turntable frames into one sequence and write the film."""
    out = f"{CACHE}/frames_film" + ("_test" if test else "")
    os.makedirs(out, exist_ok=True)
    fr = 0
    for i in range(n_dial):
        dst = f"{out}/{fr:05d}.png"
        if not os.path.exists(dst):
            os.link(f"{dial_out}/{i:05d}.png", dst)
        fr += 1
    for i in range(n_turn):
        dst = f"{out}/{fr:05d}.png"
        if not os.path.exists(dst):
            os.link(f"{turn_out}/{i:05d}.png", dst)
        fr += 1
    mp4 = f"{GAL}/film_rough_skin{'_test' if test else ''}.mp4"
    gif = f"{GAL}/film_rough_skin{'_test' if test else ''}.gif"
    r3d.write_film(f"{out}/%05d.png", mp4, fps=FPS, gif=gif, gif_width=480)
    print(mp4, os.path.getsize(mp4) / 1e6, "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["all"])
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    what = a.what
    d_out = t_out = None
    n_d = n_t = 0
    if "dial" in what or "all" in what:
        d_out, n_d = dial(a.size, a.device, a.test)
    if "turntable" in what or "all" in what:
        t_out, n_t = turntable(a.size, a.device, a.test)
    if "all" in what:
        stitch(d_out, n_d, t_out, n_t, a.size, a.test)
