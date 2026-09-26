"""Follow-up (2026-09-26 night): THREE SHADOWS OF THE SAME DATA — the triptych object.

SGD (a soft core), Adam (a flat plateau over every pixel ever lit), signum (only the rim survives).
Same LeNet-300-100, same raw [0,1] MNIST, same IMP schedule, same round 15 (3.5% of layer 1: 8,275
surviving input weights in every ticket, by construction of layer-wise pruning). Only the optimiser differs.

Sources (all cached, no new training):
  SGD    lr 0.1            cache/mnist_sgd_raw_rw0     (job 728), seeds 0,1,2
  Adam   lr 1.2e-3, e 1e-8 cache/mnist_adam_raw_rw0    (job 664), seeds 0,1,2
  signum lr 1e-4, b 0.9    cache/followup_mechsign     (job 873) seed 0, cache/followup_signlr (job 880) seed 1

Outputs (gallery/, all new, prefixed followup_triptych_):
  _{sgd,adam,signum}.svg   laser-cut sheets, seed 0, render_object.py geometry unchanged (368 mm, 12 mm cells,
                           300-slot sunflower per cell, 0.34 mm holes, slots by descending unit degree)
  _{..}_sheet.png          the sheet as cut (holes as ink on cream)
  _wall[_src20mm].png      the three walls side by side (single walls in cache/followup_triptych_walls/); simulated backlit wall, render_object.py light model (8 mm disc source 0.6 m behind,
                           wall 1.2 m in front, pinhole + Airy blur, cos^3 falloff) with ONE shared exposure
                           for all three walls (declared): E0 = the 99.5th percentile of the three pooled.
  _plate.png               the ink-on-cream plate (seed-mean shadows), house register of the diptych.

    python followup_triptych.py [--scale panel|shared] [--src 8]
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from common import *
import render_object as RO

SETS = [  # key, label, one-line reading, (cond, cfg-or-None) per seed
    ("sgd", "SGD", "a soft core: the pixels that vary most",
     [("mnist_sgd_raw_rw0", None, 0), ("mnist_sgd_raw_rw0", None, 1), ("mnist_sgd_raw_rw0", None, 2)]),
    ("adam", "ADAM", "a flat plateau: every pixel ever lit",
     [("mnist_adam_raw_rw0", None, 0), ("mnist_adam_raw_rw0", None, 1), ("mnist_adam_raw_rw0", None, 2)]),
    ("signum", "SIGNUM", "a rim: the pixels almost never lit",
     [("followup_mechsign", "signum_lr1e-4", 0), ("followup_signlr", "signum_lr1e-4", 1)]),
]
BINS = [0, 1, 10, 100, 1000, 5000, 20000, 60000]


def load_seed(cond, cfg, seed, r=15):
    z = np.load(f"{CACHE}/{cond}/round_{r:02d}.npz")
    idx = [i for i in range(len(z["modes"])) if z["modes"][i] == "imp" and int(z["seeds"][i]) == seed
           and (cfg is None or z["cfg"][i] == cfg)]
    assert len(idx) == 1, (cond, cfg, seed, idx)
    m = idx[0]
    M1 = np.unpackbits(z["mask0"], axis=-1)[..., :300].astype(bool)[m]
    img = np.zeros_like(M1); img[z["perms"][m]] = M1
    z0 = np.load(f"{CACHE}/{cond}/round_00.npz")
    return img, float(z["es_test_acc"][m]), float(z0["es_test_acc"][m])


def wall_E(P, src_mm, a_m, b_m, px, extent_scale=1.12):
    """render_object.wall_render's irradiance, returned before tone-mapping (so exposure can be shared)."""
    size = 2 * RO.MARGIN_MM + 28 * RO.CELL_MM
    mag = (a_m + b_m) / a_m
    W_mm = size * mag * extent_scale; mm_per_px = W_mm / px; c = size / 2
    wx = (P[:, 0] - c) * mag + W_mm / 2; wy = (P[:, 1] - c) * mag + W_mm / 2
    E = np.zeros((px, px))
    np.add.at(E, (np.clip((wy / mm_per_px).astype(int), 0, px - 1), np.clip((wx / mm_per_px).astype(int), 0, px - 1)), 1.0)
    dsrc = src_mm * b_m / a_m; dhole = RO.HOLE_MM * mag
    fwhm = 1.03 * 550e-6 * (b_m * 1000) / RO.HOLE_MM
    rad = max(dsrc, dhole) / 2 + 3 * fwhm
    n = int(np.ceil(rad / mm_per_px)) * 2 + 1
    yy, xx = (np.mgrid[0:n, 0:n] - n // 2) * mm_per_px
    disc = (np.hypot(xx, yy) <= max(dsrc, dhole) / 2).astype(float)
    if disc.sum() == 0:
        disc[n // 2, n // 2] = 1
    sig = fwhm / 2.355 / mm_per_px
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (sig * mm_per_px) ** 2 + 1e-12))
    K = fftconvolve(disc, g, mode="same"); K /= K.sum()
    E = fftconvolve(E, K, mode="same")
    Y, X = (np.mgrid[0:px, 0:px] + 0.5) * mm_per_px - W_mm / 2
    E *= np.cos(np.arctan(np.hypot(X, Y) / ((a_m + b_m) * 1000))) ** 3
    return np.maximum(E, 0), dict(src_mm=src_mm, a_m=a_m, b_m=b_m, magnification=mag, wall_width_m=W_mm / 1000,
                                  source_blur_mm=dsrc, diffraction_fwhm_mm=fwhm, px=px)


def tone(E, E0, light=(255, 238, 205), wall=NIGHT):
    t = 1 - np.exp(-E / E0 * 1.6)
    rgb = np.array(wall, float) * (1 - t[..., None]) + np.array(light, float) * t[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="panel")   # shared: one vmax for all three panels; panel: per-panel max
    ap.add_argument("--src", type=float, default=8.0)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--wall_px", type=int, default=2400)
    ap.add_argument("--no_objects", action="store_true")
    a = ap.parse_args()
    from imp import load_dataset
    xtr = load_dataset("mnist")[0][:55000]
    cnt = (xtr > 0).sum(0); std = xtr.std(0); lit = (cnt > 0).astype(float)

    data = {}
    for key, lab, reading, seeds in SETS:
        masks, acc, acc0 = [], [], []
        for cond, cfg, s in seeds:
            M, ac, ac0 = load_seed(cond, cfg, s)
            masks.append(M); acc.append(ac); acc0.append(ac0)
        sh = np.stack([m.sum(1) for m in masks]).astype(float)
        mean = sh.mean(0)
        bins = [float(mean[(cnt >= BINS[i]) & (cnt < BINS[i + 1])].mean()) for i in range(len(BINS) - 1)]
        data[key] = dict(label=lab, reading=reading, M0=masks[0], shadow=mean, shadow0=sh[0], n=len(masks),
                         holes=[int(m.sum()) for m in masks], acc=acc, acc0=acc0, bins=bins,
                         r_std=corr(mean, std), r_lit=corr(mean, lit), r_seed0_mean=corr(sh[0], mean),
                         r_seeds=float(np.mean([corr(sh[i], sh[j]) for i in range(len(sh)) for j in range(i + 1, len(sh))])),
                         max_cell=int(sh[0].max()))
    keys = [k for k, *_ in SETS]
    rx = {f"{p}~{q}": corr(data[p]["shadow"], data[q]["shadow"]) for i, p in enumerate(keys) for q in keys[i + 1:]}

    # ---------- the plate ----------
    vmax_shared = max(d["shadow"].max() for d in data.values())
    panel = 28 * a.cell; fs = panel // 60
    margin = panel // 7; gap = panel // 8; head = fs * 7; foot = fs * 11
    W = 2 * margin + 3 * panel + 2 * gap; H = margin + head + panel + foot + margin // 2
    im = Image.new("RGB", (W, H), CREAM); d = ImageDraw.Draw(im)
    text(d, (margin, margin), "THREE SHADOWS OF THE SAME DATA", int(fs * 1.8))
    text(d, (margin, margin + int(fs * 2.6)),
         "LeNet-300-100 lottery tickets on MNIST, pixels in [0,1], round 15 of iterative magnitude pruning: 8,275 input "
         "weights (3.5%) survive in every ticket. Only the optimiser differs.", fs, MUTED)
    for k, key in enumerate(keys):
        o = data[key]
        x = margin + k * (panel + gap); y = margin + head
        im.paste(dots(o["shadow"], a.cell, vmax=vmax_shared if a.scale == "shared" else None), (x, y))
        b = o["bins"]
        text(d, (x, y + panel + fs), o["label"], int(fs * 1.5))
        text(d, (x + panel, y + panel + int(fs * 1.35)), o["reading"], fs, INK, anchor="ra")
        text(d, (x, y + panel + int(fs * 3.2)),
             f"pixel lit in 10–99 of 55,000 images keeps {b[2]:.1f} connections; lit in >20,000 keeps {b[6]:.1f}", fs, MUTED)
        acc = 100 * np.mean(o["acc"]); acc0 = 100 * np.mean(o["acc0"])
        text(d, (x, y + panel + int(fs * 4.8)),
             f"r(pixel std) {o['r_std']:.2f} · r(ever lit) {o['r_lit']:.2f} · mean of {o['n']} seeds", fs, MUTED)
        worse = acc < acc0 - 0.5
        text(d, (x, y + panel + int(fs * 6.4)),
             f"ticket test {acc:.1f}% (dense {acc0:.1f}%)" + ("  — a worse ticket" if worse else ""), fs,
             INK if worse else MUTED)  # madder is reserved for negative values in this project
    scale_note = (f"dot area = seed-mean connections kept / {vmax_shared:.0f} (one scale for all three panels, so each carries the same total ink)"
                  if a.scale == "shared" else "dot area = seed-mean connections kept / max in panel")
    text(d, (margin, H - margin // 2 - int(fs * 1.2)),
         scale_note + ".  signum = sign of an EMA momentum (β 0.9), lr 1e-4; SGD lr 0.1; Adam lr 1.2e-3.", fs, MUTED)
    out = f"{GALLERY}/followup_triptych_plate{'' if a.scale == 'panel' else '_sharedscale'}.png"
    im.save(out); print(out, im.size)
    if a.no_objects:
        return

    # ---------- the objects ----------
    meta = dict(round=15, cross_r=rx, walls={}, sheets={})
    Es = {}
    for key in keys:
        o = data[key]
        P = RO.holes(o["M0"])
        base = f"{GALLERY}/followup_triptych_{key}"
        RO.write_svg(P, base + ".svg", f"Three Shadows - {o['label']} ticket, seed 0, round 15: {len(P)} holes, "
                                       "one per surviving input weight of LeNet-300-100 on raw MNIST")
        RO.sheet_png(P, 8.6, base + "_sheet.png")
        E, geo = wall_E(P, a.src, 0.6, 1.2, a.wall_px)
        Es[key] = E; meta["walls"] = geo
        meta["sheets"][key] = dict(svg=base + ".svg", holes=int(len(P)), test_acc_es=o["acc"][0],
                                   max_holes_per_cell=o["max_cell"], r_seed0_vs_mean=o["r_seed0_mean"])
    E0 = float(np.percentile(np.concatenate([E.ravel() for E in Es.values()]), 99.5))
    meta["shared_E0"] = E0
    meta["per_wall_E99.5"] = {k: float(np.percentile(E, 99.5)) for k, E in Es.items()}
    meta["per_wall_peak_over_E0"] = {k: float(E.max() / E0) for k, E in Es.items()}
    walls = [tone(Es[k], E0) for k in keys]
    sfx = "" if a.src == 8.0 else f"_src{a.src:g}mm"
    for k, w in zip(keys, walls):
        os.makedirs(f"{CACHE}/followup_triptych_walls", exist_ok=True); w.save(f"{CACHE}/followup_triptych_walls/followup_triptych_{k}_wall{sfx}.png")
    # the three walls hung side by side, same exposure, dark ground, no apparatus
    px = a.wall_px; g = px // 10
    strip = Image.new("RGB", (3 * px + 4 * g, px + 2 * g), NIGHT)
    for k, w in enumerate(walls):
        strip.paste(w, (g + k * (px + g), g))
    strip.save(f"{GALLERY}/followup_triptych_wall{sfx}.png"); print("wall strip", strip.size)
    summary = {k: {kk: vv for kk, vv in data[k].items() if kk not in ("M0", "shadow", "shadow0")} for k in keys}
    json.dump(dict(meta=meta, conditions=summary), open(f"{CACHE}/followup_triptych{sfx}.json", "w"), indent=1)
    print(json.dumps(dict(meta=meta, conditions=summary), indent=1))


if __name__ == "__main__":
    main()
