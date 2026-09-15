"""M3 films (glow style, CPU): day-tower turntable, month-tower turntable, helix layer sweep (measured | null).
Reads cache/geom_M2.npz + .json only; frames in cache/frames/<film>/, films in gallery/ via r3d.write_film.

  python films.py [--only days months sweep] [--size 1024]

Declared: turntables rotate the orthographic camera once around the tower's vertical axis in 24 s at 30 fps
(el 28 deg), with ortho height fixed to the maximum extent over all azimuths. The layer sweep holds each measured
layer for 1.0 s and linearly interpolates bead and fit positions between consecutive layers for 0.5 s (the
in-between frames are interpolations, not measurements); camera fixed; both panels divided by the measured layer's
RMS in-plane radius, with a declared per-layer zoom
(frame fitted to that layer's beads in both panels, printed as 'zoom'). Frames resume: existing PNGs are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/fzeng/ml/research/art/_shared")
import r3d  # noqa: E402

import common as C  # noqa: E402
import render_m2 as R  # noqa: E402

FPS = 30


def fixed_turntable_cams(scene, S, el, n):
    probe = [R.shared_camera([scene], S, S, az, el) for az in np.linspace(0, 360, 13)[:-1]]
    oh = max(c.ortho_height for c in probe)
    tgt = probe[0].target
    return [r3d.orbit(tgt, 50.0, az_deg=30 + 360 * i / n, el_deg=el, width=S, height=S, ortho_height=oh) for i in range(n)]


def turntable(G, name, S, title):
    scene = R.scene_tower(G, name, "measured")
    n = 24 * FPS
    cams = fixed_turntable_cams(scene, S, 28, n)
    d = C.CACHE / "frames" / f"turntable_{name}"; d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, cam in enumerate(cams):
        f = d / f"{i:05d}.png"
        if f.exists():
            continue
        img = R.style_glow(scene, cam, torch.device("cpu"), S).numpy()
        R.label(img, [title], S, (225, 225, 225), S).save(f)
    dt = time.time() - t0
    r3d.write_film(str(d / "%05d.png"), str(C.ROOT / "gallery" / f"film_turntable_{name}.mp4"), fps=FPS,
                   gif=str(C.ROOT / "gallery" / f"film_turntable_{name}.gif"), gif_width=400)
    return dt


def sweep(G, meta, S, test_layer=None):
    L = G["sweep_measured"].shape[0]
    hold, morph = FPS, FPS // 2
    seq = []                                      # (layer_float) per frame
    for l in range(L):
        seq += [float(l)] * hold
        if l < L - 1:
            seq += [l + (k + 1) / (morph + 1) for k in range(morph)]
    W, H = S // 2, S * 2 // 3
    a = G["a"]

    def scene_at(kind, x):
        l0 = int(np.floor(x)); l1 = min(l0 + 1, L - 1); f = x - l0
        P = (1 - f) * G[f"sweep_{kind}"][l0] + f * G[f"sweep_{kind}"][l1]
        Fc = (1 - f) * G[f"sweepfit_{kind}"][l0] + f * G[f"sweepfit_{kind}"][l1]
        af = np.arange(len(Fc)) / 20.0
        return dict(beads=P.astype(np.float64), value=a, K=100, bead_r=0.045, glow=0.12,
                    lines=[dict(P=P.astype(np.float64), value=a, K=100, r=0.008, closed=False, opaque=False),
                           dict(P=Fc.astype(np.float64), value=af, K=100, r=0.009, closed=False, opaque=True, dim=0.45)])

    # fixed frame = projected extent of the union of every bead over all layers and both panels (declared):
    # nothing leaves the frame, so early layers look smaller than the loose last layers
    # declared per-layer zoom: both panels are centred on their bead mean (the origin); the frame height at layer l
    # is set by the largest projected offset of any bead of layer l in either panel, and is linearly interpolated
    # during morphs. Measured and null always share one camera. The overlay prints the zoom relative to layer 0.
    probe = r3d.orbit((0.0, 0.0, 0.0), 50.0, az_deg=35, el_deg=30, width=W, height=H, ortho_height=2.0)
    ps = probe.pixel_scale()
    oh_layer = []
    for l in range(L):
        pts = torch.tensor(np.concatenate([G[f"sweep_{k}"][l] for k in ("measured", "null")]), dtype=torch.float64)
        pix, _ = probe.project(pts)
        half_w = (pix[:, 0] - W / 2).abs().max().item() * ps
        half_h = (pix[:, 1] - H / 2).abs().max().item() * ps
        oh_layer.append(1.06 * 2 * max(half_h, half_w * H / W))

    def cam_at(x):
        l0 = int(np.floor(x)); l1 = min(l0 + 1, L - 1); f = x - l0
        oh = (1 - f) * oh_layer[l0] + f * oh_layer[l1]
        return r3d.orbit((0.0, 0.0, 0.0), 50.0, az_deg=35, el_deg=30, width=W, height=H, ortho_height=oh), oh
    d = C.CACHE / "frames" / "sweep_helix"
    if test_layer is not None:
        seq, d = [test_layer], Path("/tmp/nk_sweep_test")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, x in enumerate(seq):
        f = d / f"{i:05d}.png"
        if f.exists():
            continue
        cam, oh = cam_at(x)
        panels = [R.style_glow(scene_at(k, x), cam, torch.device("cpu"), H) for k in ("measured", "null")]
        img = torch.cat(panels, 1).numpy()
        lay = int(round(x))
        info = f"layer {x:.1f}" if abs(x - lay) > 1e-6 else f"layer {lay}"
        R.label(img, [f"OLMo-2 T=100 helix, measured - {info}", f"shuffled labels (null) - zoom x{oh_layer[0] / oh:.2f}"],
                H, (225, 225, 225), W).save(f)
    dt = time.time() - t0
    if test_layer is not None:
        return dt, 1
    r3d.write_film(str(d / "%05d.png"), str(C.ROOT / "gallery" / "film_sweep_helix.mp4"), fps=FPS,
                   gif=str(C.ROOT / "gallery" / "film_sweep_helix.gif"), gif_width=480)
    return dt, len(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["days", "months", "sweep"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--test", type=float, default=None, help="render one sweep frame at this layer into /tmp")
    args = ap.parse_args()
    torch.set_num_threads(4)
    G = dict(np.load(C.CACHE / "geom_M2.npz"))
    meta = json.loads((C.CACHE / "geom_M2.json").read_text())
    tf = C.ROOT / "gallery" / "timings_films.json"
    times = json.loads(tf.read_text()) if tf.exists() else {}
    if args.test is not None:
        print(sweep(G, meta, 1080, test_layer=args.test)); return
    if "days" in args.only:
        times["turntable_days_s"] = turntable(G, "days", args.size, "Qwen3-0.6B day tokens, layers 0-28 (bottom to top)")
    if "months" in args.only:
        times["turntable_months_s"] = turntable(G, "months", args.size, "Qwen3-0.6B month tokens, layers 0-28 (bottom to top)")
    if "sweep" in args.only:
        times["sweep_helix_s"], times["sweep_frames"] = sweep(G, meta, 1080)
    tf.write_text(json.dumps(times, indent=1))
    print(times)


if __name__ == "__main__":
    main()
