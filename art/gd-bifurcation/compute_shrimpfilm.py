"""Shrimp zoom film: continuous zoom into the cyclic-step-size (AB) Lyapunov plane, x1 -> x4000.

Every frame is a fresh GPU computation of the Lyapunov exponent of the oscillating mode (balanced-line
sub-dynamics of GD on 1/2(x1x2x3x4-1)^2 with eta_t = A, B, A, B, ...), 1000 burn-in + 1500 averaging steps
(shorter than the 1500 + 3000 used for stills; declared), float64, square frames.
Camera path: log-linear in width, centre moved in proportion to the zoom completed, cosine easing (declared).
Output: cache/shrimpfilm/frame_XXXX.npz
"""
import json
import os
import sys
import time
import numpy as np
from compute_lyapplane import gpu_plane

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
KEYS = [  # (centre A, centre B, width)
    (0.725, 0.725, 0.55),
    (0.7505, 0.7650, 0.055),
    (0.7482, 0.7641, 0.0055),
    (0.747070, 0.761697, 0.001375),
    (0.7466210, 0.7622460, 0.0001375),
]


def path(nseg):
    fr = []
    for j in range(len(KEYS) - 1):
        a, b = KEYS[j], KEYS[j + 1]
        for i in range(nseg):
            f = i / nseg
            f = 0.5 - 0.5 * np.cos(np.pi * f)
            w = np.exp(np.log(a[2]) + f * (np.log(b[2]) - np.log(a[2])))
            g = (a[2] - w) / (a[2] - b[2])
            fr.append((a[0] + g * (b[0] - a[0]), a[1] + g * (b[1] - a[1]), w))
    fr.append(KEYS[-1])
    return fr


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 810
    nseg = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    os.makedirs(f"{CACHE}/shrimpfilm", exist_ok=True)
    fr = path(nseg)
    json.dump(fr, open(f"{CACHE}/shrimpfilm/path.json", "w"))
    t0 = time.time()
    for i, (ca, cb, w) in enumerate(fr):
        fn = f"{CACHE}/shrimpfilm/frame_{i:04d}.npz"
        if os.path.exists(fn):
            continue
        va = ca - w / 2 + w * (np.arange(res) + 0.5) / res
        vb = cb - w / 2 + w * (np.arange(res) + 0.5) / res
        lam = gpu_plane(va, vb, "AB", 1000, 1500, rows_per_chunk=res)
        np.savez_compressed(fn, lam=lam.astype(np.float32), vals_a=va, vals_b=vb)
        if i % 10 == 0:
            print(f"frame {i}/{len(fr)} w={w:.3e} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
