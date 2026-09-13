"""Precision floor of a fractal: Mandelbrot zoom into seahorse valley, float32 vs float64 (vs 113-bit long double).

    gcc -O2 -ffp-contract=off -fopenmp mandel.c -o cache/mandel -lm
    python compute_mandel.py zoom     # 480 frames, both precisions -> cache/mandel/frame_XXXX.npz + stats
    python compute_mandel.py deep     # quad-precision reference stills at a few depths

Everything, including pixel coordinates c = centre + (i - W/2 + 0.5) * width/W, is computed in the target
type (C float / double / long double = IEEE binary128 on aarch64), no FMA contraction.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = CACHE / "mandel"
OUT.mkdir(parents=True, exist_ok=True)
CX, CY = "-0.743643887037151", "0.131825904205330"   # seahorse valley (classic zoom target)
W, H = 960, 1080
NF = 480
W0, W1 = 3.0, 3e-15


def run(t, width, maxit, w=W, h=H, threads=4):
    tmp = OUT / f"tmp_{t}.bin"
    subprocess.run([str(CACHE / "mandel"), t, CX, CY, repr(width), str(w), str(h), str(maxit), str(tmp)], check=True,
                   env={"OMP_NUM_THREADS": str(threads)})
    return np.fromfile(tmp, np.float32).reshape(h, w)


def maxit_for(width):
    return int(300 + 350 * max(0.0, np.log10(3.0 / width)))


def distinct_coords(dtype, width, n):
    cx = dtype(float(CX))
    dx = dtype(width) / dtype(n)
    xs = cx + (dtype(np.arange(n) - n // 2) + dtype(0.5)) * dx
    return len(np.unique(xs))


def zoom():
    widths = np.geomspace(W0, W1, NF)
    stats = []
    for i, w in enumerate(widths):
        f = OUT / f"frame_{i:04d}.npz"
        mi = maxit_for(w)
        if not f.exists():
            a = run("f32", w, mi)
            b = run("f64", w, mi)
            np.savez_compressed(f, f32=np.round(a * 8).clip(0, 65535).astype(np.uint16),
                                f64=np.round(b * 8).clip(0, 65535).astype(np.uint16), maxit=mi, width=w)
        d = np.load(f)
        a, b = d["f32"].astype(np.float32) / 8, d["f64"].astype(np.float32) / 8
        stats.append(dict(i=i, width=float(w), maxit=mi,
                          frac_differ=float(np.mean(np.abs(a - b) > 0.5)),
                          distinct_x_f32=distinct_coords(np.float32, w, W), distinct_x_f64=distinct_coords(np.float64, w, W)))
        if i % 20 == 0:
            print(i, stats[-1], flush=True)
    (CACHE / "mandel_stats.json").write_text(json.dumps(stats, indent=1))


def deep():
    res = {}
    for w in (1e-5, 1e-12, 3e-14, 3e-15):
        mi = maxit_for(w)
        arrs = {}
        for t in ("f32", "f64", "f128"):
            arrs[t] = run(t, w, mi, w=1080, h=1080)
        np.savez_compressed(CACHE / f"mandel_deep_{w:.0e}.npz", **arrs, maxit=mi, width=w)
        res[f"{w:.0e}"] = {f"{p}_vs_f128_frac_differ": float(np.mean(np.abs(arrs[p] - arrs["f128"]) > 0.5)) for p in ("f32", "f64")}
        print(w, res[f"{w:.0e}"], flush=True)
    (CACHE / "mandel_deep_stats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"zoom": zoom, "deep": deep}[sys.argv[1]]()
