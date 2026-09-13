"""Finite-width Heaviside networks: (A) the tile patch at 1024^2 for a width x depth grid,
(B) nested zoom windows around a boundary point for the width-cutoff measurement.
Outputs cache/nets_patch.npz, cache/nets_zoom_n<width>.npz"""
import sys, time, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(0.10)
from common import *
from nets import HeavisideNet

part = sys.argv[1]
SEED = 11
if part == "A":
    C0 = np.array([0.3, -0.5, 0.8]); HW = 0.7
    v = lambert_patch(C0, HW, 1024)
    out = {}
    plan = [(1, [64, 256, 1024, 4096, 16384, 65536]), (2, [64, 256, 1024, 4096, 16384]), (3, [64, 256, 1024, 4096])]
    for L, widths in plan:
        for n in widths:
            t0 = time.time()
            T = HeavisideNet(n, L, SEED)(v)
            out[f"L{L}_n{n}"] = T.astype(np.float64)
            s, c = boxcount(T, np.median(T))
            print(L, n, f"{time.time()-t0:.1f}s", c[:6], flush=True)
    np.savez("cache/nets_patch.npz", **out)

if part == "B":
    widths = [int(w) for w in sys.argv[2].split(",")]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    CENTER = np.array([0.2, 0.6, 0.77])
    for n in widths:
        net = HeavisideNet(n, L, SEED)
        t0 = time.time()
        e1, e2, c = rot_frame(CENTER)
        F0 = np.pi / 4
        # coarse window, median level, boundary crossing along the central row, bisection
        s = (np.arange(512) + 0.5) / 512 * F0 - F0 / 2
        row = c + s[:, None] * e1
        row /= np.linalg.norm(row, axis=1, keepdims=True)
        Tr = net(row)
        vv, _, _ = gnomonic_patch(CENTER, F0 / 2, 256)
        u = float(np.median(net(vv)))
        sg = np.sign(Tr - u)
        idx = np.nonzero(sg[:-1] != sg[1:])[0]
        i = idx[np.argmin(np.abs(idx - 256))]
        a, b = row[i], row[i + 1]
        fa = np.sign(net(a[None])[0] - u)
        for it in range(80):
            m = (a + b) / 2; m /= np.linalg.norm(m)
            fm = np.sign(net(m[None])[0] - u)
            if fm == fa: a = m
            else: b = m
            if np.linalg.norm(a - b) < 1e-15: break
        Ta, Tb = net(a[None])[0], net(b[None])[0]
        u = float((Ta + Tb) / 2)
        p = (a + b) / 2; p /= np.linalg.norm(p)
        rows = []
        js = list(range(0, 13))
        Fs = [F0 * 4.0 ** -j for j in js]
        counts, fields = [], {}
        for j, F in zip(js, Fs):
            npx = 512 if n * F < 5000 else 256
            vv, _, _ = gnomonic_patch(p, F / 2, npx)
            T = net(vv)
            sz, cnt = boxcount(T, u)
            cc = np.zeros(10, int); cc[:len(cnt)] = cnt
            counts.append(cc); rows.append(npx)
            if npx == 512:
                fields[f"f{j}"] = T.astype(np.float32)
            print(n, j, f"F={F:.2e} npx={npx}", cnt[:6], f"{time.time()-t0:.0f}s", flush=True)
        np.savez(f"cache/nets_zoom_L{L}_n{n}.npz", u=u, p=p, F=np.array(Fs), npx=np.array(rows),
                 counts=np.stack(counts), **fields)
