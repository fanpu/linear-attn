"""MNIST basin maps on a norm-preserving great-sphere slice of the 784-d noise space.

Slice (declared): exponential-map / azimuthal-equidistant coordinates on the great 2-sphere spanned by
an orthonormal triple (e0, u, v) (QR of a seeded Gaussian 784x3 matrix), scaled to radius sqrt(784)=28:
    z(a, b) = 28 [cos(rho) e0 + sin(rho)/rho (a u + b v)],  rho = sqrt(a^2 + b^2)  (radians)
Every pixel is a noise vector of norm exactly sqrt(d), i.e. on the Gaussian shell.

  python mnist_compute.py map   <kind> <tag> <R> <steps> [a0 b0 hw]   kind = ddpm | memo | memo_exact
  python mnist_compute.py stepsweep <R>
  python mnist_compute.py zoom <R> <levels>
  python mnist_compute.py uncert
  python mnist_compute.py f64check
"""
import json
import math
import sys
import time

import numpy as np
import torch

from common import CACHE, ddim, gpu_setup, great_sphere, orthonormal_triple
from mnist import load_clf, load_mnist, load_unet, unet_eps_fn

DEV = gpu_setup()
D = 784
RADIUS = math.sqrt(D)
SLICE_SEED = 2024
HW0 = math.pi / 2
BATCH = 2048


def slice_points(a0, b0, hw, R, seed=SLICE_SEED, dtype=torch.float64):
    e0, u, v = orthonormal_triple(D, seed, DEV, dtype)
    a = a0 + hw * ((torch.arange(R, device=DEV, dtype=dtype) + 0.5) / R * 2 - 1)
    b = b0 + hw * ((torch.arange(R, device=DEV, dtype=dtype) + 0.5) / R * 2 - 1)
    B, A = torch.meshgrid(b, a, indexing="ij")
    return great_sphere(e0, u, v, A.flatten(), B.flatten(), RADIUS)


class Empirical:
    """exact denoiser of the empirical distribution of the training images (a GMM with s = 0)"""

    def __init__(self, x):
        self.y = x.reshape(len(x), -1).double()

    def eps(self, x, sigma):
        xd = x.double()
        d2 = torch.cdist(xd, self.y) ** 2
        w = torch.softmax(-d2 / (2 * sigma ** 2), 1)
        Dx = w @ self.y
        return ((xd - Dx) / sigma).to(x.dtype)


def get_model(kind, dtype=torch.float32):
    if kind in ("ddpm", "memo"):
        net, ck = load_unet(kind, DEV, dtype)
        return unet_eps_fn(net, dtype=dtype), ck
    if kind == "memo_exact":
        _, ck = load_unet("memo", DEV)
        x, _ = load_mnist(True, DEV)
        return Empirical(x[ck["train_idx"].to(DEV)]).eps, ck
    raise ValueError(kind)


@torch.no_grad()
def evaluate(kind, Z, steps, dtype=torch.float32, keep_images=True):
    eps, ck = get_model(kind, dtype)
    clf = load_clf(DEV)
    xtr = None
    if kind.startswith("memo"):
        x, _ = load_mnist(True, DEV)
        xtr = x[ck["train_idx"].to(DEV)].reshape(len(ck["train_idx"]), -1)
    probs, imgs, nn_idx, nn_d = [], [], [], []
    for i in range(0, len(Z), BATCH):
        z = Z[i:i + BATCH].to(dtype)
        out = ddim(eps, z, steps).float()
        img = out.clamp(-1, 1)
        probs.append(torch.softmax(clf(img.reshape(-1, 1, 28, 28)), 1).half().cpu())
        if keep_images:
            imgs.append(((img + 1) * 127.5).round().to(torch.uint8).reshape(-1, 28, 28).cpu())
        if xtr is not None:
            d = torch.cdist(out, xtr.float())
            s, idx = d.sort(1)
            nn_idx.append(idx[:, 0].to(torch.uint8).cpu())
            nn_d.append(torch.stack([s[:, 0], s[:, 1]], 1).half().cpu())
    res = dict(probs=torch.cat(probs).numpy())
    if keep_images:
        res["images"] = torch.cat(imgs).numpy()
    if xtr is not None:
        res["nn_idx"] = torch.cat(nn_idx).numpy()
        res["nn_d12"] = torch.cat(nn_d).numpy()
    return res


def task_map(kind, tag, R, steps, a0=0.0, b0=0.0, hw=HW0, dtype=torch.float32):
    t0 = time.time()
    Z = slice_points(a0, b0, hw, R)
    res = evaluate(kind, Z, steps, dtype)
    res = {k: v.reshape((R, R) + v.shape[1:]) for k, v in res.items()}
    np.savez_compressed(f"{CACHE}/mnist_{tag}.npz", a0=a0, b0=b0, hw=hw, R=R, steps=steps, kind=kind,
                        slice_seed=SLICE_SEED, wall=time.time() - t0, **res)
    lab = res["probs"].argmax(-1)
    print(f"map {tag}: {kind} R={R} steps={steps} ({time.time()-t0:.0f}s) class hist {np.bincount(lab.flatten(), minlength=10)}",
          flush=True)


STEPS_SWEEP = [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45]


def task_stepsweep(R):
    t0 = time.time()
    Z = slice_points(0.0, 0.0, HW0, R)
    probs, imgs = [], []
    for n in STEPS_SWEEP:
        r = evaluate("ddpm", Z, n)
        probs.append(r["probs"].reshape(R, R, 10)); imgs.append(r["images"].reshape(R, R, 28, 28))
        print(f"stepsweep n={n} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(f"{CACHE}/mnist_stepsweep.npz", steps=np.array(STEPS_SWEEP[:len(probs)]), probs=np.stack(probs),
                            images=np.stack(imgs), R=R, hw=HW0)


def task_zoom(R, levels, factor=4.0, steps=50, kind="ddpm", tag="zoom"):
    """centre: a boundary point of the coarse map found by bisection along the segment between two
    neighbouring differently-labelled pixels nearest the slice centre; re-centred at every level."""
    t0 = time.time()
    base = np.load(f"{CACHE}/mnist_{'hero' if kind == 'ddpm' else 'memo_hero'}.npz")
    key = "probs" if kind == "ddpm" else "nn_idx"
    lab = base[key].argmax(-1) if kind == "ddpm" else base[key]
    Rb, hwb = int(base["R"]), float(base["hw"])
    a0, b0 = 0.0, 0.0
    centres, stacks = [], {}
    L = lab
    for k in range(1, levels + 1):
        hw = hwb / factor ** k
        hwl, Rl = hwb / factor ** (k - 1), L.shape[0]
        yy, xx = np.nonzero(L[:, 1:] != L[:, :-1])  # boundary between pixel xx and xx+1
        pa = a0 - hwl + (xx + 1.0) / Rl * 2 * hwl
        pb = b0 - hwl + (yy + 0.5) / Rl * 2 * hwl
        j = int(np.argmin(np.hypot(pa - a0, pb - b0)))
        a0, b0 = float(pa[j]), float(pb[j])
        Z = slice_points(a0, b0, hw, R)
        r = evaluate(kind, Z, steps, keep_images=True)
        L = r["probs"].argmax(-1).reshape(R, R) if kind == "ddpm" else r["nn_idx"].reshape(R, R)
        for kk, vv in r.items():
            stacks.setdefault(kk, []).append(vv.reshape((R, R) + vv.shape[1:]))
        centres.append((a0, b0, hw))
        print(f"zoom {tag} level {k} centre=({a0:.6f},{b0:.6f}) hw={hw:.3e} labels={np.unique(L)} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(f"{CACHE}/mnist_{tag}.npz", centres=np.array(centres), R=R, steps=steps, factor=factor,
                            **{kk: np.stack(vv) for kk, vv in stacks.items()})


def task_uncert(M=2048, steps=50, eps_list=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3)):
    """uncertainty exponent on the sphere: random slice-plane points, displaced by eps radians"""
    t0 = time.time()
    g = torch.Generator(device=DEV).manual_seed(0)
    A = (torch.rand(M, device=DEV, generator=g, dtype=torch.float64) * 2 - 1) * HW0
    B = (torch.rand(M, device=DEV, generator=g, dtype=torch.float64) * 2 - 1) * HW0
    th = torch.rand(M, device=DEV, generator=g, dtype=torch.float64) * 2 * math.pi
    e0, u, v = orthonormal_triple(D, SLICE_SEED, DEV)
    base = evaluate("ddpm", great_sphere(e0, u, v, A, B, RADIUS), steps, keep_images=False)["probs"].argmax(-1)
    out = {}
    for e in eps_list:
        lab = evaluate("ddpm", great_sphere(e0, u, v, A + e * th.cos(), B + e * th.sin(), RADIUS), steps,
                       keep_images=False)["probs"].argmax(-1)
        out[e] = float((lab != base).mean())
        print(f"uncert eps={e}: f={out[e]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    eps = np.array(list(out.keys())); f = np.array(list(out.values()))
    sel = f * M >= 10
    alpha = float(np.polyfit(np.log(eps[sel]), np.log(f[sel]), 1)[0]) if sel.sum() >= 3 else float("nan")
    json.dump(dict(eps=eps.tolist(), f=f.tolist(), alpha=alpha, D=2 - alpha, M=M, steps=steps, n_fit=int(sel.sum())),
              open(f"{CACHE}/mnist_uncert" + ("" if M == 2048 else f"_M{M}") + ".json", "w"), indent=1)
    print("alpha", alpha, flush=True)


def task_f64check():
    """deepest zoom window at 48^2: float32 vs float64 network labels"""
    z = np.load(f"{CACHE}/mnist_zoom.npz")
    a0, b0, hw = z["centres"][-1]
    Z = slice_points(a0, b0, hw, 48)
    r32 = evaluate("ddpm", Z, 50, torch.float32, keep_images=False)["probs"].argmax(-1)
    r64 = evaluate("ddpm", Z, 50, torch.float64, keep_images=False)["probs"].argmax(-1)
    res = dict(hw=float(hw), mismatch=float((r32 != r64).mean()), n=int(len(r32)))
    json.dump(res, open(f"{CACHE}/mnist_f64check.json", "w"))
    print(res)


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "map":
        kind, tag, R, steps = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        extra = [float(x) for x in sys.argv[6:9]] if len(sys.argv) > 6 else []
        task_map(kind, tag, R, steps, *extra)
    elif w == "stepsweep":
        task_stepsweep(int(sys.argv[2]))
    elif w == "zoom":
        task_zoom(int(sys.argv[2]), int(sys.argv[3]), kind=sys.argv[4] if len(sys.argv) > 4 else "ddpm",
                  tag=sys.argv[5] if len(sys.argv) > 5 else "zoom")
    elif w == "uncert":
        if len(sys.argv) > 2:
            task_uncert(int(sys.argv[2]), eps_list=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4))
        else:
            task_uncert()
    elif w == "f64check":
        task_f64check()
