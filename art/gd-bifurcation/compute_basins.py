"""2D initialisation basin maps (torch float64, GPU if available).

Systems
  zhu4   : GD on 1/2(1 - xyzw)^2 with the symmetric init z=x, w=y (Zhu et al. 2023, Sec. 5, eta=0.2).
           Exact reduced map  x <- x - eta x y^2 (x^2y^2 - 1),  y <- y - eta x^2 y (x^2y^2 - 1).
  prod2  : GD on 1/2(xy - 1)^2 (null model: Liang & Montufar Thm 1 prove this convergence
           region equals a smooth ellipse up to measure zero).
  lmreg  : GD on 1/2(uv - 0.5)^2 + 0.1(u^2+v^2), eta=1 (Liang & Montufar Fig. 3: proven
           self-similar boundary, their box-counting estimate 1.249) -- positive control.
  quad   : GD on 1/4(x^2+y^2-1)^2... not used; see README.

Per pixel: status (0 = still bounded at T, 1 = converged, 2 = diverged), event step,
final coordinates (which minimum), final sharpness (converged pixels).

Usage: python compute_basins.py zhu4 --x0 0 --x1 5 --y0 0 --y1 4 --res 4096 --T 20000 --tag wide
"""
import argparse
import time
import numpy as np
import torch

CACHE = "/home/fzeng/ml/research/art/gd-bifurcation/cache"
DT = torch.float64


def make_step(name, eta):
    if name == "zhu4":
        def step(x, y):
            r = eta * (x * x * y * y - 1.0)
            return x - r * x * y * y, y - r * x * x * y

        def loss(x, y):
            return 0.5 * (1 - x * x * y * y) ** 2

        def sharp(x, y):  # top Hessian eigenvalue at a minimum of the 4-variable objective: 2/x^2 + 2/y^2
            return 2 / (x * x) + 2 / (y * y)
        lmin = 0.0
    elif name == "prod2":
        def step(x, y):
            r = eta * (x * y - 1.0)
            return x - r * y, y - r * x

        def loss(x, y):
            return 0.5 * (x * y - 1) ** 2

        def sharp(x, y):
            return x * x + y * y
        lmin = 0.0
    elif name == "lmreg":
        lam, yt = 0.2, 0.5

        def step(u, v):
            r = u * v - yt
            return u - eta * (r * v + lam * u), v - eta * (r * u + lam * v)

        def loss(u, v):
            return 0.5 * (u * v - yt) ** 2 + 0.5 * lam * (u * u + v * v)

        def sharp(u, v):
            return u * u + v * v + lam
        # minimisers u = v = +-sqrt(y - lam)
        m2 = yt - lam
        lmin = 0.5 * (m2 - yt) ** 2 + lam * m2
    else:
        raise ValueError(name)
    return step, loss, sharp, lmin


def run(name, eta, xs, ys, T, tol, check=50, dev="cuda", chunk=1 << 22, escape=1e3):
    """Per-step event tracking on device (no host sync inside the `check` block):
       t_conv  first step with loss - lmin < tol
       t_esc   first step with max(|x|,|y|) > escape, and a smooth escape value
               nu = t_esc - log(log r / log escape) / log(7)   (the reduced map has polynomial degree 7)."""
    step, loss, sharp, lmin = make_step(name, eta)
    X, Y = np.meshgrid(xs, ys)
    X = X.ravel()
    Y = Y.ravel()
    Npix = X.size
    status = np.zeros(Npix, np.uint8)
    tev = np.full(Npix, T, np.int32)
    nu = np.full(Npix, np.nan, np.float32)
    fx = np.full(Npix, np.nan, np.float32)
    fy = np.full(Npix, np.nan, np.float32)
    fs = np.full(Npix, np.nan, np.float32)
    ln_esc = float(np.log(np.log(escape)))
    for c0 in range(0, Npix, chunk):
        idx = torch.arange(c0, min(Npix, c0 + chunk), device=dev)
        x = torch.tensor(X[c0:c0 + chunk], dtype=DT, device=dev)
        y = torch.tensor(Y[c0:c0 + chunk], dtype=DT, device=dev)
        te = torch.full_like(x, -1.0)
        nue = torch.full_like(x, float("nan"))
        tc = torch.full_like(x, -1.0)
        t = 0
        while t < T and idx.numel() > 0:
            for _ in range(check):
                x, y = step(x, y)
                t += 1
                r = torch.maximum(x.abs(), y.abs())
                new_e = (te < 0) & (r > escape)
                te = torch.where(new_e, torch.full_like(te, t), te)
                nue = torch.where(new_e, t - (torch.log(torch.log(r.clamp_min(escape))) - ln_esc) / np.log(7.0), nue)
                new_c = (tc < 0) & ((loss(x, y) - lmin) < tol)
                tc = torch.where(new_c, torch.full_like(tc, t), tc)
                # freeze escaped points so they do not overflow
                x = torch.where(te >= 0, torch.zeros_like(x), x)
                y = torch.where(te >= 0, torch.zeros_like(y), y)
            div = te >= 0
            conv = (tc >= 0) & ~div
            done = div | conv
            if done.any():
                di = idx[done].cpu().numpy()
                status[di] = np.where(div[done].cpu().numpy(), 2, 1)
                tev[di] = torch.where(div, te, tc)[done].cpu().numpy().astype(np.int32)
                nu[di] = nue[done].float().cpu().numpy()
                cc = conv
                ci = idx[cc].cpu().numpy()
                fx[ci] = x[cc].float().cpu().numpy()
                fy[ci] = y[cc].float().cpu().numpy()
                fs[ci] = sharp(x[cc], y[cc]).float().cpu().numpy()
                keep = ~done
                idx, x, y, te, nue, tc = idx[keep], x[keep], y[keep], te[keep], nue[keep], tc[keep]
        if idx.numel():
            ri = idx.cpu().numpy()
            fx[ri] = x.float().cpu().numpy()
            fy[ri] = y.float().cpu().numpy()
    shp = (len(ys), len(xs))
    return dict(status=status.reshape(shp), tev=tev.reshape(shp), nu=nu.reshape(shp), fx=fx.reshape(shp),
                fy=fy.reshape(shp), fs=fs.reshape(shp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--x0", type=float, required=True)
    ap.add_argument("--x1", type=float, required=True)
    ap.add_argument("--y0", type=float, required=True)
    ap.add_argument("--y1", type=float, required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    eta = a.eta if a.eta is not None else {"zhu4": 0.2, "prod2": 0.2, "lmreg": 1.0}[a.name]
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.10)
    t0 = time.time()
    # pixel centres
    h = (a.x1 - a.x0) / a.res
    xs = a.x0 + h * (np.arange(a.res) + 0.5)
    hy = (a.y1 - a.y0) / a.res
    ys = a.y0 + hy * (np.arange(a.res) + 0.5)
    o = run(a.name, eta, xs, ys, a.T, a.tol, dev=dev)
    fn = f"{CACHE}/basin_{a.name}_{a.tag}_{a.res}.npz"
    np.savez_compressed(fn, xs=xs, ys=ys, eta=eta, T=a.T, tol=a.tol, **o)
    s = o["status"]
    print(f"{fn}: conv {np.mean(s == 1):.3f} div {np.mean(s == 2):.3f} bounded {np.mean(s == 0):.3f} "
          f"in {time.time() - t0:.0f}s on {dev}")


if __name__ == "__main__":
    main()
