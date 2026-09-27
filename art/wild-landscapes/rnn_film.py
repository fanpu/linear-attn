"""Zoom film into the unrolled-RNN loss (T = 30), 10^5x, every frame a fresh float64 measurement.

    python rnn_film.py compute --frames 384 --res 513          # CPU, ~6 s/frame -> cache/rnn/zoom/
    python rnn_film.py render  --device cuda                   # GPU -> cache/film/zoom/f%04d.png

Window k: half-width s_k = s0 * (s1/s0)^(k/(N-1)) (exponential zoom, s0 = 2.5, s1 = 2.5e-5), held for
the last 24 frames.  Centre: c_k = p + (c0 - p) (s_k/s0)^1.5 with c0 = (0, 0) (the trained net) and
p = (-1.2402, -1.2744), a strongly chaotic point (lambda = +0.25 at T = 30), so the view drifts
from the whole plane onto p.
Declared: height = K_k (max(log10 L, -1.5) - median), K_k = min(0.25 (s0/s_k)^0.35, 0.55/range_k),
range_k = the window's 2-98% log-loss range (the cap keeps the relief below the camera): relief grows with
zoom at the rate the loss range shrinks in the chaotic band (measured: window range ~ s^0.3-0.5 for
s > 1e-4); below that the true surface smooths faster than s^0.35, so the last frames visibly
calm down.  Colour: Lyapunov exponent, as the stills.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rnn_fractal as RF

S0, S1, P = 2.5, 2.5e-5, np.array([-1.2402, -1.2744])
ZD = os.path.join(HERE, "cache", "rnn", "zoom")


def window(k, N, hold=24):
    m = N - hold
    u = min(k, m - 1) / (m - 1)
    s = S0 * (S1 / S0) ** u
    c = P + (np.zeros(2) - P) * (s / S0) ** 1.5
    return s, c


def compute(a):
    os.makedirs(ZD, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    m = a.frames - 24
    for k in range(m):
        f = os.path.join(ZD, f"z{k:04d}.npz")
        if os.path.exists(f):
            continue
        s, c = window(k, a.frames)
        ax = np.linspace(-s, s, a.res)
        B_, A_ = np.meshgrid(c[1] + ax, c[0] + ax, indexing="ij")
        L, ly = RF.plane(A_.ravel(), B_.ravel(), 30, torch.float64)
        np.savez(f, loss=L.reshape(a.res, a.res).astype(np.float32), lyap=ly.reshape(a.res, a.res).astype(np.float32),
                 logloss=np.log10(L).reshape(a.res, a.res), s=s, c=c)
        print(k, s, flush=True)


def render(a):
    import pasar_job
    import wl_render as R
    pasar_job.apply_memory_limit()
    out = os.path.join(HERE, "cache", "film", "zoom"); os.makedirs(out, exist_ok=True)
    m = a.frames - 24
    for k in range(a.frames):
        fo = os.path.join(out, f"f{k:04d}.png")
        if os.path.exists(fo):
            continue
        kk = min(k, m - 1)
        d = np.load(os.path.join(ZD, f"z{kk:04d}.npz"))
        s = float(d["s"])
        F = np.maximum(d["logloss"], -1.5)
        t = np.clip(0.5 + d["lyap"] / 0.8, 0, 1)
        col = R.cmap(t, "crameri_berlin")
        rng = float(np.percentile(F, 98) - np.percentile(F, 2))
        K = min(0.25 * (S0 / s) ** 0.35, 0.55 / max(rng, 1e-9))
        Z0 = float(np.percentile(F, 50))
        S = R.Scene(F, col, K=K, Z0=Z0, up=1, device=a.device)
        img, _ = S.render(a.orbit, width=a.width, ss=a.ss, key=(200, 30), exposure=1.4, fog=0.2)
        R.save(img, fo)
        pasar_job.progress(k + 1, a.frames)
        print(k, round(s, 8), round(K, 3), flush=True)


HD = os.path.join(HERE, "cache", "rnn", "horizon")


@torch.no_grad()
def horizon_compute(a):
    """one float64 pass over the span-2.5 plane, recording the running loss L_T = mean_{t<T} and the
    running Lyapunov estimate at every horizon T = 1..Tmax."""
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    os.makedirs(HD, exist_ok=True)
    m = torch.load(os.path.join(RF.C, "model.pt"))
    W, u, v, D1, D2 = (m[k].double() for k in ("W", "u", "v", "D1", "D2"))
    x, y = RF.task(torch.float64)
    ax = torch.linspace(-2.5, 2.5, a.res, dtype=torch.float64)
    B_, A_ = torch.meshgrid(ax, ax, indexing="ij")
    A = A_.reshape(-1, 1); B = B_.reshape(-1, 1)
    h = torch.zeros(A.shape[0], RF.N_H, dtype=torch.float64); q = torch.ones_like(h) / RF.N_H ** 0.5
    S = torch.zeros(A.shape[0], dtype=torch.float64); lam = torch.zeros_like(S)
    for t in range(a.Tmax):
        h = torch.tanh(h @ W.T + A * (h @ D1.T) + B * (h @ D2.T) + u * x[t])
        S += ((h @ v) - y[t]) ** 2
        q = (1 - h * h) * (q @ W.T + A * (q @ D1.T) + B * (q @ D2.T))
        nq = q.norm(dim=1, keepdim=True); lam += torch.log(nq[:, 0]); q = q / nq
        T = t + 1
        np.savez(os.path.join(HD, f"h{T:04d}.npz"), logloss=torch.log10(S / T).view(a.res, a.res).numpy().astype(np.float32),
                 lyap=(lam / T).view(a.res, a.res).numpy().astype(np.float32), T=T)
        print(T, flush=True)


def horizon_render(a):
    """fixed camera and light; frame k shows horizon T = 1 + k // hold (each measurement held `hold`
    video frames, no interpolation).  Same height/colour mapping as the stills (floor -1.5, K 0.25)."""
    import pasar_job
    import wl_render as R
    pasar_job.apply_memory_limit()
    out = os.path.join(HERE, "cache", "film", "horizon"); os.makedirs(out, exist_ok=True)
    N = a.Tmax * a.hold + 36
    for k in range(N):
        fo = os.path.join(out, f"f{k:04d}.png")
        if os.path.exists(fo):
            continue
        T = min(1 + k // a.hold, a.Tmax)
        d = np.load(os.path.join(HD, f"h{T:04d}.npz"))
        F = np.maximum(d["logloss"].astype(np.float64), -1.5)
        col = R.cmap(np.clip(0.5 + d["lyap"] / 0.8, 0, 1), "crameri_berlin")
        S = R.Scene(F, col, K=0.25, Z0=-1.5, up=1, device=a.device)
        img, _ = S.render(a.orbit, width=a.width, ss=a.ss, key=(200, 30), exposure=1.4, fog=0.2)
        R.save(img, fo)
        pasar_job.progress(k + 1, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--frames", type=int, default=384)
    ap.add_argument("--res", type=int, default=513)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--orbit", default="0,0.1,0,1.15,-90,64,42")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--Tmax", type=int, default=100)
    ap.add_argument("--hold", type=int, default=3)
    a = ap.parse_args()
    dict(compute=compute, render=render, hcompute=horizon_compute, hrender=horizon_render)[a.cmd](a)


if __name__ == "__main__":
    main()
