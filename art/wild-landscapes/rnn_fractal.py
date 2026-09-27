"""The loss of a trained recurrent net, unrolled over a long horizon, on a 2-D weight plane.

    python rnn_fractal.py train                      # CPU, ~1 min: trains the RNN, saves cache/rnn/model.pt
    python rnn_fractal.py plane --res 2048 --T 100 --span 1.5 --device cuda --out cache/rnn/plane_T100.npz
    python rnn_fractal.py plane ... --centre 0.9,0.3 --span 0.015      # a zoom window

Model (declared): h_t = tanh(W h_{t-1} + u x_t), y_hat_t = v . h_t, n = 32 hidden units, h_0 = 0.
Task: y_t = x_{t-3} (recall the input three steps back) on ONE fixed input sequence x ~ N(0,1)
(seed 0).  Loss L(W) = mean_{t<T} (y_hat_t - y_t)^2 over that fixed sequence: a deterministic
function of the weights, so every bump in the surface is real (no sampling anywhere).

Plane (Li et al. 2018 style): W(a,b) = W* + a D1 + b D2, D1, D2 Gaussian with every row (a
recurrent 'filter') rescaled to the norm of the same row of W*.  u and v stay at their trained
values.  Also measured per point: the largest Lyapunov exponent of the driven dynamics
lambda = (1/T) sum_t log ||J_t q_{t-1}||, J_t = diag(1 - h_t^2) W, q renormalised each step.

Evaluation uses the linearity of the pre-activation in (a, b):
    W(a,b) h = W* h + a (D1 h) + b (D2 h)
so P points share three (n x n) matrices.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache", "rnn")
N_H, DELAY, T_TRAIN, T_MAX = 32, 3, 200, 4000


def task(dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T_MAX, generator=g, dtype=dtype)
    y = torch.zeros_like(x); y[DELAY:] = x[:-DELAY]
    return x, y


def train(steps=1500):
    torch.manual_seed(0)
    x, y = task()
    n = N_H
    W = (0.9 * torch.randn(n, n, dtype=torch.float64) / math.sqrt(n)).requires_grad_()
    u = torch.randn(n, dtype=torch.float64)
    v = (torch.randn(n, dtype=torch.float64) / math.sqrt(n)).requires_grad_()
    opt = torch.optim.Adam([W, v], lr=3e-3)
    for s in range(steps):
        h = torch.zeros(n, dtype=torch.float64); L = 0
        for t in range(T_TRAIN):
            h = torch.tanh(W @ h + u * x[t]); L = L + (v @ h - y[t]) ** 2
        L = L / T_TRAIN
        opt.zero_grad(); L.backward(); opt.step()
        if s % 250 == 0:
            print(s, float(L), float(torch.linalg.eigvals(W).abs().max()), flush=True)
    g = torch.Generator().manual_seed(1)
    D = []
    for k in range(2):
        d = torch.randn(n, n, generator=g, dtype=torch.float64)
        d = d * (W.detach().norm(dim=1) / d.norm(dim=1))[:, None]
        D.append(d)
    os.makedirs(C, exist_ok=True)
    torch.save(dict(W=W.detach(), u=u, v=v.detach(), D1=D[0], D2=D[1], loss=float(L)), os.path.join(C, "model.pt"))
    print("final loss", float(L), "spectral radius", float(torch.linalg.eigvals(W.detach()).abs().max()))


@torch.no_grad()
def plane(a, b, T, dtype=torch.float32, device="cpu", lyap=True, chunk=1 << 20, prog=None):
    """a, b: 1-D arrays (P,).  Returns loss (P,), lyap (P,) as float64 numpy."""
    m = torch.load(os.path.join(C, "model.pt"))
    W, u, v, D1, D2 = (m[k].to(device, dtype) for k in ("W", "u", "v", "D1", "D2"))
    x, y = (z.to(device, dtype) for z in task(torch.float64))   # always the float64 draw
    P = len(a)
    Ls, Ly = np.zeros(P), np.zeros(P)
    for c0 in range(0, P, chunk):
        A = torch.as_tensor(a[c0:c0 + chunk], device=device, dtype=dtype)[:, None]
        B = torch.as_tensor(b[c0:c0 + chunk], device=device, dtype=dtype)[:, None]
        p = A.shape[0]
        h = torch.zeros(p, N_H, device=device, dtype=dtype)
        q = torch.ones(p, N_H, device=device, dtype=dtype) / math.sqrt(N_H)
        L = torch.zeros(p, device=device, dtype=torch.float64)
        lam = torch.zeros(p, device=device, dtype=torch.float64)
        for t in range(T):
            h = torch.tanh(h @ W.T + A * (h @ D1.T) + B * (h @ D2.T) + u * x[t])
            L += ((h @ v) - y[t]).double() ** 2
            if lyap:
                q = (1 - h * h) * (q @ W.T + A * (q @ D1.T) + B * (q @ D2.T))
                nq = q.norm(dim=1, keepdim=True).clamp_min(1e-300)
                lam += torch.log(nq[:, 0]).double(); q = q / nq
        Ls[c0:c0 + p] = (L / T).cpu().numpy(); Ly[c0:c0 + p] = (lam / T).cpu().numpy()
        if prog: prog(c0 + p, P)
    return Ls, Ly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--span", type=float, default=1.5)
    ap.add_argument("--centre", default="0,0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no_lyap", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        return train()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    prog = None
    if a.device == "cuda":
        import pasar_job; pasar_job.apply_memory_limit(); prog = lambda s, t: pasar_job.progress(s, t)
    ca, cb = (float(v) for v in a.centre.split(","))
    ax = np.linspace(-a.span, a.span, a.res)
    B_, A_ = np.meshgrid(cb + ax, ca + ax, indexing="ij")          # row = beta, col = alpha
    t0 = time.time()
    L, ly = plane(A_.ravel(), B_.ravel(), a.T, getattr(torch, a.dtype), a.device, not a.no_lyap, prog=prog)
    secs = time.time() - t0
    out = a.out or os.path.join(C, f"plane_T{a.T}_r{a.res}_s{a.span}.npz")
    np.savez_compressed(out, loss=L.reshape(a.res, a.res).astype(np.float32 if a.dtype == "float32" else np.float64),
                        lyap=ly.reshape(a.res, a.res).astype(np.float32), centre=(ca, cb), span=a.span, T=a.T,
                        dtype=a.dtype, secs=secs)
    print(json.dumps(dict(out=out, T=a.T, res=a.res, span=a.span, centre=(ca, cb), secs=round(secs, 1),
                          lmin=float(L.min()), lmax=float(L.max()), chaotic_frac=float((ly > 0).mean()))), flush=True)


if __name__ == "__main__":
    main()
