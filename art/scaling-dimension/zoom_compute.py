"""Zoom film data: the d=2 teacher's graph (z1, z2, T(z)) and several trained students' graphs,
sampled on a view window that shrinks isotropically by 10^-decades around z0.

Each frame uses the SAME K random offsets u in [-1,1]^2 (so points move continuously, no flicker):
    z = z0 + rho(t) * u,   h_s = (f_s(z) - T(z0)) / rho(t)
i.e. heights are magnified by the same factor as the plane (isotropic zoom).
Saves float16 heights (frames x surfaces x K) + metadata to cache/zoom/zoom_<fam>.npz.

    python zoom_compute.py --npz cache/main/ts_main.npz --fam relub --widths 6,16,45,90
"""
import argparse, json, math
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--npz", default="cache/main/ts_main.npz")
p.add_argument("--fam", default="relub")
p.add_argument("--seed", default="0")
p.add_argument("--widths", default="6,16,45,90")
p.add_argument("--frames", type=int, default=900)
p.add_argument("--K", type=int, default=80000)
p.add_argument("--decades", type=float, default=3.0)
p.add_argument("--rho0", type=float, default=0.42)
p.add_argument("--z0", default="0.137,-0.083")
p.add_argument("--out", default="cache/zoom/zoom.npz")
a = p.parse_args()
torch.cuda.set_per_process_memory_fraction(0.04)
dev = "cuda"
z = np.load(a.npz)
cfg = z["configs"]
d = 2
Q = torch.as_tensor(z[f"Q{d}"], device=dev).float()
mu, sd = json.loads(str(z["norms"]))[f"{a.fam}_{d}"]
TW = [(torch.as_tensor(z[f"teacher_{a.fam}_W{i}"], device=dev), torch.as_tensor(z[f"teacher_{a.fam}_b{i}"], device=dev)) for i in range(3)]
j = int(np.where((cfg[:, 0] == a.fam) & (cfg[:, 1] == str(d)) & (cfg[:, 2] == a.seed))[0][0])
widths = [int(w) for w in a.widths.split(",") if f"w{w}_W0" in z.files]
SW = {w: [(torch.as_tensor(z[f"w{w}_W{i}"][j], device=dev), torch.as_tensor(z[f"w{w}_b{i}"][j], device=dev)) for i in range(3)] for w in widths}


def mlp(params, x):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < len(params) - 1:
            h = torch.relu(h)
    return h[:, 0]


def teacher(x):
    return (mlp(TW, x) - mu) / sd


z0 = torch.tensor([float(v) for v in a.z0.split(",")], device=dev)
g = torch.Generator(device=dev).manual_seed(0)
u = torch.rand(a.K, 2, device=dev, generator=g) * 2 - 1
t = np.linspace(0, 1, a.frames)
ease = 0.5 - 0.5 * np.cos(np.pi * t)          # declared: cosine ease in/out of the zoom
rhos = a.rho0 * 10 ** (-a.decades * ease)
H = np.empty((a.frames, 1 + len(widths), a.K), dtype=np.float16)
with torch.no_grad():
    # teacher value and gradient at z0 (float64 central differences); heights are shown relative to
    # the teacher's tangent plane at z0 (declared), so the teacher tends to a flat plane under zoom
    TWd = [(W.double(), b.double()) for W, b in TW]
    f64 = lambda zz: (mlp(TWd, zz @ Q.double().T) - mu) / sd
    z0d = z0.double(); e = 1e-7
    grad = torch.stack([(f64((z0d + e * v)[None]) - f64((z0d - e * v)[None]))[0] / (2 * e) for v in torch.eye(2, device=dev, dtype=torch.float64)]).float()
    T0 = f64(z0d[None])[0].float()
    print("grad T(z0) =", grad.cpu().numpy())
    for fi, rho in enumerate(rhos):
        x = (z0 + rho * u) @ Q.T
        plane = T0 + rho * (u @ grad)
        H[fi, 0] = ((teacher(x) - plane) / rho).cpu().numpy()
        for si, w in enumerate(widths):
            H[fi, 1 + si] = ((mlp(SW[w], x) - plane) / rho).cpu().numpy()
        if fi % 100 == 0:
            print(fi, rho, flush=True)
Ns = [int(z[f"w{w}_N"]) for w in widths]
tests = [float(z[f"w{w}_test"][j]) for w in widths]
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, H=H, grad=grad.cpu().numpy(), u=u.cpu().numpy(), rhos=rhos, widths=widths, Ns=Ns, tests=tests, z0=z0.cpu().numpy(), fam=a.fam)
print("saved", a.out, H.shape, "student test losses", tests)
