"""Quick check of the Schoenholz et al. (2017) trainability prediction (max trainable depth
~ 6 xi_c) for tanh MLPs on MNIST at sigma_b^2 = 0.05.

Model: 784 -> [width tanh]^L -> 10, weights N(0, sigma_w^2 / fan_in), biases N(0, sigma_b^2).
Optimiser: SGD, momentum 0.9, lr 1e-3, batch 128, --steps steps (full-precision float32).
Metric: training accuracy on a fixed 10k training subset after training.

  python compute_trainability.py --steps 1500
"""
import argparse, math, os, time
import numpy as np
import torch
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--width", type=int, default=300)
ap.add_argument("--sw2", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.76, 2.0, 2.5, 3.0, 4.0])
ap.add_argument("--depths", type=int, nargs="+", default=[10, 20, 40, 80, 120, 160, 240])
ap.add_argument("--sb2", type=float, default=0.05)
ap.add_argument("--out", default="cache/trainability.npz")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.10)
dev = "cuda"
here = os.path.dirname(os.path.abspath(__file__))
ds = torchvision.datasets.MNIST(root="/home/fzeng/ml/research/art/data", train=True, download=False)
X = (ds.data.float().reshape(-1, 784) / 255.0)
X = ((X - X.mean()) / X.std()).to(dev)
Y = ds.targets.to(dev)
g = torch.Generator(device="cpu").manual_seed(0)
sub = torch.randperm(len(X), generator=g)[:10000].to(dev)


def run(sw2, L, seed=0):
    torch.manual_seed(seed)
    dims = [784] + [args.width] * L + [10]
    Ws, bs = [], []
    for i in range(len(dims) - 1):
        Ws.append((torch.randn(dims[i + 1], dims[i], device=dev) * math.sqrt(sw2 / dims[i])).requires_grad_())
        bs.append((torch.randn(dims[i + 1], device=dev) * math.sqrt(args.sb2)).requires_grad_())
    params = Ws + bs
    opt = torch.optim.SGD(params, lr=1e-3, momentum=0.9)

    def fwd(x):
        h = x
        for i in range(len(Ws) - 1):
            h = torch.tanh(h @ Ws[i].T + bs[i])
        return h @ Ws[-1].T + bs[-1]

    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    for s in range(args.steps):
        idx = torch.randint(0, len(X), (128,), device=dev, generator=gen)
        loss = torch.nn.functional.cross_entropy(fwd(X[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = sum((fwd(X[sub[i:i + 2000]]).argmax(1) == Y[sub[i:i + 2000]]).sum().item() for i in range(0, 10000, 2000)) / 10000
    return acc, float(loss.item()) if math.isfinite(loss.item()) else float("nan")


acc = np.zeros((len(args.depths), len(args.sw2)))
t0 = time.time()
for j, sw2 in enumerate(args.sw2):
    for i, L in enumerate(args.depths):
        acc[i, j], lo = run(sw2, L)
        print(f"sw2={sw2} L={L} acc={acc[i, j]:.3f} loss={lo:.3f} {time.time() - t0:.0f}s", flush=True)
        np.savez(os.path.join(here, args.out), acc=acc, sw2=np.array(args.sw2), depths=np.array(args.depths),
                 sb2=args.sb2, steps=args.steps, width=args.width, wall=time.time() - t0)
print("done", time.time() - t0)
