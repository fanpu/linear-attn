"""Evaluate the loss surface of a trained net on a 2D grid (or a 1D line) spanned by two
filter-normalized random directions.  Resumable; writes cache/surf/<tag>.npz.

  python landscape.py --model resnet56_noshort --res 101 --xlim -1 1 --ylim -1 1 --tag hero
  python landscape.py --model resnet56_noshort --line --res 2001 --xlim -1 1 --tag slice
  python landscape.py --model resnet56_noshort --epoch 10 --res 41 --tag anim   # checkpoint

Coordinates (a, b) mean  w = w* + a*dx + b*dy  with dx, dy drawn from seeds 1 and 2 and
filter-normalized to the weights of the checkpoint being evaluated.
For --line, points run from (xlim0, ylim0) to (xlim1, ylim1) (ylim defaults to 0 0).
"""
import argparse, json, os, time
import numpy as np
import torch
from common import CACHE, load_model, load_cifar, fixed_subset, get_directions, LossEvaluator

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--epoch", type=int, default=None)
p.add_argument("--res", type=int, default=21)
p.add_argument("--resy", type=int, default=None)
p.add_argument("--xlim", type=float, nargs=2, default=[-1, 1])
p.add_argument("--ylim", type=float, nargs=2, default=None)
p.add_argument("--line", action="store_true")
p.add_argument("--n", type=int, default=5000, help="fixed training subset size")
p.add_argument("--seedx", type=int, default=1)
p.add_argument("--seedy", type=int, default=2)
p.add_argument("--float64", action="store_true")
p.add_argument("--batch", type=int, default=2500)
p.add_argument("--tag", required=True)
p.add_argument("--memfrac", type=float, default=0.10)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(args.memfrac)
dtype = torch.float64 if args.float64 else torch.float32
os.makedirs(os.path.join(CACHE, "surf"), exist_ok=True)
ep = "final" if args.epoch is None else f"ep{args.epoch:03d}"
out = os.path.join(CACHE, "surf", f"{args.model}_{ep}_{args.tag}.npz")

net = load_model(args.model, args.epoch)
dx, dy = get_directions(net, args.seedx, args.seedy)
xtr, ytr = load_cifar(True, dtype=dtype)
idx = torch.tensor(fixed_subset(args.n), device="cuda")
ev = LossEvaluator(net, xtr[idx], ytr[idx], dx, dy, batch=args.batch, dtype=dtype)
del xtr

if args.line:
    ylim = args.ylim or [0.0, 0.0]
    t = np.linspace(0, 1, args.res)
    pts = [(args.xlim[0] + (args.xlim[1] - args.xlim[0]) * s, ylim[0] + (ylim[1] - ylim[0]) * s) for s in t]
    shape = (args.res,)
    xs, ys = np.array([q[0] for q in pts]), np.array([q[1] for q in pts])
else:
    ylim = args.ylim or args.xlim
    ry = args.resy or args.res
    xs = np.linspace(*args.xlim, args.res); ys = np.linspace(*ylim, ry)
    pts = [(a, b) for b in ys for a in xs]  # row = y index, col = x index
    shape = (ry, args.res)

loss = np.full(len(pts), np.nan); acc = np.full(len(pts), np.nan)
if os.path.exists(out + ".partial.npz"):
    old = np.load(out + ".partial.npz")
    if old["loss"].size == loss.size:
        loss[:] = old["loss"].ravel(); acc[:] = old["acc"].ravel()
        print(f"resuming: {np.isfinite(loss).sum()} / {loss.size} done", flush=True)

meta = dict(vars(args), dtype=str(dtype), checkpoint=ep,
            note="w = w* + a*dx + b*dy; filter-normalized Gaussian dirs, biasbn zeroed, BN eval")
t0 = time.time(); last = t0; done0 = np.isfinite(loss).sum()


def dump(path):
    np.savez(path, loss=loss.reshape(shape), acc=acc.reshape(shape), xs=xs, ys=ys,
             meta=json.dumps(meta))


for k, (a, b) in enumerate(pts):
    if np.isfinite(loss[k]):
        continue
    loss[k], acc[k] = ev(a, b)
    if time.time() - last > 60:
        dump(out + ".partial.npz"); last = time.time()
        d = np.isfinite(loss).sum()
        rate = (d - done0) / (time.time() - t0)
        print(f"{d}/{loss.size}  {rate:.2f} pts/s  eta {(loss.size - d) / max(rate, 1e-9) / 60:.1f} min", flush=True)

meta["wall_s"] = time.time() - t0
meta["pts_per_s"] = (loss.size - done0) / max(meta["wall_s"], 1e-9)
dump(out)
if os.path.exists(out + ".partial.npz"):
    os.remove(out + ".partial.npz")
print(f"done {out}  wall {meta['wall_s']:.0f}s  center loss {ev(0, 0)[0]:.4f}  "
      f"min {np.nanmin(loss):.4f} max {np.nanmax(loss):.4f}", flush=True)
