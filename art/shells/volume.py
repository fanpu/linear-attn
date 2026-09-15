"""3D loss volume on a 27^3 grid, K points per forward pass (vmap), one checkpoint per
z-slab (c index) so a rerun resumes.
  python volume.py --model resnet20 --dirs random --K 16          # a,b,c in grid_axis(): [-1.04, 1.04], h = 0.08
  python volume.py --model resnet20 --dirs pca --K 16             # axes span the checkpoint trajectory
writes cache/vol/<model>_<ckpt>_<dirs>_g27.npz  {loss (c,b,a), acc, a, b, c, meta}"""
import argparse, json, os, time, datetime
import numpy as np, torch
from lib import CACHE, grid_axis, load_net, random_dirs3, pca_dirs3, load_subset, VmapEvaluator

p = argparse.ArgumentParser()
p.add_argument("--model", default="resnet20")
p.add_argument("--epoch", type=int, default=None)
p.add_argument("--dirs", choices=["random", "pca"], default="random")
p.add_argument("--res", type=int, default=27)
p.add_argument("--K", type=int, default=16)
p.add_argument("--img-batch", type=int, default=1000)
p.add_argument("--device", default="cuda")
p.add_argument("--h", type=float, default=0.08, help="random: grid spacing")
p.add_argument("--tag", default="")
p.add_argument("--memfrac", type=float, default=0.10)
p.add_argument("--margin", type=float, default=0.08, help="pca: fractional margin around the trajectory")
args = p.parse_args()

dev = args.device
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(args.memfrac)
    torch.backends.cudnn.benchmark = True
ck = "final" if args.epoch is None else f"ep{args.epoch:03d}"
name = f"{args.model}_{ck}_{args.dirs}_g{args.res}{args.tag}"
out = os.path.join(CACHE, "vol", name + ".npz")
sdir = os.path.join(CACHE, "vol", name + ".slabs")
os.makedirs(sdir, exist_ok=True)

net = load_net(args.model, args.epoch, device=dev)
if args.dirs == "random":
    dirs = random_dirs3(net, device=dev)
    axes = [grid_axis(args.res, args.h)] * 3
    dmeta = "filter-normalised Gaussian directions, seeds 1,2,3, biasbn zeroed, normalised to this checkpoint"
else:
    dirs, info = pca_dirs3(args.model, device=dev)
    co = info["coords"]; axes = []
    for i in range(3):
        lo, hi = float(min(co[:, i].min(), 0)), float(max(co[:, i].max(), 0))
        m = args.margin * (hi - lo)
        h = (hi - lo + 2 * m) / (args.res - 2)
        k0 = int(np.ceil((m - lo) / h))
        axes.append((np.arange(args.res) - k0) * h)
        assert axes[-1][0] <= lo - m + 1e-9 and axes[-1][-1] >= hi + m - 1e-9
    dmeta = (f"top-3 PCA directions of {args.model} checkpoints ep000..ep040 minus final, unit norm, biasbn "
             f"zeroed; explained {np.round(info['explained'][:3], 4).tolist()}")
A, B, C = axes
x, y = load_subset(dev)
ev = VmapEvaluator(net, x, y, dirs, K=args.K, img_batch=args.img_batch)
n = args.res
order = sorted(range(n), key=lambda k: (abs(k - n // 2), k))  # centre slab first
meta = dict(vars(args), name=name, directions=dmeta, subset="loss-landscape fixed_subset(1000), seed 0",
            dtype="float32 weights/activations, TF32 off, CE summed in float64", bn="eval",
            torch=torch.__version__, gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", driver="580.173.02", cuda="13.0",
            date=datetime.date.today().isoformat(), coords="w = w* + a*d1 + b*d2 + c*d3; loss[c, b, a]")
print(json.dumps(meta), flush=True)
t_all = time.time(); done_pts = 0
for k in order:
    f = os.path.join(sdir, f"slab{k:02d}.npz")
    if os.path.exists(f):
        continue
    t0 = time.time()
    P = np.array([(a, b, C[k]) for b in B for a in A])  # row b, col a
    L = np.empty(len(P)); Acc = np.empty(len(P))
    for i in range(0, len(P), args.K):
        chunk = P[i:i + args.K]; m = len(chunk)
        if m < args.K:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], args.K - m, 0)])
        l, a_ = ev(chunk)
        L[i:i + m], Acc[i:i + m] = l[:m], a_[:m]
    np.savez(f + ".tmp.npz", loss=L.reshape(n, n), acc=Acc.reshape(n, n), wall_s=time.time() - t0)
    os.replace(f + ".tmp.npz", f)
    done_pts += len(P)
    dt = time.time() - t0
    left = sum(not os.path.exists(os.path.join(sdir, f"slab{j:02d}.npz")) for j in range(n))
    print(f"[{datetime.datetime.now():%H:%M:%S}] slab {k:2d} c={C[k]:+.3f} {dt:.1f}s {len(P)/dt:.2f} pts/s "
          f"loss [{L.min():.4f}, {L.max():.3f}]  left {left} slabs, eta {left*dt/60:.1f} min", flush=True)
    if k == n // 2 and args.dirs == "random":
        print(f"  centre loss {L.reshape(n, n)[n//2, n//2]:.6f}", flush=True)

loss = np.stack([np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["loss"] for k in range(n)])
acc = np.stack([np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["acc"] for k in range(n)])
walls = [float(np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["wall_s"]) for k in range(n)]
meta["wall_s_sum_slabs"] = sum(walls)
meta["pts_per_s"] = n ** 3 / sum(walls)
np.savez(out, loss=loss, acc=acc, a=A, b=B, c=C, meta=json.dumps(meta))
print(f"done {out}  pts/s {meta['pts_per_s']:.2f}  wall {sum(walls)/60:.1f} min  min {loss.min():.4f} max {loss.max():.2f}", flush=True)
