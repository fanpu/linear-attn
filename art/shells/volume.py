"""3D loss volume on a 27^3 grid, K points per forward pass (vmap), one checkpoint per
z-slab (c index) so a rerun resumes.
  python volume.py --model resnet20 --dirs random --K 16          # a,b,c in grid_axis(): [-1.04, 1.04], h = 0.08
  python volume.py --model resnet20 --dirs pca --K 16             # axes span the checkpoint trajectory
writes cache/vol/<model>_<ckpt>_<dirs>_g27.npz  {loss (c,b,a), acc, a, b, c, meta}"""
import argparse, json, os, time, datetime
import numpy as np, torch
from lib import CACHE, grid_axis, load_net, random_dirs3, pca_dirs3, load_subset, VmapEvaluator, SeqEvaluator

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
p.add_argument("--extend", type=int, nargs=6, default=None, metavar=("ALO", "AHI", "BLO", "BHI", "CLO", "CHI"),
               help="append this many grid points (same spacing) below/above each axis; output tag _ext")
p.add_argument("--slab-shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"),
               help="only the slabs whose position in the centre-first order is K mod N (parallel CPU runs)")
p.add_argument("--assemble-only", action="store_true", help="never evaluate; assemble if every slab exists")
p.add_argument("--reuse", default=None, help="npz volume whose coincident grid points prefill this run")
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
if args.extend:
    e = args.extend
    axes = [np.concatenate([ax[0] - (ax[1] - ax[0]) * np.arange(lo, 0, -1), ax, ax[-1] + (ax[1] - ax[0]) * np.arange(1, hi + 1)])
            for ax, lo, hi in zip(axes, e[0::2], e[1::2])]
    name += "_ext"; out = os.path.join(CACHE, "vol", name + ".npz"); sdir = os.path.join(CACHE, "vol", name + ".slabs")
    os.makedirs(sdir, exist_ok=True)
A, B, C = axes
if os.path.exists(out):
    print(f"{out} already exists; nothing to do", flush=True)
    raise SystemExit(0)
pre = {}
if args.reuse:
    r = np.load(args.reuse)
    for kk, cc in enumerate(r["c"]):
        for jj, bb in enumerate(r["b"]):
            for ii, aa in enumerate(r["a"]):
                pre[(round(float(aa), 5), round(float(bb), 5), round(float(cc), 5))] = (r["loss"][kk, jj, ii], r["acc"][kk, jj, ii])
if args.assemble_only:
    x = y = None
else:
    x, y = load_subset(dev)
if args.assemble_only:
    ev = None; evaluator = "assembled from slabs"
elif args.K == 1:  # the sequential loss-landscape path (faster than vmap on the GB10, see NOTES.md)
    _seq = SeqEvaluator(net, x, y, dirs, batch=args.img_batch)
    def ev(chunk):
        l, a_ = _seq(*chunk[0])
        return np.array([l]), np.array([a_])
    evaluator = "sequential (SeqEvaluator, one point per pass)"
else:
    ev = VmapEvaluator(net, x, y, dirs, K=args.K, img_batch=args.img_batch)
    evaluator = f"vmap K={args.K}"
n = len(C); na, nb = len(A), len(B)
k0 = int(np.argmin(np.abs(C)))
order = sorted(range(n), key=lambda k: (abs(k - k0), k))  # centre slab first
meta = dict(vars(args), name=name, directions=dmeta, subset="loss-landscape fixed_subset(1000), seed 0",
            dtype="float32 weights/activations, TF32 off, CE summed in float64", bn="eval",
            torch=torch.__version__, gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", driver="580.173.02", cuda="13.0",
            date=datetime.date.today().isoformat(), coords="w = w* + a*d1 + b*d2 + c*d3; loss[c, b, a]", evaluator=evaluator)
print(json.dumps(meta), flush=True)
t_all = time.time(); done_pts = 0
for pos, k in enumerate(order):
    f = os.path.join(sdir, f"slab{k:02d}.npz")
    if os.path.exists(f) or pos % args.slab_shard[1] != args.slab_shard[0] or args.assemble_only:
        continue
    t0 = time.time()
    P = np.array([(a, b, C[k]) for b in B for a in A])  # row b, col a
    L = np.full(len(P), np.nan); Acc = np.full(len(P), np.nan)
    for q, (a, b, c) in enumerate(P):
        v = pre.get((round(float(a), 5), round(float(b), 5), round(float(c), 5)))
        if v is not None:
            L[q], Acc[q] = v
    todo = np.nonzero(~np.isfinite(L))[0]
    for i in range(0, len(todo), args.K):
        ids = todo[i:i + args.K]; chunk = P[ids]; m = len(chunk)
        if m < args.K:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], args.K - m, 0)])
        l, a_ = ev(chunk)
        L[ids], Acc[ids] = l[:m], a_[:m]
    np.savez(f + ".tmp.npz", loss=L.reshape(nb, na), acc=Acc.reshape(nb, na), wall_s=time.time() - t0, evaluator=evaluator,
             n_eval=len(todo))
    os.replace(f + ".tmp.npz", f)
    done_pts += len(todo)
    dt = time.time() - t0
    left = sum(not os.path.exists(os.path.join(sdir, f"slab{j:02d}.npz")) for j in range(n))
    print(f"[{datetime.datetime.now():%H:%M:%S}] slab {k:2d} c={C[k]:+.3f} {dt:.1f}s {len(todo)} new pts {len(todo)/dt:.2f} pts/s "
          f"loss [{L.min():.4f}, {L.max():.3f}]  left {left} slabs, eta {left*dt/60:.1f} min", flush=True)
    if k == k0 and args.dirs == "random":
        print(f"  centre loss {L.reshape(nb, na)[nb//2, na//2]:.6f}", flush=True)

missing = [k for k in range(n) if not os.path.exists(os.path.join(sdir, f"slab{k:02d}.npz"))]
if missing:
    print(f"shard {args.slab_shard} finished; slabs still missing {missing}; rerun without --slab-shard to assemble", flush=True)
    raise SystemExit(0)
loss = np.stack([np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["loss"] for k in range(n)])
acc = np.stack([np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["acc"] for k in range(n)])
walls = [float(np.load(os.path.join(sdir, f"slab{k:02d}.npz"))["wall_s"]) for k in range(n)]
meta["wall_s_sum_slabs"] = sum(walls)
n_eval = [int(np.load(os.path.join(sdir, f"slab{k:02d}.npz")).get("n_eval", na * nb)) for k in range(n)]
meta["points_evaluated"] = sum(n_eval)
meta["slab_evaluators"] = sorted(set(str(np.load(os.path.join(sdir, f"slab{k:02d}.npz")).get("evaluator", "vmap K=1")) for k in range(n)))
meta["pts_per_s"] = sum(n_eval) / sum(walls)
np.savez(out, loss=loss, acc=acc, a=A, b=B, c=C, meta=json.dumps(meta))
print(f"done {out}  pts/s {meta['pts_per_s']:.2f}  wall {sum(walls)/60:.1f} min  min {loss.min():.4f} max {loss.max():.2f}", flush=True)
