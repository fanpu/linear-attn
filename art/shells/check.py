"""Pipeline checks before the big volume.
 1. vmap-vs-sequential equivalence on N grid points of the 27^3 random-direction grid
    (acceptance: max relative loss difference <= 1e-5), plus timing of both paths;
 2. slice reproduction: points on the c = 0 plane against loss-landscape's
    resnet20_final_g51 values at coincident (a, b);
 3. (cuda) throughput vs K.
  python check.py --device cpu --n 12 --K 4
  python check.py --device cuda --n 50 --K 16 --Ks 1 4 8 16 32
writes cache/check_<device>.json"""
import argparse, json, os, time
import numpy as np, torch
from lib import CACHE, LL_ROOT, grid_axis, load_net, random_dirs3, load_subset, SeqEvaluator, VmapEvaluator

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--n", type=int, default=12)
p.add_argument("--nslice", type=int, default=6)
p.add_argument("--K", type=int, default=4)
p.add_argument("--Ks", type=int, nargs="*", default=[])
p.add_argument("--img-batch", type=int, default=1000)
p.add_argument("--memfrac", type=float, default=0.10)
args = p.parse_args()
dev = args.device
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(args.memfrac)
    torch.backends.cudnn.benchmark = True  # as in loss-landscape/landscape.py


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


net = load_net("resnet20", device=dev)
dirs = random_dirs3(net, device=dev)
x, y = load_subset(dev)
ax = grid_axis()
rng = np.random.default_rng(0)
pts = np.stack([rng.choice(ax, args.n) for _ in range(3)], 1)
pts[0] = 0.0

# slice points: c = 0 plane, (a, b) on the interior grid (= odd g51 indices), include the centre
g = np.load(os.path.join(LL_ROOT, "cache", "surf", "resnet20_final_g51.npz"))
gx = np.round(g["xs"], 6)
sl = np.stack([rng.choice(ax[1:-1], args.nslice), rng.choice(ax[1:-1], args.nslice), np.zeros(args.nslice)], 1)
sl[0, :2] = 0.0
ref = np.array([g["loss"][np.where(gx == round(b, 6))[0][0], np.where(gx == round(a, 6))[0][0]] for a, b, _ in sl])

seq = SeqEvaluator(net, x, y, dirs, batch=args.img_batch)
seq(0.3, 0.3, 0.3); sync()  # warm-up
t0 = time.time()
ls = np.array([seq(*q)[0] for q in pts]); sync()
t_seq = time.time() - t0
sls = np.array([seq(*q)[0] for q in sl])
net = load_net("resnet20", device=dev)  # fresh copy of w*
vm = VmapEvaluator(net, x, y, dirs, K=args.K, img_batch=args.img_batch)


def run_vmap(vm, P):
    K = vm.K; out = []
    for i in range(0, len(P), K):
        chunk = P[i:i + K]; m = len(chunk)
        if m < K:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], K - m, 0)])
        out.append(vm(chunk)[0][:m])
    return np.concatenate(out)


run_vmap(vm, pts[:1]); sync()  # warm-up
t0 = time.time()
lv = run_vmap(vm, pts); sync()
t_vm = time.time() - t0
slv = run_vmap(vm, sl)

rel = np.abs(lv - ls) / np.abs(ls)
res = dict(device=dev, n=args.n, K=args.K, img_batch=args.img_batch,
           equiv_max_rel=float(rel.max()), equiv_median_rel=float(np.median(rel)),
           equiv_pass=bool(rel.max() <= 1e-5),
           seq_pts_per_s=args.n / t_seq, vmap_pts_per_s=args.n / t_vm, speedup=t_seq / t_vm,
           slice_pts=sl.tolist(), slice_ref_g51=ref.tolist(), slice_seq=sls.tolist(), slice_vmap=slv.tolist(),
           slice_seq_max_rel=float((np.abs(sls - ref) / ref).max()),
           slice_vmap_max_rel=float((np.abs(slv - ref) / ref).max()),
           slice_vmap_median_rel=float(np.median(np.abs(slv - ref) / ref)),
           loss_range=[float(ls.min()), float(ls.max())])
print(json.dumps({k: v for k, v in res.items() if not k.startswith("slice_") or "rel" in k}, indent=1), flush=True)

if args.Ks:
    thr = {}
    for K in args.Ks:
        if dev == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ib = min(args.img_batch, max(125, 8000 // K))  # keep K * img_batch <= 8000 (fixed shapes)
        try:
            vmK = VmapEvaluator(net, x, y, dirs, K=K, img_batch=ib)
            P = np.concatenate([pts] * int(np.ceil(4 * K / len(pts))))[:4 * K]
            run_vmap(vmK, P[:K]); sync()
            t0 = time.time(); run_vmap(vmK, P); sync(); dt = time.time() - t0
            lk = run_vmap(vmK, pts)
            rk = float((np.abs(lk - ls) / np.abs(ls)).max())
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
            thr[K] = dict(pts_per_s=4 * K / dt, img_batch=ib, equiv_max_rel=rk, peak_gib=mem, speedup_vs_seq=(4 * K / dt) * t_seq / args.n)
            print(f"K={K} img_batch={ib}: {thr[K]['pts_per_s']:.2f} pts/s (x{thr[K]['speedup_vs_seq']:.2f} vs seq)  "
                  f"equiv max rel {rk:.2e}  peak alloc {mem:.2f} GiB", flush=True)
            del vmK
        except torch.OutOfMemoryError as e:
            print(f"K={K}: OOM ({str(e)[:80]})", flush=True)
            thr[K] = dict(oom=True)
    res["throughput_vs_K"] = thr
json.dump(res, open(os.path.join(CACHE, f"check_{dev}.json"), "w"), indent=1)
