"""Measure one quantity on the res x res filter-normalised plane (alpha, beta in [-1, 1]).

kinds (all on the SAME plane: directions from seeds 1, 2):
  A    fixed eval set of --n train images, BN eval mode        -> the smooth "true" surface
       (+ 256 test-image probes: agreement with centre argmax, PCA3 of predicted probs)
  A16  the same --n images split into n/16 fixed minibatches of 16, BN train mode per
       minibatch, averaged                                     -> E[B], the mean of the Ideami measurement
  B    a fresh random minibatch of 16 train images per point, BN train mode  -> Ideami-style
  C    one fixed minibatch of 16 (the first of A16), BN train mode
  D    a fresh random minibatch of 16 per point, BN eval mode

usage: python measure.py --kind A --res 1024 --n 2048 [--seed 1] [--row0 0 --row1 1024] [--tf32]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job
from dl_common import BatchedNet, load_cifar, load_trained, random_direction, fixed_indices, grid_axis, CACHE

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True, choices=["A", "A16", "B", "C", "D"])
p.add_argument("--res", type=int, default=1024)
p.add_argument("--n", type=int, default=2048)
p.add_argument("--seed", type=int, default=1, help="minibatch sampling seed for B/D")
p.add_argument("--P", type=int, default=0, help="nets per forward (0 = auto)")
p.add_argument("--chunk", type=int, default=512, help="images per forward for A/A16")
p.add_argument("--row0", type=int, default=0)
p.add_argument("--row1", type=int, default=-1)
p.add_argument("--tf32", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--bench", action="store_true", help="time a few chunks and exit")
args = p.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = args.tf32
torch.backends.cuda.matmul.allow_tf32 = args.tf32
pasar_job.apply_memory_limit()
dev = "cuda"
res = args.res
row1 = res if args.row1 < 0 else args.row1
name = f"{args.kind}" + (f"_s{args.seed}" if args.kind in ("B", "D") else "") + \
       (f"_n{args.n}" if args.kind in ("A", "A16") else "") + f"_r{res}" + \
       ("_tf32" if args.tf32 else "") + (f"_rows{args.row0}-{row1}" if (args.row0, row1) != (0, res) else "") + args.tag
os.makedirs(os.path.join(CACHE, "grid"), exist_ok=True)
base = os.path.join(CACHE, "grid", name)

net = load_trained(dev)
dx, dy = random_direction(net, 1), random_direction(net, 2)
bnet = BatchedNet(net, dx, dy, dev)
xtr, ytr = load_cifar(True, dev)
xte, yte = load_cifar(False, dev)
fix = torch.tensor(fixed_indices(args.n, seed=0), device=dev)
xa, ya = xtr[fix], ytr[fix]
probe_idx = torch.tensor(np.sort(np.random.default_rng(7).permutation(10000)[:256]), device=dev)
xp = xte[probe_idx]
axis = torch.tensor(grid_axis(res), device=dev, dtype=torch.float32)

fields = ["loss", "acc"] + (["agree", "pc1", "pc2", "pc3"] if args.kind == "A" else [])
P = args.P or {"A": 16, "A16": 16, "C": 256, "B": 1024, "D": 1024}[args.kind]
P = min(P, res)


@torch.no_grad()
def probe_probs(a, b):
    return torch.softmax(bnet.forward(xp, a, b), -1).permute(1, 0, 2)          # (P, 256, 10)


@torch.no_grad()
def eval_chunk(a, b, row_seed):
    """returns dict of (P,) tensors for grid points (a_p, b_p)."""
    Pn = a.shape[0]
    out = {}
    if args.kind in ("A", "A16"):
        tot = torch.zeros(Pn, device=dev, dtype=torch.float64); cor = torch.zeros(Pn, device=dev)
        for i in range(0, args.n, args.chunk):
            x, y = xa[i:i + args.chunk], ya[i:i + args.chunk]
            lg = bnet.forward(x, a, b, bn_train=(args.kind == "A16"), gs=16)     # (N, P, 10)
            ce = F.cross_entropy(lg.permute(0, 2, 1), y[:, None].expand(-1, Pn), reduction="none")
            tot += ce.double().sum(0); cor += (lg.argmax(-1) == y[:, None]).float().sum(0)
        out["loss"] = tot / args.n; out["acc"] = cor / args.n
        if args.kind == "A":
            pr = probe_probs(a, b)
            out["agree"] = (pr.argmax(-1) == centre_arg[None]).float().mean(1)
            z = (pr.reshape(Pn, -1).double() - pca_mu) @ pca_V                   # (P, 3)
            out["pc1"], out["pc2"], out["pc3"] = z[:, 0], z[:, 1], z[:, 2]
    elif args.kind == "C":
        x, y = xa[:16], ya[:16]
        lg = bnet.forward(x, a, b, bn_train=True)
        out["loss"] = F.cross_entropy(lg.permute(0, 2, 1), y[:, None].expand(-1, Pn), reduction="none").double().mean(0)
        out["acc"] = (lg.argmax(-1) == y[:, None]).float().mean(0)
    else:  # B, D: fresh minibatch of 16 per grid point
        g = torch.Generator(device=dev).manual_seed(row_seed)
        idx = torch.randint(0, 50000, (Pn, 16), device=dev, generator=g)
        x = xtr[idx]                                                               # (P, 16, 3, 32, 32)
        y = ytr[idx].t()                                                           # (16, P)
        lg = bnet.forward(x, a, b, bn_train=(args.kind == "B"), shared_input=False)
        out["loss"] = F.cross_entropy(lg.permute(0, 2, 1), y, reduction="none").double().mean(0)
        out["acc"] = (lg.argmax(-1) == y).float().mean(0)
    return out


if args.kind == "A":
    z0 = torch.zeros(1, device=dev)
    centre_arg = probe_probs(z0, z0)[0].argmax(-1)
    pca_path = os.path.join(CACHE, "probe_pca.npz")
    if os.path.exists(pca_path):
        d = np.load(pca_path)
        pca_mu = torch.tensor(d["mu"], device=dev); pca_V = torch.tensor(d["V"], device=dev)
    else:  # basis from a 64^2 coarse grid over the same plane
        c = torch.tensor(grid_axis(64), device=dev, dtype=torch.float32)
        B_, A_ = torch.meshgrid(c, c, indexing="ij")
        feats = []
        for i in range(0, 4096, 64):
            feats.append(probe_probs(A_.flatten()[i:i + 64], B_.flatten()[i:i + 64]).reshape(64, -1).double())
        X = torch.cat(feats)
        mu = X.mean(0); U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
        ev = (S ** 2) / (S ** 2).sum()
        np.savez(pca_path, mu=mu.cpu().numpy(), V=Vh[:3].t().contiguous().cpu().numpy(),
                 explained=ev[:10].cpu().numpy(), centre_arg=centre_arg.cpu().numpy(),
                 probe_idx=probe_idx.cpu().numpy())
        print("PCA explained", ev[:5].tolist(), flush=True)
        pca_mu, pca_V = mu, Vh[:3].t().contiguous()

if args.bench:
    for Pb in ([8, 16, 32, 64] if args.kind in ("A", "A16") else [64, 128, 256, 512, 1024]):
        a = axis[:Pb].contiguous(); b = torch.full_like(a, 0.3)
        eval_chunk(a, b, 0); torch.cuda.synchronize()
        t = time.time(); reps = 3
        for r in range(reps):
            eval_chunk(a, b, r)
        torch.cuda.synchronize(); dt = (time.time() - t) / reps
        print(json.dumps(dict(kind=args.kind, P=Pb, n=args.n, tf32=args.tf32, s_per_chunk=dt,
                              pts_per_s=Pb / dt, est_h_1024sq=1024 ** 2 / (Pb / dt) / 3600,
                              maxmem_GB=torch.cuda.max_memory_allocated() / 1e9)), flush=True)
    raise SystemExit

# ---------------------------------------------------------------- resumable row loop
meta_path = base + "_meta.json"
nrows = row1 - args.row0
mode = "r+" if os.path.exists(meta_path) else "w+"
arrs = {f: np.lib.format.open_memmap(f"{base}_{f}.npy", mode=mode, dtype=np.float32, shape=(nrows, res))
        for f in fields}
meta = json.load(open(meta_path)) if mode == "r+" else dict(vars(args), name=name, rows_done=0, wall_s=0.0)
start = meta["rows_done"]
if start > 0:
    pasar_job.resumed(start)
t0 = time.time() - meta["wall_s"]
for r in range(start, nrows):
    j = args.row0 + r                                   # row index = beta index
    outs = {f: [] for f in fields}
    for c0 in range(0, res, P):
        a = axis[c0:c0 + P].contiguous()
        b = torch.full_like(a, float(axis[j]))
        o = eval_chunk(a, b, row_seed=args.seed * 1_000_003 + j * 1009 + c0)
        for f in fields:
            outs[f].append(o[f].float())
    for f in fields:
        arrs[f][r] = torch.cat(outs[f]).cpu().numpy()
    if r % 8 == 7 or r == nrows - 1:
        for f in fields:
            arrs[f].flush()
        meta["rows_done"] = r + 1; meta["wall_s"] = time.time() - t0
        json.dump(meta, open(meta_path, "w"))
        pasar_job.checkpoint(r + 1)
        pasar_job.progress(r + 1, nrows, loss_mean=float(arrs["loss"][r].mean()))
        print(f"row {r + 1}/{nrows}  {time.time() - t0:.0f}s  loss[{j}] mean {arrs['loss'][r].mean():.4f}", flush=True)
meta["done"] = True; json.dump(meta, open(meta_path, "w"))
print("done", name, time.time() - t0)
