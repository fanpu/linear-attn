"""Iterative magnitude pruning (IMP) of LeNet-300-100, vectorised over models.

One process = one condition (dataset x preprocessing x optimiser x rewind x mask mode),
with several seeds trained side by side as a stacked batch of independent MLPs (bmm).
The loss is a sum of per-model means and Adam/SGD are elementwise, so each model trains
exactly as it would alone.

Follows Frankle & Carbin (ICLR 2019) for LeNet-300-100: Gaussian Glorot init, zero
biases, Adam lr 1.2e-3, batch 60, layer-wise magnitude pruning of 20% of the remaining
weights per round in the two hidden layers and 10% in the output layer, biases never
pruned, surviving weights rewound to their init values. Training length is shorter
than the paper (declared in README; --steps).

Mask modes per model:
  imp     : prune by |final trained weight| among survivors, rewind to theta_0 (or theta_k)
  reinit  : copy the mask of the paired imp model each round, fresh random init (F&C control)
  random  : nested random masks at the same per-layer counts, init theta_0
  maginit : keep the largest |theta_0| at the same per-layer counts (no training signal)

Outputs one npz per round in cache/<cond>/round_XX.npz (resumable at round granularity).
"""
import argparse, os, json, time, gzip
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job

ROOT = "/home/fzeng/ml/research/art"
SIZES = [(784, 300), (300, 100), (100, 10)]
KEEP_RATE = [0.8, 0.8, 0.9]


def load_idx(path):
    with open(path, "rb") as f:
        data = f.read()
    nd = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def load_dataset(name):
    d = f"{ROOT}/data/{'MNIST' if name == 'mnist' else 'FashionMNIST'}/raw"
    xtr = load_idx(f"{d}/train-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    ytr = load_idx(f"{d}/train-labels-idx1-ubyte").astype(np.int64)
    xte = load_idx(f"{d}/t10k-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    yte = load_idx(f"{d}/t10k-labels-idx1-ubyte").astype(np.int64)
    return xtr, ytr, xte, yte


def keep_counts(rounds):
    """Per-layer surviving weight counts for rounds 0..rounds-1 (iterative, rounded)."""
    out = []
    cur = [a * b for a, b in SIZES]
    for r in range(rounds):
        out.append(list(cur))
        cur = [int(round(c * k)) for c, k in zip(cur, KEEP_RATE)]
    return out


def glorot_init(M, seeds, gen_offset, device):
    Ws = []
    for li, (fi, fo) in enumerate(SIZES):
        W = torch.empty(M, fi, fo)
        for m in range(M):
            g = torch.Generator().manual_seed(int(seeds[m]) * 1000 + gen_offset * 10 + li)
            W[m] = torch.randn(fi, fo, generator=g) * np.sqrt(2.0 / (fi + fo))
        Ws.append(W.to(device))
    return Ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--prep", default="norm", choices=["norm", "raw"])
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--rewind", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="imp", help="comma list; each mode applied to every seed")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/ticket-shadow/cache")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if dev.type == "cuda":
        pasar_job.apply_memory_limit()
        torch.backends.cuda.matmul.allow_tf32 = False
    outdir = f"{args.out}/{args.cond}"
    os.makedirs(outdir, exist_ok=True)
    json.dump(vars(args), open(f"{outdir}/args.json", "w"), indent=1)

    seeds0 = [int(s) for s in args.seeds.split(",")]
    modes0 = args.modes.split(",")
    seeds, modes = [], []
    for md in modes0:
        for s in seeds0:
            seeds.append(s); modes.append(md)
    M = len(seeds)
    lr = args.lr if args.lr is not None else (1.2e-3 if args.opt == "adam" else 0.1)

    # ---------------- data
    xtr, ytr, xte, yte = load_dataset(args.dataset)
    if args.prep == "norm":  # torchvision/open_lth convention: global mean/std of the train set
        mu, sd = float(xtr.mean()), float(xtr.std())
        xtr = (xtr - mu) / sd; xte = (xte - mu) / sd
    xva, yva = xtr[55000:], ytr[55000:]
    xtr, ytr = xtr[:55000], ytr[:55000]
    perms = np.stack([np.arange(784)] * M)
    if args.perm:
        for m in range(M):
            perms[m] = np.random.RandomState(10_000 + seeds[m]).permutation(784)
    # per-model data copies only if permuted
    D = M if args.perm else 1
    dmap = torch.arange(M, device=dev) if args.perm else torch.zeros(M, dtype=torch.long, device=dev)

    def stack(x):
        return torch.tensor(np.stack([x[:, perms[d]] for d in range(D)]), device=dev)
    Xtr, Xva, Xte = stack(xtr), stack(xva), stack(xte)
    Ytr = torch.tensor(ytr, device=dev); Yva = torch.tensor(yva, device=dev); Yte = torch.tensor(yte, device=dev)
    N = Xtr.shape[1]

    counts = keep_counts(args.rounds)

    # ---------------- init / masks / resume
    theta0 = glorot_init(M, seeds, 0, dev)  # reinit models get the same theta0 in round 0
    masks = [torch.ones(M, fi, fo, dtype=torch.bool, device=dev) for fi, fo in SIZES]
    rewind_W = None
    start_round = 0
    done = sorted(f for f in os.listdir(outdir) if f.startswith("round_") and f.endswith(".npz"))
    if done:
        last = np.load(f"{outdir}/{done[-1]}")
        r_last = int(last["round"])
        z0 = np.load(f"{outdir}/round_00.npz")
        theta0 = [torch.tensor(z0[f"init_W{l}"], device=dev) for l in range(3)]
        if args.rewind > 0:
            rewind_W = [torch.tensor(z0[f"rewind_W{l}"], device=dev) for l in range(3)]
        cur_masks = [torch.tensor(np.unpackbits(last[f"mask{l}"], axis=-1)[..., :SIZES[l][1]].astype(bool), device=dev)
                     for l in range(3)]
        final_W = [torch.tensor(last[f"W{l}"].astype(np.float32), device=dev) for l in range(3)]
        masks = next_masks(cur_masks, final_W, theta0, modes, seeds, counts, r_last + 1, M, dev)
        start_round = r_last + 1
        pasar_job.resumed(start_round)
        print(f"resumed at round {start_round}", flush=True)

    total_steps = args.rounds * args.steps
    t_start = time.time()
    for r in range(start_round, args.rounds):
        # weights for this round
        if r == 0:
            Wi = [w.clone() for w in theta0]
        else:
            base = rewind_W if (args.rewind > 0) else theta0
            Wi = [w.clone() for w in base]
            fresh = glorot_init(M, seeds, r, dev)
            for m in range(M):
                if modes[m] == "reinit":
                    for l in range(3):
                        Wi[l][m] = fresh[l][m]
        Ws = [torch.nn.Parameter(w * mk) for w, mk in zip(Wi, masks)]
        bs = [torch.nn.Parameter(torch.zeros(M, 1, fo, device=dev)) for _, fo in SIZES]
        params = Ws + bs
        if args.opt == "adam":
            opt = torch.optim.Adam(params, lr=lr, fused=dev.type == "cuda")
        else:
            opt = torch.optim.SGD(params, lr=lr)
        mf = [mk.float() for mk in masks]

        def fwd(x, Ws=Ws, bs=bs, mf=mf):
            h = x
            for l in range(3):
                h = torch.baddbmm(bs[l], h, Ws[l] * mf[l])
                if l < 2:
                    h = F.relu(h)
            return h

        @torch.no_grad()
        def evaluate(X, Y):
            accs, losses = torch.zeros(M, device=dev), torch.zeros(M, device=dev)
            for i in range(0, X.shape[1], 5000):
                xb = X[:, i:i + 5000]
                xb = xb if D == M else xb.expand(M, -1, -1)
                lo = fwd(xb)
                yb = Y[i:i + 5000]
                losses += F.cross_entropy(lo.reshape(-1, 10), yb.repeat(M), reduction="none").view(M, -1).sum(1)
                accs += (lo.argmax(-1) == yb).float().sum(1)
            return (losses / X.shape[1]).cpu().numpy(), (accs / X.shape[1]).cpu().numpy()

        steps_this = args.steps - (args.rewind if r > 0 else 0)
        gen = torch.Generator(device=dev).manual_seed(12345 + 7919 * r + seeds[0])
        curve = []  # (step, val_loss[M], val_acc[M], test_acc[M])
        order, pos = None, N
        spe = N // args.batch
        ar = torch.arange(M, device=dev)[:, None]
        for step in range(steps_this + 1):
            if step % args.eval_every == 0 or step == steps_this:
                vl, va = evaluate(Xva, Yva)
                _, ta = evaluate(Xte, Yte)
                curve.append((step, vl, va, ta))
            if r == 0 and args.rewind > 0 and step == args.rewind:
                rewind_W = [w.detach().clone() for w in Ws]
            if step == steps_this:
                break
            if pos + args.batch > N:
                order = torch.argsort(torch.rand(M, N, device=dev, generator=gen), dim=1)
                pos = 0
            idx = order[:, pos:pos + args.batch]; pos += args.batch
            xb = Xtr[dmap[:, None], idx]
            yb = Ytr[idx]
            lo = fwd(xb)
            loss = F.cross_entropy(lo.reshape(-1, 10), yb.reshape(-1), reduction="sum") / args.batch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                done_steps = r * args.steps + step
                pasar_job.progress(done_steps, total_steps, loss=float(loss.detach()) / M)
        final_W = [w.detach() * mk for w, mk in zip(Ws, mf)]
        steps_arr = np.array([c[0] for c in curve])
        vl = np.stack([c[1] for c in curve]); va = np.stack([c[2] for c in curve]); ta = np.stack([c[3] for c in curve])
        es = vl.argmin(0)
        save = dict(round=r, seeds=np.array(seeds), modes=np.array(modes), perms=perms,
                    counts=np.array(counts[r]), steps=steps_arr, val_loss=vl, val_acc=va, test_acc=ta,
                    es_iter=steps_arr[es], es_test_acc=ta[es, np.arange(M)], final_test_acc=ta[-1],
                    b0=bs[0].detach().cpu().numpy(), b1=bs[1].detach().cpu().numpy(), b2=bs[2].detach().cpu().numpy())
        for l in range(3):
            save[f"mask{l}"] = np.packbits(masks[l].cpu().numpy(), axis=-1)
            save[f"W{l}"] = final_W[l].cpu().numpy().astype(np.float16)
            if r == 0:
                save[f"init_W{l}"] = theta0[l].cpu().numpy()
                if args.rewind > 0:
                    save[f"rewind_W{l}"] = rewind_W[l].cpu().numpy()
        np.savez_compressed(f"{outdir}/round_{r:02d}.npz", **save)
        pasar_job.checkpoint(r)
        print(f"[{args.cond}] round {r:2d} keep={counts[r]} es_test={np.round(save['es_test_acc'], 4).tolist()} "
              f"final_test={np.round(save['final_test_acc'], 4).tolist()} t={time.time() - t_start:.0f}s", flush=True)
        if r + 1 < args.rounds:
            masks = next_masks(masks, final_W, theta0, modes, seeds, counts, r + 1, M, dev)
    print("done", flush=True)


def next_masks(masks, final_W, theta0, modes, seeds, counts, r_next, M, dev):
    new = []
    for l in range(3):
        k = counts[r_next][l]
        nm = torch.zeros_like(masks[l])
        for m in range(M):
            md = modes[m]
            if md == "imp":
                score = final_W[l][m].abs().float()
            elif md == "maginit":
                score = theta0[l][m].abs().float()
            elif md == "random":
                g = torch.Generator().manual_seed(int(seeds[m]) * 7 + 99991 * (l + 1))
                score = torch.rand(masks[l][m].shape, generator=g).to(dev)  # fixed scores -> nested masks
            elif md == "reinit":
                continue
            score = torch.where(masks[l][m], score, torch.full_like(score, -1.0))
            top = torch.topk(score.flatten(), k).indices
            flat = torch.zeros(score.numel(), dtype=torch.bool, device=dev)
            flat[top] = True
            nm[m] = flat.view_as(score)
        # reinit models take the mask of the imp model with the same seed
        for m in range(M):
            if modes[m] == "reinit":
                src = [i for i in range(M) if modes[i] == "imp" and seeds[i] == seeds[m]][0]
                nm[m] = nm[src]
        new.append(nm)
    return new


if __name__ == "__main__":
    main()
