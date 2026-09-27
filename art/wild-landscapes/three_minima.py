"""Three minima: the plane through three independently trained MLPs (Garipov et al. 2018, Fig. 1 left),
and the same plane after permutation alignment (weight matching, Ainsworth et al. 2023).

    python three_minima.py train  --device cuda            # 3 nets, seeds 0,1,2  -> cache/three/net{k}.pt
    python three_minima.py plane  --align 0 --res 1024 --device cuda --out cache/three/plane_raw.npz
    python three_minima.py plane  --align 1 ...

Net: MLP 784-256-256-10, ReLU, no BN, Fashion-MNIST.  Each net: its own init seed and data order,
SGD momentum 0.9, lr 0.05 (cosine), wd 5e-4, bs 128, 20 epochs, no augmentation.
Plane: theta(u, v) = c + u e1 + v e2, c = centroid of the three nets, e1 = (theta_1 - c)/|theta_1 - c|,
e2 = the component of theta_2 - c orthogonal to e1, normalised; axes in units of r = |theta_1 - c|
(so each net sits at distance 1 from the centre).  Loss: mean cross-entropy on a FIXED 4096 training
images (first 4096 of a seed-0 permutation), fp32, TF32 off.  Colour data: argmax agreement of the
point's predictions with each of the three nets on 2048 fixed test images, on all probes and on the
'disputed' probes (where the three nets do not all agree).
Alignment: nets 2 and 3 have their hidden units permuted to match net 1 (weight matching, the
function each net computes is unchanged: checked).
"""
import argparse, gzip, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache", "three")
DATA = "/home/fzeng/ml/research/art/data/FashionMNIST/raw"
H = 256


def load(split, device):
    pre = "train" if split == "train" else "t10k"
    with gzip.open(os.path.join(DATA, f"{pre}-images-idx3-ubyte.gz")) as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(os.path.join(DATA, f"{pre}-labels-idx1-ubyte.gz")) as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    x = (x - 0.2860) / 0.3530
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


def init(seed, device):
    g = torch.Generator().manual_seed(seed)
    def lin(i, o):
        b = 1 / math.sqrt(i)
        return ((torch.rand(o, i, generator=g) * 2 - 1) * b).to(device), ((torch.rand(o, generator=g) * 2 - 1) * b).to(device)
    W1, b1 = lin(784, H); W2, b2 = lin(H, H); W3, b3 = lin(H, 10)
    return [W1, b1, W2, b2, W3, b3]


def fwd(p, x):
    W1, b1, W2, b2, W3, b3 = p
    h = F.relu(x @ W1.T + b1); h = F.relu(h @ W2.T + b2)
    return h @ W3.T + b3


def train(device):
    xtr, ytr = load("train", device); xte, yte = load("test", device)
    os.makedirs(C, exist_ok=True)
    for seed in range(3):
        p = [t.requires_grad_() for t in init(seed, device)]
        opt = torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=5e-4)
        E, bs = 20, 128
        steps = E * (60000 // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        g = torch.Generator(device=device).manual_seed(100 + seed)
        for e in range(E):
            perm = torch.randperm(60000, device=device, generator=g)
            for i in range(60000 // bs):
                idx = perm[i * bs:(i + 1) * bs]
                loss = F.cross_entropy(fwd(p, xtr[idx]), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            tr = F.cross_entropy(fwd(p, xtr), ytr).item(); acc = (fwd(p, xte).argmax(1) == yte).float().mean().item()
        print(f"net{seed}: train loss {tr:.4f} test acc {acc:.4f}", flush=True)
        torch.save(dict(p=[t.detach().cpu() for t in p], train_loss=tr, test_acc=acc), os.path.join(C, f"net{seed}.pt"))


def weight_match(pa, pb, iters=30):
    """Permute hidden units of net b to match net a (Ainsworth et al. 2023, weight matching, MLP)."""
    from scipy.optimize import linear_sum_assignment
    W1a, b1a, W2a, b2a, W3a, _ = [t.double() for t in pa]
    W1b, b1b, W2b, b2b, W3b, b3b = [t.double() for t in pb]
    P1 = torch.arange(H); P2 = torch.arange(H)
    for it in range(iters):
        changed = False
        # layer 1 perm: incoming (W1, b1) and outgoing (W2 columns, with layer-2 perm applied)
        S = W1a @ W1b.T + torch.outer(b1a, b1b) + W2a.T @ W2b[P2]
        r, c = linear_sum_assignment(S.numpy(), maximize=True); n1 = torch.tensor(c)
        changed |= bool((n1 != P1).any()); P1 = n1
        S = W2a @ W2b[:, P1].T + torch.outer(b2a, b2b) + W3a.T @ W3b
        r, c = linear_sum_assignment(S.numpy(), maximize=True); n2 = torch.tensor(c)
        changed |= bool((n2 != P2).any()); P2 = n2
        if not changed:
            break
    out = [pb[0][P1], pb[1][P1], pb[2][P2][:, P1], pb[3][P2], pb[4][:, P2], pb[5]]
    return out, it


def flat(p):
    return torch.cat([t.flatten() for t in p])


def unflat(v, like):
    out, i = [], 0
    for t in like:
        out.append(v[..., i:i + t.numel()].reshape(*v.shape[:-1], *t.shape)); i += t.numel()
    return out


@torch.no_grad()
def plane(args):
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    nets = [torch.load(os.path.join(C, f"net{k}.pt"))["p"] for k in range(3)]
    xtr, ytr = load("train", dev); xte, yte = load("test", dev)
    info = {}
    if args.align:
        for k in (1, 2):
            al, it = weight_match(nets[0], nets[k])
            d = (fwd([t.to(dev) for t in al], xte) - fwd([t.to(dev) for t in nets[k]], xte)).abs().max().item()
            info[f"align{k}"] = dict(iters=it, max_logit_change=d)
            nets[k] = al
    g = torch.Generator().manual_seed(0)
    itr = torch.randperm(60000, generator=g)[:args.n].to(dev); ite = torch.arange(2048, device=dev)
    x, y, xp = xtr[itr], ytr[itr], xte[ite]
    T = [flat(p).double() for p in nets]
    c = (T[0] + T[1] + T[2]) / 3
    e1 = T[0] - c; r = e1.norm(); e1 = e1 / r
    e2 = T[1] - c; e2 = e2 - (e2 @ e1) * e1; e2 = e2 / e2.norm()
    marks = [[float((t - c) @ e1 / r), float((t - c) @ e2 / r)] for t in T]
    off = float(((T[2] - c) - ((T[2] - c) @ e1) * e1 - ((T[2] - c) @ e2) * e2).norm() / r)
    info.update(r=float(r), marks=marks, off_plane_net3=off, dist=[float((T[i] - T[j]).norm()) for i, j in ((0, 1), (0, 2), (1, 2))])
    ref = [fwd([t.to(dev) for t in p], xp).argmax(1) for p in nets]
    disputed = ~((ref[0] == ref[1]) & (ref[1] == ref[2]))
    info["disputed_frac"] = float(disputed.float().mean())
    like = nets[0]
    cf, e1f, e2f = (v.float().to(dev) * s for v, s in ((c, 1.0), (e1, float(r)), (e2, float(r))))
    C0, E1, E2 = unflat(cf, like), unflat(e1f, like), unflat(e2f, like)
    # first layer is linear in (u, v): precompute x W^T for c, e1, e2
    X1 = [x @ w.T for w in (C0[0], E1[0], E2[0])]; X1p = [xp @ w.T for w in (C0[0], E1[0], E2[0])]
    ax = np.linspace(-args.span, args.span, args.res)
    R = args.res
    out = {k: np.zeros((R, R), np.float32) for k in ("loss", "acc", "ag0", "ag1", "ag2", "dg0", "dg1", "dg2")}
    import pasar_job
    Pc = args.P
    for i, vv in enumerate(ax):
        for j0 in range(0, R, Pc):
            u = torch.tensor(ax[j0:j0 + Pc], device=dev, dtype=torch.float32); v = torch.full_like(u, float(vv))
            p_ = u.shape[0]
            def layer_rest(X1_, xs):
                h = X1_[0][None] + u[:, None, None] * X1_[1][None] + v[:, None, None] * X1_[2][None]
                h = h + (C0[1][None] + u[:, None] * E1[1][None] + v[:, None] * E2[1][None])[:, None, :]
                h = F.relu(h)
                W2 = C0[2][None] + u[:, None, None] * E1[2][None] + v[:, None, None] * E2[2][None]
                b2 = C0[3][None] + u[:, None] * E1[3][None] + v[:, None] * E2[3][None]
                h = F.relu(torch.bmm(h, W2.transpose(1, 2)) + b2[:, None])
                W3 = C0[4][None] + u[:, None, None] * E1[4][None] + v[:, None, None] * E2[4][None]
                b3 = C0[5][None] + u[:, None] * E1[5][None] + v[:, None] * E2[5][None]
                return torch.bmm(h, W3.transpose(1, 2)) + b3[:, None]          # (p, N, 10)
            lo = layer_rest(X1, x)
            L = F.cross_entropy(lo.reshape(-1, 10).double(), y.repeat(p_), reduction="none").view(p_, -1).mean(1)
            out["loss"][i, j0:j0 + p_] = L.cpu().numpy()
            out["acc"][i, j0:j0 + p_] = (lo.argmax(-1) == y[None]).float().mean(1).cpu().numpy()
            pr = layer_rest(X1p, xp).argmax(-1)
            for k in range(3):
                m = (pr == ref[k][None]).float()
                out[f"ag{k}"][i, j0:j0 + p_] = m.mean(1).cpu().numpy()
                out[f"dg{k}"][i, j0:j0 + p_] = m[:, disputed].mean(1).cpu().numpy()
        if i % 16 == 0:
            pasar_job.progress(i + 1, R)
    np.savez_compressed(args.out, span=args.span, align=args.align, info=json.dumps(info), **out)
    print(json.dumps(dict(out=args.out, lmin=float(out["loss"].min()), lmax=float(out["loss"].max()), **info)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--align", type=int, default=0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--span", type=float, default=1.8)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pasar_job; pasar_job.apply_memory_limit()
    if a.cmd == "train":
        train(a.device)
    else:
        plane(a)


if __name__ == "__main__":
    main()
