"""Compute step: train a CIFAR-10 ResNet18 into the terminal phase and dump
penultimate-layer statistics + feature subsets at many epochs.

Setup follows Papyan, Han & Donoho 2020 (arXiv:2008.08186) Sec. E-G:
  no augmentation, per-channel standardisation, SGD momentum 0.9, wd 5e-4,
  batch 128, 350 epochs, lr /10 at 1/3 and 2/3 of training.
Deviations (declared in README): one learning rate instead of a 25-lr sweep,
bf16 autocast for the training forward/backward (features extracted in fp32,
statistics in float64), a 4-class subset run in addition to full 10-class.

Usage:
  python train_nc.py --classes 0 1 2 3 --tag c4 --epochs 350
  python train_nc.py --classes 0 1 2 3 4 5 6 7 8 9 --tag c10 --epochs 350
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--classes", type=int, nargs="+", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--epochs", type=int, default=350)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--width", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_sub_train", type=int, default=500, help="stored train feats per class")
p.add_argument("--n_sub_test", type=int, default=500, help="stored test feats per class")
p.add_argument("--max_epochs_run", type=int, default=None, help="stop early (toy timing)")
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
dev = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", args.tag)
os.makedirs(OUT, exist_ok=True)
DATA = "/home/fzeng/ml/research/art/data/cifar-10-batches-py"
CLASSES = args.classes
C = len(CLASSES)


def load(files):
    xs, ys = [], []
    for f in files:
        with open(os.path.join(DATA, f), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"]); ys.extend(d[b"labels"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32)
    y = np.array(ys)
    keep = np.isin(y, CLASSES)
    remap = {c: i for i, c in enumerate(CLASSES)}
    return x[keep], np.array([remap[v] for v in y[keep]])


xtr, ytr = load([f"data_batch_{i}" for i in range(1, 6)])
xte, yte = load(["test_batch"])
assert np.all(np.bincount(ytr) == np.bincount(ytr)[0]), "classes must be balanced"
xtr = torch.tensor(xtr, dtype=torch.float32, device=dev) / 255
xte = torch.tensor(xte, dtype=torch.float32, device=dev) / 255
mean = xtr.mean((0, 2, 3), keepdim=True); std = xtr.std((0, 2, 3), keepdim=True)
xtr = ((xtr - mean) / std).contiguous(memory_format=torch.channels_last)
xte = ((xte - mean) / std).contiguous(memory_format=torch.channels_last)
ytr = torch.tensor(ytr, device=dev); yte = torch.tensor(yte, device=dev)
Ntr, Nte = len(ytr), len(yte)
print(f"C={C} Ntr={Ntr} Nte={Nte}", flush=True)

rng = np.random.RandomState(1234)
sub_tr = np.concatenate([rng.choice(np.where(ytr.cpu().numpy() == c)[0], args.n_sub_train, replace=False) for c in range(C)])
sub_te = np.concatenate([rng.choice(np.where(yte.cpu().numpy() == c)[0], args.n_sub_test, replace=False) for c in range(C)])


# ---------------- ResNet18, CIFAR variant (3x3 stem, no max-pool) -------------
class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.sh = nn.Sequential()
        if st != 1 or i != o:
            s.sh = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        out = F.relu(s.b1(s.c1(x)))
        out = s.b2(s.c2(out))
        return F.relu(out + s.sh(x))


class ResNet18(nn.Module):
    def __init__(s, C, w=64):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        layers, i = [], w
        for o, st in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers += [Block(i, o, st), Block(o, o, 1)]; i = o
        s.body = nn.Sequential(*layers)
        s.fc = nn.Linear(8 * w, C)

    def features(s, x):
        return F.adaptive_avg_pool2d(s.body(s.stem(x)), 1).flatten(1)

    def forward(s, x):
        return s.fc(s.features(x))


model = ResNet18(C, args.width).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
E = args.epochs
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [E // 3, 2 * E // 3], 0.1)


def ckpt_epochs(E):
    s = set(range(0, 11)) | set(range(12, 51, 2)) | set(range(55, E + 1, 5)) | {E}
    return sorted(e for e in s if e <= E)


CK = set(ckpt_epochs(E))


@torch.no_grad()
def extract(x):
    model.eval()
    hs = []
    for i in range(0, len(x), 1000):
        hs.append(model.features(x[i:i + 1000]).float())
    model.train()
    return torch.cat(hs)


@torch.no_grad()
def measure(epoch):
    Htr = extract(xtr).double(); Hte = extract(xte).double()
    W = model.fc.weight.detach().double(); b = model.fc.bias.detach().double()
    out = {"epoch": epoch}
    muG = Htr.mean(0)
    mu = torch.stack([Htr[ytr == c].mean(0) for c in range(C)])  # C x d
    M = mu - muG
    Hc = Htr - mu[ytr]
    SW = Hc.T @ Hc / Ntr
    SB = M.T @ M / C
    SBp = torch.linalg.pinv(SB, hermitian=True, rtol=1e-10)
    out["nc1"] = float(torch.trace(SW @ SBp) / C)
    out["tr_SW"] = float(torch.trace(SW)); out["tr_SB"] = float(torch.trace(SB))
    # test NC1 with test means
    muGt = Hte.mean(0)
    mut = torch.stack([Hte[yte == c].mean(0) for c in range(C)])
    Mt = mut - muGt
    Hct = Hte - mut[yte]
    SWt = Hct.T @ Hct / Nte
    SBt = Mt.T @ Mt / C
    out["nc1_test"] = float(torch.trace(SWt @ torch.linalg.pinv(SBt, hermitian=True, rtol=1e-10)) / C)

    def nc2(V):
        n = V.norm(dim=1)
        G = (V / n[:, None]) @ (V / n[:, None]).T
        off = G[~torch.eye(C, dtype=bool, device=dev)]
        return dict(equinorm=float(n.std() / n.mean()), cos_std=float(off.std()),
                    cos_dev=float((off + 1 / (C - 1)).abs().mean()), gram=G.cpu().numpy())
    a = nc2(M); w = nc2(W); at = nc2(Mt)
    for k in ["equinorm", "cos_std", "cos_dev"]:
        out[f"M_{k}"] = a[k]; out[f"W_{k}"] = w[k]; out[f"Mtest_{k}"] = at[k]
    out["nc3"] = float((W / W.norm() - M / M.norm()).norm())
    # NC4 + accuracy
    for name, H, y in [("train", Htr, ytr), ("test", Hte, yte)]:
        logits = H @ W.T + b
        pred = logits.argmax(1)
        ncc = torch.cdist(H, mu).argmin(1)  # nearest *train* class mean
        out[f"acc_{name}"] = float((pred == y).double().mean())
        out[f"loss_{name}"] = float(F.cross_entropy(logits, y))
        out[f"nc4_{name}"] = float((pred != ncc).double().mean())
    out["w_norm"] = float(W.norm()); out["M_norm"] = float(M.norm())
    # orthonormal basis of train class-mean subspace (C-1 dims) and projected
    # per-class covariance there
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    B = Vh[: C - 1].T  # d x (C-1)
    covs = []
    for c in range(C):
        Z = Hc[ytr == c] @ B
        covs.append((Z.T @ Z / len(Z)).cpu().numpy())
    arrays = dict(
        mu=mu.cpu().numpy(), muG=muG.cpu().numpy(), mu_test=mut.cpu().numpy(),
        W=W.cpu().numpy(), b=b.cpu().numpy(), gram_M=a["gram"], gram_W=w["gram"],
        gram_Mtest=at["gram"], sv_M=S.cpu().numpy(), cov_sub=np.stack(covs),
        SW_eig=torch.linalg.eigvalsh(SW).cpu().numpy(),
        h_train=Htr[sub_tr].float().cpu().numpy().astype(np.float16),
        h_test=Hte[sub_te].float().cpu().numpy().astype(np.float16),
    )
    # float16 overflow guard: features are O(1-10) after BN/ReLU/avgpool
    assert np.isfinite(arrays["h_train"]).all()
    np.savez_compressed(os.path.join(OUT, f"ep{epoch:04d}.npz"), **arrays)
    return out


meta = dict(vars(args), classes=CLASSES, sub_train_idx=sub_tr.tolist(), sub_test_idx=sub_te.tolist(),
            y_sub_train=ytr[torch.tensor(sub_tr, device=dev)].tolist(),
            y_sub_test=yte[torch.tensor(sub_te, device=dev)].tolist(), ckpts=sorted(CK))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
logf = open(os.path.join(OUT, "metrics.jsonl"), "a" if False else "w")
t0 = time.time()
m = measure(0); m["time"] = 0; logf.write(json.dumps(m) + "\n"); logf.flush()
print(m, flush=True)
last = E if args.max_epochs_run is None else args.max_epochs_run
for ep in range(1, last + 1):
    model.train()
    perm = torch.randperm(Ntr, device=dev)
    tl, tc = 0.0, 0
    for i in range(0, Ntr, args.bs):
        idx = perm[i:i + args.bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xtr[idx])
        loss = F.cross_entropy(logits.float(), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tl += loss.detach() * len(idx); tc += (logits.argmax(1) == ytr[idx]).sum()
    sched.step()
    msg = f"ep {ep} loss {float(tl)/Ntr:.5f} acc {float(tc)/Ntr:.4f} lr {sched.get_last_lr()[0]:.4g} t {time.time()-t0:.0f}s"
    if ep in CK:
        m = measure(ep); m["time"] = time.time() - t0; m["lr"] = sched.get_last_lr()[0]
        logf.write(json.dumps(m) + "\n"); logf.flush()
        msg += f" | NC1 {m['nc1']:.3g} cosstd {m['M_cos_std']:.3g} NC3 {m['nc3']:.3g} NC4 {m['nc4_train']:.3g} tracc {m['acc_train']:.4f} teacc {m['acc_test']:.4f}"
    print(msg, flush=True)
torch.save(model.state_dict(), os.path.join(OUT, "final_model.pt"))
print("done", time.time() - t0, flush=True)
