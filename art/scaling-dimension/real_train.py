"""Real-data check (Sharma & Kaplan sec. 3.2): width-scaled TF-tutorial CNN on MNIST / FashionMNIST /
CIFAR10. For each channel width c: Conv(c)-pool-Conv(2c)-pool-Conv(2c)-Dense(2c)-Dense(10), ReLU,
Adam, no augmentation, early stopping on test loss (the paper's only regulariser).
Records per width: N, best test loss / error, the train loss at that epoch, and the TwoNN / MLE ID
of the final hidden layer (Dense(2c)) on the 10k test images at the best epoch.
Also the ID of the raw pixels (10k training images).

    gpu_run.sh python real_train.py --data mnist,fmnist,cifar10 --out cache/real/real.json
"""
import argparse, json, os, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from idlib import knn_dists, twonn, mle_levina_bickel

p = argparse.ArgumentParser()
p.add_argument("--data", default="mnist,fmnist,cifar10")
p.add_argument("--widths", default="2,3,4,6,8,12,16,24,32")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.cuda.set_per_process_memory_fraction(0.08)
dev = "cuda"
ROOT = "/home/fzeng/ml/research/art/data"


def load(name):
    cls = dict(mnist=torchvision.datasets.MNIST, fmnist=torchvision.datasets.FashionMNIST,
               cifar10=torchvision.datasets.CIFAR10)[name]
    out = []
    for train in (True, False):
        ds = cls(ROOT, train=train, download=False)
        X = torch.as_tensor(np.asarray(ds.data)).float() / 255.
        X = X[:, None] if X.ndim == 3 else X.permute(0, 3, 1, 2)
        y = torch.as_tensor(np.asarray(ds.targets)).long()
        out += [X, y]
    Xtr, ytr, Xte, yte = out
    mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True)
    return ((Xtr - mu) / sd).to(dev), ytr.to(dev), ((Xte - mu) / sd).to(dev), yte.to(dev), Xtr


class Net(nn.Module):
    def __init__(self, cin, c, hw):
        super().__init__()
        self.c1 = nn.Conv2d(cin, c, 3); self.c2 = nn.Conv2d(c, 2 * c, 3); self.c3 = nn.Conv2d(2 * c, 2 * c, 3)
        s = ((hw - 2) // 2 - 2) // 2 - 2
        self.f1 = nn.Linear(2 * c * s * s, 2 * c); self.f2 = nn.Linear(2 * c, 10)

    def forward(self, x, hidden=False):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x)).flatten(1)
        h = F.relu(self.f1(x))
        return (self.f2(h), h) if hidden else self.f2(h)


@torch.no_grad()
def evaluate(net, X, y, hidden=False):
    net.eval(); L = 0.; E = 0; H = []
    for i in range(0, len(X), 2000):
        o = net(X[i:i + 2000], hidden)
        if hidden: o, h = o; H.append(h.double().cpu())
        L += F.cross_entropy(o, y[i:i + 2000], reduction="sum").item(); E += (o.argmax(1) != y[i:i + 2000]).sum().item()
    net.train()
    r = (L / len(X), E / len(X))
    return (r, torch.cat(H).numpy()) if hidden else r


def ids(H):
    H = H[:, H.std(0) > 0]
    r = knn_dists(H, 20, device=dev)
    return dict(twonn=twonn(r)["d_fit"], mle10=mle_levina_bickel(r, 10)["d"], mle20=mle_levina_bickel(r, 20)["d"])


res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for name in a.data.split(","):
    Xtr, ytr, Xte, yte, raw = load(name)
    R = res.setdefault(name, {})
    if "pixel_id" not in R:
        g = np.random.default_rng(0).choice(len(raw), 10000, replace=False)
        R["pixel_id"] = ids(raw[g].flatten(1).double().numpy())
        print(name, "pixel ID", R["pixel_id"], flush=True)
    for c in [int(v) for v in a.widths.split(",")]:
        for seed in range(a.seeds):
            key = f"c{c}_s{seed}"
            if key in R: continue
            torch.manual_seed(seed)
            net = Net(Xtr.shape[1], c, Xtr.shape[2]).to(dev)
            N = sum(q.numel() for q in net.parameters())
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)
            best = None; t0 = time.time(); hist = []
            for ep in range(a.epochs):
                perm = torch.randperm(len(Xtr), device=dev)
                trl = 0.
                for i in range(0, len(Xtr), a.batch):
                    idx = perm[i:i + a.batch]
                    loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    trl += loss.item() * len(idx)
                (tl, te), H = evaluate(net, Xte, yte, hidden=True)
                hist.append((tl, te, trl / len(Xtr)))
                if best is None or tl < best["test_loss"]:
                    best = dict(test_loss=tl, test_err=te, train_loss_running=trl / len(Xtr), epoch=ep, **{"id_" + k: v for k, v in ids(H).items()})
            best.update(N=N, c=c, hist=hist, wall=time.time() - t0)
            R[key] = best
            print(name, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items() if k != "hist"}, flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
