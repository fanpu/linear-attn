"""Deep double descent at small scale: 5-layer CNN (4 conv + FC) of width k on a CIFAR-10 subset with label noise.

Follows Nakkiran et al. (2020) "Deep Double Descent" standard CNN family: conv widths [k, 2k, 4k, 8k],
each conv3x3 -> BN -> ReLU, max-pooling, then a linear layer. Adam lr 1e-4, batch 128, no augmentation.
Reduced scale: n_train subset (default 10k), fewer epochs.

Runs several widths *concurrently in one process* (round-robin, one minibatch step each), which shares
one GPU slot and keeps the tiny widths from idling the GPU. Checkpoints every --ckpt-every epochs and resumes.

    python train_cnn.py --widths 1 2 4 --epochs 3 --n 1000 --tag smoke        # smoke test
    python train_cnn.py --widths 1 2 3 4 --epochs 1000 --noise 0.2 --tag main
"""
import argparse, math, os, pathlib, pickle, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path("/home/fzeng/ml/research/art/data/cifar-10-batches-py")


def load_cifar():
    def unpickle(f):
        with open(f, "rb") as fh:
            return pickle.load(fh, encoding="bytes")
    xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(DATA / f"data_batch_{i}")
        xs.append(d[b"data"]); ys += d[b"labels"]
    xtr = np.concatenate(xs).reshape(-1, 3, 32, 32); ytr = np.array(ys)
    d = unpickle(DATA / "test_batch")
    xte = d[b"data"].reshape(-1, 3, 32, 32); yte = np.array(d[b"labels"])
    return xtr, ytr, xte, yte


def make_data(n, noise, seed=0):
    xtr, ytr, xte, yte = load_cifar()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xtr))[:n]
    x, y = xtr[idx], ytr[idx].copy()
    flip = rng.random(n) < noise
    # noisy label = uniformly random *incorrect* label
    y_noisy = y.copy()
    y_noisy[flip] = (y[flip] + rng.integers(1, 10, flip.sum())) % 10
    mean = x.reshape(-1, 3, 1024).mean((0, 2)) / 255; std = x.reshape(-1, 3, 1024).std((0, 2)) / 255
    norm = lambda a: ((torch.tensor(a, dtype=torch.float32) / 255) - torch.tensor(mean).view(1, 3, 1, 1).float()) / torch.tensor(std).view(1, 3, 1, 1).float()
    return norm(x), torch.tensor(y_noisy), torch.tensor(y), torch.tensor(flip), norm(xte), torch.tensor(yte)


class CNN(nn.Module):
    def __init__(self, k, num_classes=10):
        super().__init__()
        c = [3, k, 2 * k, 4 * k, 8 * k]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(c[i + 1]), nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(1 if i == 0 else 2))  # 32 -> 32 -> 16 -> 8 -> 4
        layers.append(nn.MaxPool2d(4))                         # 4 -> 1
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(8 * k, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def eval_schedule(epochs):
    # fixed grid (independent of --epochs, so a run can be extended by resuming with a larger --epochs)
    e = np.unique(np.round(np.geomspace(1, 4000, 90)).astype(int))
    return set(int(v) for v in e) | set(range(100, 4001, 100)) | {epochs}


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Returns (error, mean CE loss, argmax predictions)."""
    model.eval()
    loss, preds = 0.0, []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[i:i + bs])
        out = out.float()
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        preds.append(out.argmax(1))
    model.train()
    preds = torch.cat(preds)
    return (preds != y).float().mean().item(), loss / len(x), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--mem", type=float, default=0.08)
    a = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(a.mem)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out_dir = HERE / "cache" / "cnn" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, yclean, flip, xte, yte = make_data(a.n, a.noise, seed=0)
    x, y, yclean, flip, xte, yte = [t.to(dev) for t in (x, y, yclean, flip, xte, yte)]
    x = x.contiguous(memory_format=torch.channels_last); xte = xte.contiguous(memory_format=torch.channels_last)
    sched = eval_schedule(a.epochs)

    runs = []
    for k in a.widths:
        path = out_dir / f"n{a.n}_p{int(round(a.noise * 100))}_k{k}_s{a.seed}.pt"
        torch.manual_seed(a.seed * 1000 + k)
        model = CNN(k).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        state = dict(k=k, epoch=0, hist=[], n_params=sum(p.numel() for p in model.parameters()))
        if path.exists():
            ck = torch.load(path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            state = ck["state"]
            print(f"[k={k}] resumed at epoch {state['epoch']}", flush=True)
        runs.append(dict(k=k, model=model, opt=opt, state=state, path=path,
                         gen=torch.Generator(device=dev).manual_seed(a.seed * 7919 + k + state["epoch"])))

    def save(r):
        tmp = r["path"].with_suffix(".tmp")
        torch.save(dict(model=r["model"].state_dict(), opt=r["opt"].state_dict(), state=r["state"], args=vars(a)), tmp)
        os.replace(tmp, r["path"])

    n = len(x)
    steps = math.ceil(n / a.bs)
    t0 = time.time()
    while True:
        active = [r for r in runs if r["state"]["epoch"] < a.epochs]
        if not active:
            break
        perms = [torch.randperm(n, device=dev, generator=r["gen"]) for r in active]
        for s in range(steps):
            for r, perm in zip(active, perms):
                idx = perm[s * a.bs:(s + 1) * a.bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(r["model"](x[idx]), y[idx])
                r["opt"].zero_grad(set_to_none=True)
                loss.backward()
                r["opt"].step()
        for r in active:
            st = r["state"]; st["epoch"] += 1; ep = st["epoch"]
            if ep in sched:
                tr_err, tr_loss, pred = evaluate(r["model"], x, y)
                te_err, te_loss, _ = evaluate(r["model"], xte, yte)
                fit_noisy = (pred[flip] == y[flip]).float().mean().item()      # memorized fraction of noisy labels
                fit_clean = (pred[~flip] == y[~flip]).float().mean().item()
                st["hist"].append(dict(epoch=ep, train_err=tr_err, train_loss=tr_loss, test_err=te_err, test_loss=te_loss,
                                       fit_noisy=fit_noisy, fit_clean=fit_clean, wall=time.time() - t0))
                print(f"[k={r['k']:>2}] ep {ep:>5} train_err {tr_err:.4f} test_err {te_err:.4f} "
                      f"test_loss {te_loss:.3f} memorized {fit_noisy:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if ep % a.ckpt_every == 0 or ep == a.epochs:
                save(r)
    print("done", flush=True)


if __name__ == "__main__":
    main()
