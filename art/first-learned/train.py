"""Train one small network on all 60,000 training images and record, at a dense grid of steps,
the predicted label of EVERY training example (a full pass over the train set, eval mode).

Output cache/runs/<task>_<arch>_<labels>_s<seed>.npz
  steps : (K,) evaluation steps (0 = untrained init)
  pred  : (K, 60000) uint8 predicted label at each evaluation step
  y     : (60000,) the labels trained on (true or shuffled)
  p_fin : (60000, 10) float16 final softmax
  test_acc : final test accuracy
Matmuls/convs in TF32 (train and eval). Optimiser for every run: SGD, momentum 0.9, lr 0.02, batch 128, no weight decay, no augmentation
(the same default as one-road). Minibatch order is a fixed permutation per epoch drawn from the seed.
"""
import argparse, os, sys, struct, time
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "one-road"))
from models import build  # one-road's architectures, unchanged

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="mnist")        # mnist | fashion
ap.add_argument("--arch", required=True)          # mlp_256 | cnn | resnet | vit | ...
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--labels", default="true")       # true | shuf
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--bn", default="running")      # running | batch : BatchNorm statistics used when evaluating
a = ap.parse_args()

pasar_job.apply_memory_limit()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True  # TF32 in train and eval
torch.backends.cudnn.benchmark = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed)


def idx(path):
    with open(path, "rb") as f:
        nd = struct.unpack(">I", f.read(4))[0] & 0xFF
        shape = struct.unpack(">" + "I" * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


raw = os.path.join(ROOT, "..", "data", {"mnist": "MNIST", "fashion": "FashionMNIST"}[a.task], "raw")
xtr = idx(os.path.join(raw, "train-images-idx3-ubyte"))[:, None].astype(np.float32) / 255
ytr = idx(os.path.join(raw, "train-labels-idx1-ubyte")).astype(np.int64)
xte = idx(os.path.join(raw, "t10k-images-idx3-ubyte"))[:, None].astype(np.float32) / 255
yte = idx(os.path.join(raw, "t10k-labels-idx1-ubyte")).astype(np.int64)
m, s = xtr.mean(), xtr.std()
xtr = torch.tensor((xtr - m) / s, device=dev); xte = torch.tensor((xte - m) / s, device=dev)
if a.labels == "shuf":  # one fixed relabelling shared by every shuffled run (seed 12345)
    ytr = np.random.RandomState(12345).permutation(ytr)
y_np = ytr.copy()
ytr = torch.tensor(ytr, device=dev); yte = torch.tensor(yte, device=dev)
N = len(ytr)

model = build(a.task, a.arch, 10, (1, 28)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)
spe = N // a.bs
T = a.epochs * spe
# evaluation grid: every 2 steps to 100, every 10 to 500, then every 50 (declared; resolves the fast early phase)
steps = np.unique(np.concatenate([np.arange(0, 101, 2), np.arange(100, 501, 10), np.arange(500, T + 1, 50), [T]]))
steps = steps[steps <= T]
stepset = set(steps.tolist())


@torch.no_grad()
def predict(x, chunk=10000):
    if a.bn == "batch":  # evaluate with BN batch statistics (chunks of 10k), without touching running stats
        bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        mom = [m.momentum for m in bns]
        for m in bns: m.momentum = 0.0
        model.train(); out = torch.cat([model(x[i:i + chunk]).argmax(1) for i in range(0, len(x), chunk)])
        for m, mo in zip(bns, mom): m.momentum = mo
        return out
    model.eval()
    out = torch.cat([model(x[i:i + chunk]).argmax(1) for i in range(0, len(x), chunk)])
    model.train()
    return out


@torch.no_grad()
def probs(x, chunk=10000):
    model.eval()
    out = torch.cat([F.softmax(model(x[i:i + chunk]), 1) for i in range(0, len(x), chunk)])
    model.train()
    return out


name = f"{a.task}_{a.arch}{'bnb' if a.bn == 'batch' else ''}_{a.labels}_s{a.seed}"
out = os.path.join(ROOT, "cache", "runs"); os.makedirs(out, exist_ok=True)
pred = np.zeros((len(steps), N), np.uint8)
g = torch.Generator(device="cpu"); g.manual_seed(1000 + a.seed)
t0 = time.time(); k = 0; step = 0
pred[k] = predict(xtr).cpu().numpy(); k += 1
model.train()
for ep in range(a.epochs):
    perm = torch.randperm(N, generator=g).to(dev)
    for b in range(spe):
        i = perm[b * a.bs:(b + 1) * a.bs]
        loss = F.cross_entropy(model(xtr[i]), ytr[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if step in stepset:
            pred[k] = predict(xtr).cpu().numpy(); k += 1
            if k % 10 == 0:
                acc = float((torch.tensor(pred[k - 1]) == torch.tensor(y_np)).float().mean())
                pasar_job.progress(step, T, loss=loss.item(), train_acc=acc)
                print(f"step {step}/{T} loss {loss.item():.3f} train_acc {acc:.4f} {time.time()-t0:.0f}s", flush=True)
assert k == len(steps), (k, len(steps))
test_acc = float((predict(xte) == yte).float().mean())
pf = probs(xtr).cpu().numpy().astype(np.float16)
np.savez_compressed(os.path.join(out, name + ".npz"), steps=steps, pred=pred, y=y_np, p_fin=pf,
                    test_acc=test_acc, secs=time.time() - t0)
print(name, "final train acc", (pred[-1] == y_np).mean(), "test acc", test_acc, f"{time.time()-t0:.0f}s")
