"""Train ConvNet4 on CIFAR-10. SGD+Nesterov 0.9, lr 0.05 cosine, wd 5e-4, bs 128,
crop+flip augmentation on GPU, fp32.  Writes cache/convnet4.pt."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job
from dl_common import ConvNet4, load_cifar, CACHE

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
os.makedirs(CACHE, exist_ok=True)
pasar_job.apply_memory_limit()
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
xtr, ytr = load_cifar(True); xte, yte = load_cifar(False)
net = ConvNet4().cuda()
opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
E, bs = args.epochs, 128
steps = E * (50000 // bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1,
                                            anneal_strategy="cos")


def augment(x):
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device); j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None]).view(B, 1, 32, 1).expand(B, 3, 32, 40)
    cols = (j[:, None] + ar[None]).view(B, 1, 1, 32).expand(B, 3, 32, 32)
    out = xp.gather(2, rows).gather(3, cols)
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


@torch.no_grad()
def evaluate(x, y):
    net.eval(); tot = 0.0; cor = 0
    for i in range(0, x.shape[0], 2000):
        out = net(x[i:i + 2000])
        tot += F.cross_entropy(out, y[i:i + 2000], reduction="sum").item()
        cor += (out.argmax(1) == y[i:i + 2000]).sum().item()
    net.train()
    return tot / x.shape[0], cor / x.shape[0]


t0 = time.time(); hist = []
for ep in range(E):
    net.train()
    perm = torch.randperm(50000, device="cuda")
    for k in range(50000 // bs):
        idx = perm[k * bs:(k + 1) * bs]
        loss = F.cross_entropy(net(augment(xtr[idx])), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    trl, tra = evaluate(xtr, ytr); tel, tea = evaluate(xte, yte)
    hist.append(dict(ep=ep + 1, train_loss=trl, train_acc=tra, test_loss=tel, test_acc=tea, t=time.time() - t0))
    print(json.dumps(hist[-1]), flush=True)
    pasar_job.progress(ep + 1, E, train_loss=trl, test_acc=tea)
torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "hist": hist, "args": vars(args)},
           os.path.join(CACHE, "convnet4.pt"))
print("done", time.time() - t0)
