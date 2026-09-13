"""Train a CIFAR-10 ResNet (with or without shortcuts), Li et al. recipe, shortened.

Li et al.: SGD+Nesterov, lr 0.1, momentum 0.9, wd 5e-4, batch 128, 300 epochs with
x0.1 decays at 150/225/275.  Here: same optimizer, E epochs, decays at the same
*fractions* of training (0.5, 0.75, 0.9167).  Standard augmentation (4-px pad random
crop + horizontal flip), done on GPU.  Mixed precision (bf16 autocast) for training only.

usage: python train.py --depth 56 --noshort --epochs 60
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from common import ResNetCifar, load_cifar, model_name, CACHE

p = argparse.ArgumentParser()
p.add_argument("--depth", type=int, default=20)
p.add_argument("--noshort", action="store_true")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
name = model_name(args.depth, not args.noshort)
os.makedirs(os.path.join(CACHE, "ckpt"), exist_ok=True)
logf = open(os.path.join(CACHE, "ckpt", f"{name}_train.log"), "w")

def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

xtr, ytr = load_cifar(True)
xte, yte = load_cifar(False)
net = ResNetCifar(args.depth, not args.noshort).cuda().to(memory_format=torch.channels_last)
fwd = torch.compile(net) if os.environ.get("COMPILE") else net
opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
E = args.epochs
milestones = [int(round(E * f)) for f in (150 / 300, 225 / 300, 275 / 300)]
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, 0.1)
# checkpoints for the training animation (dense early, where the landscape changes most)
save_eps = sorted(set([e for e in [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25] if e <= E] + list(range(0, E + 1, 5)) + [E]))


def save(tag, extra=None):
    sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": sd, "args": vars(args), **(extra or {})},
               os.path.join(CACHE, "ckpt", f"{name}_{tag}.pt"))


@torch.no_grad()
def evaluate(x, y):
    net.eval(); tot = 0.0; cor = 0
    for i in range(0, x.shape[0], 1000):
        with torch.autocast("cuda", torch.bfloat16):
            out = net(x[i:i + 1000].contiguous(memory_format=torch.channels_last))
        tot += F.cross_entropy(out.float(), y[i:i + 1000], reduction="sum").item()
        cor += (out.argmax(1) == y[i:i + 1000]).sum().item()
    net.train()
    return tot / x.shape[0], cor / x.shape[0]


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


save("ep000")
hist = []
t0 = time.time()
for ep in range(1, E + 1):
    net.train()
    perm = torch.randperm(50000, device="cuda")
    tl = 0.0; nb = 0
    for k in range(0, 50000, args.bs):
        idx = perm[k:k + args.bs]
        xb = augment(xtr[idx]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(fwd(xb).float(), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tl += loss.detach(); nb += 1
    sched.step()
    tr = (tl / nb).item()
    te_loss, te_acc = evaluate(xte, yte)
    rec = dict(epoch=ep, train_loss_aug=tr, test_loss=te_loss, test_acc=te_acc,
               lr=opt.param_groups[0]["lr"], time=time.time() - t0)
    hist.append(rec)
    log(json.dumps(rec))
    if ep in save_eps:
        save(f"ep{ep:03d}")
    if not np.isfinite(tr):
        log("DIVERGED"); break

trl, tra = evaluate(xtr, ytr)
te_loss, te_acc = evaluate(xte, yte)
summary = dict(name=name, epochs=E, train_loss=trl, train_acc=tra, test_loss=te_loss,
               test_acc=te_acc, wall_s=time.time() - t0, milestones=milestones, save_eps=save_eps)
log("SUMMARY " + json.dumps(summary))
save("final", {"summary": summary})
json.dump({"summary": summary, "history": hist},
          open(os.path.join(CACHE, "ckpt", f"{name}_history.json"), "w"), indent=1)
