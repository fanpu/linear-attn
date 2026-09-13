"""Train a small classifier and dump checkpoints at log-spaced steps.

python train.py --ds MNIST --arch mlp --C 10 --epochs 20 --lr 0.02 --name mnist_mlp
"""
import argparse, json, os, time, math
import numpy as np
import torch
import torch.nn.functional as F
from common import load_dataset, class_subset, ARCHS, n_params, CACHE

ap = argparse.ArgumentParser()
ap.add_argument('--ds', required=True)
ap.add_argument('--arch', required=True)
ap.add_argument('--C', type=int, default=10)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--lr', type=float, default=0.02)
ap.add_argument('--wd', type=float, default=5e-4)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_ckpt', type=int, default=6, help='number of log-spaced checkpoints (plus step 0 and final)')
ap.add_argument('--name', required=True)
ap.add_argument('--down', type=int, default=0, help='avg-pool images to down x down (0 = off)')
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
out = os.path.join(CACHE, 'runs', args.name)
os.makedirs(out, exist_ok=True)

x, y = load_dataset(args.ds, train=True, down=args.down or None)
x, y = class_subset(x, y, args.C)
xt, yt = load_dataset(args.ds, train=False, down=args.down or None)
xt, yt = class_subset(xt, yt, args.C)
N = len(y)
model = ARCHS[args.arch](args.ds, args.C)
P = n_params(model)
steps_per_epoch = N // args.bs
total = steps_per_epoch * args.epochs
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
# step-anneal by 10x at 1/2 and 3/4 of training (Papyan-style annealing, shortened)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, [total // 2, 3 * total // 4], 0.1)
ck = sorted(set([0, total] + [int(round(v)) for v in np.geomspace(10, total, args.n_ckpt)]))
print(f'{args.name}: N={N} P={P} steps={total} ckpts={ck}', flush=True)


def evaluate(xx, yy):
    model.eval(); L = A = 0.
    with torch.no_grad():
        for s in range(0, len(yy), 5000):
            o = model(xx[s:s + 5000])
            L += F.cross_entropy(o, yy[s:s + 5000], reduction='sum').item()
            A += (o.argmax(1) == yy[s:s + 5000]).sum().item()
    model.train()
    return L / len(yy), A / len(yy)


log = []
g = torch.Generator(device='cpu').manual_seed(args.seed)
step, t0 = 0, time.time()
torch.save(model.state_dict(), os.path.join(out, f'step_{0:06d}.pt'))
while step < total:
    perm = torch.randperm(N, device='cpu', generator=g)
    for b in range(steps_per_epoch):
        idx = perm[b * args.bs:(b + 1) * args.bs]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        step += 1
        if step in ck:
            torch.save(model.state_dict(), os.path.join(out, f'step_{step:06d}.pt'))
            trl, tra = evaluate(x, y); tel, tea = evaluate(xt, yt)
            log.append(dict(step=step, train_loss=trl, train_acc=tra, test_loss=tel, test_acc=tea))
            print(f'  step {step:6d} train {trl:.4f}/{tra:.4f} test {tel:.4f}/{tea:.4f}  {time.time()-t0:.0f}s', flush=True)
        if step >= total:
            break
json.dump(dict(vars(args), N=N, P=P, total_steps=total, ckpts=ck, log=log,
               wall=time.time() - t0), open(os.path.join(out, 'meta.json'), 'w'), indent=1)
