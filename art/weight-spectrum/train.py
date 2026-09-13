"""Train an MLP (or MiniAlexNet) and record weight-matrix spectra at log-spaced checkpoints.

Compute step only: writes raw measurements to cache/<run>.npz (+ cache/<run>_W/ for full matrices).
Nothing here renders images.

For each tracked dense layer W (PyTorch shape out x in) we orient it as N x M with N >= M,
Q = N/M, and compute the ESD of X = W^T W / N, i.e. eigenvalues lambda_i = s_i^2 / N where s_i
are the singular values of W (float64 SVD).
"""
import argparse, json, math, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = '/home/fzeng/ml/research/art/data'
HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(name, device):
    if name == 'cifar10':
        d = os.path.join(ROOT, 'cifar-10-batches-py')
        xs, ys = [], []
        for i in range(1, 6):
            with open(os.path.join(d, f'data_batch_{i}'), 'rb') as f:
                b = pickle.load(f, encoding='latin1')
            xs.append(b['data']); ys += b['labels']
        xtr = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255
        ytr = np.array(ys)
        with open(os.path.join(d, 'test_batch'), 'rb') as f:
            b = pickle.load(f, encoding='latin1')
        xte = b['data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255
        yte = np.array(b['labels'])
    elif name == 'fmnist':
        d = os.path.join(ROOT, 'FashionMNIST', 'raw')
        def img(fn):
            with open(os.path.join(d, fn), 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255
        def lab(fn):
            with open(os.path.join(d, fn), 'rb') as f:
                return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
        xtr, ytr = img('train-images-idx3-ubyte'), lab('train-labels-idx1-ubyte')
        xte, yte = img('t10k-images-idx3-ubyte'), lab('t10k-labels-idx1-ubyte')
    else:
        raise ValueError(name)
    mean = xtr.mean(axis=(0, 2, 3), keepdims=True); std = xtr.std(axis=(0, 2, 3), keepdims=True)
    xtr = (xtr - mean) / std; xte = (xte - mean) / std
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return t(xtr, torch.float32), t(ytr, torch.long), t(xte, torch.float32), t(yte, torch.long)


class MLP(nn.Module):
    def __init__(self, d_in, widths, n_out=10):
        super().__init__()
        dims = [d_in] + list(widths) + [n_out]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.tracked = {f'FC{i+1}': l for i, l in enumerate(self.layers)}

    def forward(self, x):
        x = x.flatten(1)
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)


class MiniAlexNet(nn.Module):
    """Martin & Mahoney's MiniAlexNet: 2 conv(+BN+maxpool), FC1 4096->384, FC2 384->192, FC3 192->10."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 96, 5, padding=2); self.b1 = nn.BatchNorm2d(96)
        self.c2 = nn.Conv2d(96, 256, 5, padding=2); self.b2 = nn.BatchNorm2d(256)
        # 32 -> pool3/s2 -> 15 -> pool3/s2 -> 7 ; 256*7*7 = 12544. Use adaptive pool to 4x4 -> 4096 (matches FC1 input 4096)
        self.fc1 = nn.Linear(4096, 384); self.fc2 = nn.Linear(384, 192); self.fc3 = nn.Linear(192, 10)
        self.tracked = {'FC1': self.fc1, 'FC2': self.fc2, 'FC3': self.fc3}

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 3, 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 3, 2)
        x = F.adaptive_avg_pool2d(x, 4).flatten(1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)


def glorot_normal_(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)


def ckpt_steps(total, n):
    s = np.unique(np.round(np.logspace(0, math.log10(total), n)).astype(int))
    return np.concatenate([[0], s]).tolist()


@torch.no_grad()
def spectrum(W, k_vec):
    """W: out x in torch tensor. Returns dict with eigenvalues of X=W^T W/N (descending) etc."""
    W64 = W.detach().double()
    out_dim, in_dim = W64.shape
    N, M = max(out_dim, in_dim), min(out_dim, in_dim)
    U, S, Vh = torch.linalg.svd(W64, full_matrices=False)  # U: out x r, Vh: r x in, r = M
    lam = (S ** 2 / N)
    ipr_u = (U ** 4).sum(0)  # IPR of left singular vectors (in out-space)
    ipr_v = (Vh ** 4).sum(1)  # IPR of right singular vectors (in in-space)
    # null model (Martin & Mahoney): same entries, randomly permuted -> destroys correlations, keeps marginals
    g = torch.Generator(device=W64.device); g.manual_seed(1234)
    flat = W64.flatten()
    Wsh = flat[torch.randperm(flat.numel(), device=W64.device, generator=g)].view_as(W64)
    lam_sh = torch.linalg.svdvals(Wsh) ** 2 / N
    return dict(
        lam=lam.cpu().numpy(), lam_shuf=lam_sh.cpu().numpy(), ipr_out=ipr_u.float().cpu().numpy(), ipr_in=ipr_v.float().cpu().numpy(),
        U_top=U[:, :k_vec].float().cpu().numpy(), V_top=Vh[:k_vec].float().cpu().numpy(),
        elem_var=float(W64.var()), N=N, M=M,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', default='cifar10')
    ap.add_argument('--arch', default='mlp')
    ap.add_argument('--widths', default='2048,1024,512')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--epochs', type=float, default=30)
    ap.add_argument('--n_ckpt', type=int, default=80)
    ap.add_argument('--n_full', type=int, default=0, help='save full W at this many log-spaced checkpoints')
    ap.add_argument('--crop', type=int, default=0, help='save top-left crop x crop of W at every checkpoint')
    ap.add_argument('--k_vec', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mem', type=float, default=0.10)
    ap.add_argument('--compile', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    torch.set_num_threads(args.threads)
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.mem)
    else:
        args.compile = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xtr, ytr, xte, yte = load_data(args.data, dev)
    ntr = xtr.shape[0]
    if args.arch == 'mlp':
        model = MLP(xtr[0].numel(), [int(w) for w in args.widths.split(',')])
    else:
        model = MiniAlexNet(xtr.shape[1])
    glorot_normal_(model)
    model.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    steps_per_epoch = ntr // args.bs
    total = int(round(args.epochs * steps_per_epoch))
    cks = ckpt_steps(total, args.n_ckpt)
    full_set = set(ckpt_steps(total, args.n_full)) if args.n_full else set()
    full_set |= {0, cks[-1]}
    print(f'[{args.name}] total steps {total}, {len(cks)} checkpoints', flush=True)

    def train_step(xb, yb):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        return loss

    step_fn = torch.compile(train_step, mode='max-autotune-no-cudagraphs') if args.compile else train_step

    rec = {k: [] for k in ['step', 'epoch', 'train_loss', 'train_acc', 'test_acc', 'time']}
    layers = {n: {'lam': [], 'lam_shuf': [], 'ipr_out': [], 'ipr_in': [], 'U_top': [], 'V_top': [], 'elem_var': [], 'crop': []}
              for n in model.tracked}
    shapes = {}
    wdir = os.path.join(HERE, 'cache', f'{args.name}_W'); os.makedirs(wdir, exist_ok=True)

    @torch.no_grad()
    def evaluate(x, y, n=10000):
        model.eval()
        correct, loss = 0, 0.0
        for i in range(0, min(n, x.shape[0]), 2000):
            o = model(x[i:i + 2000]); loss += F.cross_entropy(o, y[i:i + 2000], reduction='sum').item()
            correct += (o.argmax(1) == y[i:i + 2000]).sum().item()
        model.train()
        m = min(n, x.shape[0]); return loss / m, correct / m

    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    sample_every = max(1, steps_per_epoch // 20)
    fine = {'step': [], 'loss': []}
    perm = torch.randperm(ntr, device=dev, generator=g); pos = 0
    t0 = time.time(); ci = 0; step = 0
    running = float('nan')
    model.train()
    while True:
        if ci < len(cks) and step == cks[ci]:
            trl, tra = evaluate(xtr, ytr); _, tea = evaluate(xte, yte)
            for k, v in zip(rec, [step, step / steps_per_epoch, trl, tra, tea, time.time() - t0]):
                rec[k].append(v)
            full = {}
            for n, l in model.tracked.items():
                sp = spectrum(l.weight, args.k_vec)
                shapes[n] = (sp['N'], sp['M'], tuple(l.weight.shape))
                for k in ['lam', 'lam_shuf', 'ipr_out', 'ipr_in', 'U_top', 'V_top', 'elem_var']:
                    layers[n][k].append(sp[k])
                if args.crop:
                    layers[n]['crop'].append(l.weight[:args.crop, :args.crop].detach().float().cpu().numpy())
                if step in full_set:
                    full[n] = l.weight.detach().float().cpu().numpy()
            if full:
                np.savez(os.path.join(wdir, f'step{step:08d}.npz'), **full)
            lam1 = layers['FC1']['lam'][-1]
            print(f'[{args.name}] step {step} ep {step/steps_per_epoch:.2f} loss {trl:.4f} tr {tra:.3f} te {tea:.3f} '
                  f'FC1 lmax {lam1[0]:.3f} t {time.time()-t0:.0f}s', flush=True)
            ci += 1
        if step >= total:
            break
        if pos + args.bs > ntr:
            perm = torch.randperm(ntr, device=dev, generator=g); pos = 0
        idx = perm[pos:pos + args.bs]; pos += args.bs
        loss = step_fn(xtr[idx], ytr[idx])
        step += 1
        if step % sample_every == 0:
            fine['step'].append(step); fine['loss'].append(loss.item())
        if step % 2000 == 0 and not math.isfinite(loss.item()):
            print(f'[{args.name}] diverged at step {step}', flush=True); break

    out = dict(meta=json.dumps(dict(vars(args), shapes=shapes, steps_per_epoch=steps_per_epoch, total=total,
                                    wall=time.time() - t0)))
    for k, v in rec.items():
        out[k] = np.array(v)
    out['fine_step'] = np.array(fine['step']); out['fine_loss'] = np.array(fine['loss'])
    for n, d in layers.items():
        for k, v in d.items():
            if len(v):
                out[f'{n}/{k}'] = np.stack(v)
    np.savez_compressed(os.path.join(HERE, 'cache', f'{args.name}.npz'), **out)
    print(f'[{args.name}] done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
