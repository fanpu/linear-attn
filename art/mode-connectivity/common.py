"""Shared compute code for One Basin: data, models, training, weight matching,
interpolation, REPAIR / BN reset, Bezier curves, loss evaluation.

Everything is functional on plain dicts of tensors so interpolation, curves
and 2-D planes are just arithmetic on dicts.
"""
import math, os, time, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = '/home/fzeng/ml/research/art/data'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
DEV = 'cuda'


def gpu_setup():
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cuda.matmul.allow_tf32 = False  # keep float32 exact-ish for small loss diffs
    torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------- data
def load_data(name, flatten=True):
    """Return (Xtr, ytr, Xte, yte) as float32 GPU tensors, standardized."""
    import torchvision
    if name == 'mnist':
        tr = torchvision.datasets.MNIST(ROOT, train=True, download=False)
        te = torchvision.datasets.MNIST(ROOT, train=False, download=False)
        Xtr, ytr = tr.data.float() / 255., tr.targets
        Xte, yte = te.data.float() / 255., te.targets
        Xtr, Xte = Xtr[:, None], Xte[:, None]
    elif name == 'fmnist':
        tr = torchvision.datasets.FashionMNIST(ROOT, train=True, download=False)
        te = torchvision.datasets.FashionMNIST(ROOT, train=False, download=False)
        Xtr, ytr = tr.data.float() / 255., tr.targets
        Xte, yte = te.data.float() / 255., te.targets
        Xtr, Xte = Xtr[:, None], Xte[:, None]
    elif name == 'cifar10':
        tr = torchvision.datasets.CIFAR10(ROOT, train=True, download=False)
        te = torchvision.datasets.CIFAR10(ROOT, train=False, download=False)
        Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float() / 255.
        Xte = torch.tensor(te.data).permute(0, 3, 1, 2).float() / 255.
        ytr, yte = torch.tensor(tr.targets), torch.tensor(te.targets)
    else:
        raise ValueError(name)
    mu = Xtr.mean(dim=(0, 2, 3), keepdim=True)
    sd = Xtr.std(dim=(0, 2, 3), keepdim=True)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if flatten:
        Xtr, Xte = Xtr.flatten(1), Xte.flatten(1)
    return Xtr.to(DEV), ytr.to(DEV), Xte.to(DEV), yte.to(DEV)


# ----------------------------------------------------------------------------- MLP
# params: dict W0,b0,...,W{L},b{L}; hidden layers 0..L-1, output layer L.
N_HIDDEN = 3


def mlp_init(d_in, width, n_out=10, depth=N_HIDDEN, seed=0):
    g = torch.Generator().manual_seed(seed)
    dims = [d_in] + [width] * depth + [n_out]
    p = {}
    for l in range(depth + 1):
        fan_in = dims[l]
        bound = 1 / math.sqrt(fan_in)  # PyTorch nn.Linear default (kaiming_uniform a=sqrt5)
        p[f'W{l}'] = (torch.rand(dims[l + 1], dims[l], generator=g) * 2 - 1) * bound
        p[f'b{l}'] = (torch.rand(dims[l + 1], generator=g) * 2 - 1) * bound
    return {k: v.to(DEV) for k, v in p.items()}


def mlp_depth(p):
    return len([k for k in p if k.startswith('W')]) - 1


def mlp_forward(p, x, return_pre=False):
    L = mlp_depth(p)
    pres = []
    h = x
    for l in range(L):
        z = F.linear(h, p[f'W{l}'], p[f'b{l}'])
        pres.append(z)
        h = F.relu(z)
    out = F.linear(h, p[f'W{L}'], p[f'b{L}'])
    return (out, pres) if return_pre else out


@torch.no_grad()
def mlp_eval_single(p, X, y, bs=20000):
    tot_loss, correct = 0., 0
    for i in range(0, len(X), bs):
        out = mlp_forward(p, X[i:i + bs])
        tot_loss += F.cross_entropy(out.double(), y[i:i + bs], reduction='sum').item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return tot_loss / len(X), correct / len(X)


def stack_params(plist):
    return {k: torch.stack([p[k] for p in plist]) for k in plist[0]}


@torch.no_grad()
def eval_stack(Ws, X, y, max_elems=2.4e8, per_class=False):
    """Evaluate G parameter sets at once (dict of [G,...] tensors) with baddbmm.
    Returns (mean CE loss [G], accuracy [G]) as float64 numpy; loss summed in float64."""
    L = len([k for k in Ws if k.startswith('W')]) - 1
    G = Ws['W0'].shape[0]
    width = max(Ws['W0'].shape[1], X.shape[1])
    nb = int(max(500, min(len(X), max_elems // (G * width))))
    tot = torch.zeros(G, device=X.device, dtype=torch.float64)
    cor = torch.zeros(G, device=X.device, dtype=torch.float64)
    cls = torch.zeros(G, 10, device=X.device, dtype=torch.float64)
    for i in range(0, len(X), nb):
        h = X[i:i + nb][None].expand(G, -1, -1)
        for l in range(L + 1):
            h = torch.baddbmm(Ws[f'b{l}'][:, None, :], h, Ws[f'W{l}'].transpose(1, 2))
            if l < L:
                h = F.relu(h)
        yy = y[i:i + nb]
        ce = F.cross_entropy(h.transpose(1, 2).double(), yy[None].expand(G, -1), reduction='none')
        tot += ce.sum(1)
        cor += (h.argmax(2) == yy[None]).sum(1).double()
        if per_class:
            cls.index_add_(1, yy, ce)
    if per_class:  # per-class loss contribution: sums over classes to the mean loss
        return (tot / len(X)).cpu().numpy(), (cor / len(X)).cpu().numpy(), (cls / len(X)).cpu().numpy()
    return (tot / len(X)).cpu().numpy(), (cor / len(X)).cpu().numpy()


def eval_list(plist, X, y, G=16, per_class=False):
    """Evaluate a list of param dicts in groups of G. Returns (loss[n], acc[n]) (+ per-class [n,10])."""
    R = []
    for s in range(0, len(plist), G):
        R.append(eval_stack(stack_params(plist[s:s + G]), X, y, per_class=per_class))
    return tuple(np.concatenate([r[i] for r in R]) for i in range(len(R[0])))


def mlp_eval(p, X, y):
    l, a = eval_stack({k: v[None] for k, v in p.items()}, X, y)
    return float(l[0]), float(a[0])


def train_mlp(data, width, seed, epochs=30, bs=512, lr=1e-3, ckpt_epochs=None, log=None):
    """Adam lr 1e-3 (Git Re-Basin A.3.1). Different seed => different init AND batch order.
    ckpt_epochs: list of (possibly fractional) epochs at which to snapshot weights."""
    Xtr, ytr, Xte, yte = data
    p = mlp_init(Xtr.shape[1], width, seed=seed)
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(p.values(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    n = len(Xtr)
    steps_per_epoch = n // bs
    ckpts = {}
    ck_steps = {}
    if ckpt_epochs is not None:
        for e in ckpt_epochs:
            ck_steps[int(round(e * steps_per_epoch))] = e
    step = 0
    if 0 in ck_steps:
        ckpts[ck_steps[0]] = {k: v.detach().clone() for k, v in p.items()}
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV, generator=g)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(mlp_forward(p, Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step in ck_steps:
                ckpts[ck_steps[step]] = {k: v.detach().clone() for k, v in p.items()}
    p = {k: v.detach() for k, v in p.items()}
    if log:
        trl, tra = mlp_eval(p, Xtr, ytr)
        tel, tea = mlp_eval(p, Xte, yte)
        log(f'  width {width} seed {seed}: train loss {trl:.4f} acc {tra:.4f} | test loss {tel:.4f} acc {tea:.4f}')
    return p, ckpts


# ----------------------------------------------------------------------------- weight matching (MLP)
def mlp_apply_perm(p, perms):
    """perms[l] : index array, new unit i of layer l is old unit perms[l][i]."""
    L = mlp_depth(p)
    q = {}
    for l in range(L + 1):
        W = p[f'W{l}']
        b = p[f'b{l}']
        if l < L:
            idx = torch.as_tensor(perms[l], device=W.device)
            W = W[idx]
            b = b[idx]
        if l > 0:
            idx_prev = torch.as_tensor(perms[l - 1], device=W.device)
            W = W[:, idx_prev]
        q[f'W{l}'] = W
        q[f'b{l}'] = b
    return q


def mlp_weight_matching(pA, pB, max_iter=100, seed=0, verbose=False, history=None):
    """Git Re-Basin Algorithm 1 (weight matching), coordinate descent over hidden layers.
    Each step solves one linear assignment problem exactly (Hungarian / LAPJV via scipy)."""
    L = mlp_depth(pA)
    A = {k: v.double().cpu().numpy() for k, v in pA.items()}
    B = {k: v.double().cpu().numpy() for k, v in pB.items()}
    widths = [A[f'b{l}'].shape[0] for l in range(L)]
    perms = [np.arange(w) for w in widths]
    rng = np.random.default_rng(seed)
    obj_hist = []
    for it in range(max_iter):
        improved = False
        for l in rng.permutation(L):
            # rows of W_l / b_l (all columns permuted by perm_{l-1}), and columns of W_{l+1}
            WA, WB = A[f'W{l}'], B[f'W{l}']
            if l > 0:
                WB = WB[:, perms[l - 1]]
            C = WA @ WB.T + np.outer(A[f'b{l}'], B[f'b{l}'])
            WA2, WB2 = A[f'W{l + 1}'], B[f'W{l + 1}']
            if l + 1 < L:
                WB2 = WB2[perms[l + 1]]
            C += WA2.T @ WB2
            old = C[np.arange(widths[l]), perms[l]].sum()
            _, col = linear_sum_assignment(C, maximize=True)
            new = C[np.arange(widths[l]), col].sum()
            if new > old + 1e-12 * abs(old) + 1e-9:
                improved = True
                perms[l] = col
            obj_hist.append(float(max(new, old)))
        if history is not None:
            history.append(([np.array(q) for q in perms], obj_hist[-1]))
        if verbose:
            print(f'    WM iter {it}: obj {obj_hist[-1]:.4f}')
        if not improved:
            break
    return perms, it + 1


def perm_objective(pA, pB):
    return sum((pA[k].double() * pB[k].double()).sum().item() for k in pA)


# ----------------------------------------------------------------------------- interpolation helpers
def lerp(pA, pB, lam):
    return {k: (1 - lam) * pA[k] + lam * pB[k] for k in pA}


def bezier(pA, pC, pB, t):
    return {k: (1 - t) ** 2 * pA[k] + 2 * t * (1 - t) * pC[k] + t ** 2 * pB[k] for k in pA}


def barrier(losses, lams):
    """Git Re-Basin barrier: max_l L(l) - 0.5 (L(0)+L(1)).  We also report the
    'linear-baseline' version max_l L(l) - [(1-l)L0 + l L1] (Frankle et al. / Entezari)."""
    losses = np.asarray(losses)
    lams = np.asarray(lams)
    b_mid = losses.max() - 0.5 * (losses[0] + losses[-1])
    b_lin = (losses - ((1 - lams) * losses[0] + lams * losses[-1])).max()
    return float(b_mid), float(b_lin)


# ----------------------------------------------------------------------------- REPAIR for MLP
@torch.no_grad()
def mlp_repair(pA, pB, lam, X):
    """REPAIR (Jordan et al. 2022) for a plain MLP: rescale each hidden unit of the
    interpolated net so its pre-activation mean/std equal the interpolation of the
    endpoints' mean/std.  Absorbed into W,b so the result is still a plain MLP."""
    _, preA = mlp_forward(pA, X, return_pre=True)
    _, preB = mlp_forward(pB, X, return_pre=True)
    p = lerp(pA, pB, lam)
    p = {k: v.clone() for k, v in p.items()}
    L = mlp_depth(p)
    h = X
    for l in range(L):
        z = F.linear(h, p[f'W{l}'], p[f'b{l}'])
        mu_t = (1 - lam) * preA[l].mean(0) + lam * preB[l].mean(0)
        sd_t = (1 - lam) * preA[l].std(0) + lam * preB[l].std(0)
        mu_m, sd_m = z.mean(0), z.std(0) + 1e-8
        s = sd_t / sd_m
        p[f'W{l}'] = p[f'W{l}'] * s[:, None]
        p[f'b{l}'] = (p[f'b{l}'] - mu_m) * s + mu_t
        h = F.relu(F.linear(h, p[f'W{l}'], p[f'b{l}']))
    return p


# ----------------------------------------------------------------------------- Bezier curve training (MLP)
def train_bezier_mlp(data, pA, pB, epochs=20, bs=512, lr=1e-3, seed=0, log=None):
    """Garipov et al. 2018 quadratic Bezier: theta(t)=(1-t)^2 A + 2t(1-t) C + t^2 B,
    endpoints fixed, control point C trained by sampling t~U(0,1) per minibatch."""
    Xtr, ytr, _, _ = data
    C = {k: (0.5 * (pA[k] + pB[k])).clone().requires_grad_(True) for k in pA}
    opt = torch.optim.Adam(C.values(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(777 + seed)
    n = len(Xtr)
    spe = n // bs
    total = epochs * spe
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV, generator=g)
        for i in range(spe):
            for gr in opt.param_groups:
                gr['lr'] = lr * 0.5 * (1 + math.cos(math.pi * step / total))
            idx = perm[i * bs:(i + 1) * bs]
            t = torch.rand((), device=DEV, generator=g)
            q = bezier(pA, C, pB, t)
            loss = F.cross_entropy(mlp_forward(q, Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    C = {k: v.detach() for k, v in C.items()}
    return C


def to_cpu(p):
    return {k: v.detach().cpu() for k, v in p.items()}


def to_dev(p):
    return {k: v.to(DEV) for k, v in p.items()}


class Logger:
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(time.strftime('%H:%M:%S ') + s + '\n')
        self.f.flush()
