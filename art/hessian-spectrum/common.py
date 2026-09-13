"""Shared pieces: data, models, Hessian / Gauss-Newton vector products, Lanczos, SLQ,
and Papyan's class / cross-class decomposition of the Gauss-Newton term.

Everything spectral runs in float64.
"""
import math, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp, grad
import torchvision

ROOT = '/home/fzeng/ml/research/art/data'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')

# ----------------------------------------------------------------------------- data
_STATS = {'MNIST': (0.1307, 0.3081), 'FashionMNIST': (0.2860, 0.3530),
          'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}


def load_dataset(name, train=True, device='cpu', down=None):
    """down: optional side length for average-pool downsampling (e.g. 10 -> 10x10 MNIST)."""
    ds = getattr(torchvision.datasets, name)(root=ROOT, train=train, download=False)
    if name == 'CIFAR10':
        x = torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255.
        m, s = _STATS[name]
        x = (x - torch.tensor(m)[None, :, None, None]) / torch.tensor(s)[None, :, None, None]
        y = torch.tensor(ds.targets)
    else:
        x = ds.data.float()[:, None] / 255.
        m, s = _STATS[name]
        x = (x - m) / s
        y = ds.targets.clone()
    if down:
        x = F.adaptive_avg_pool2d(x, down)
    return x.to(device), y.to(device)


def class_subset(x, y, C, per_class=None, seed=0):
    """Keep classes 0..C-1; optionally a fixed, balanced number per class (deterministic)."""
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in range(C):
        ic = torch.nonzero(y == c).flatten().cpu()
        if per_class is not None:
            ic = ic[torch.randperm(len(ic), generator=g)[:per_class]]
            ic = ic.sort().values
        idx.append(ic)
    idx = torch.cat(idx).to(x.device)
    return x[idx], y[idx]


# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d_in, C, width=128, depth=2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, C)]
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """No batch-norm, so the Hessian is that of a plain function of the weights."""
    def __init__(self, in_ch, C, img=28, chans=(16, 32), fc=64):
        super().__init__()
        layers, c, s = [], in_ch, img
        for co in chans:
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c, s = co, s // 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * s * s, fc), nn.ReLU(), nn.Linear(fc, C))

    def forward(self, x):
        return self.head(self.features(x))


ARCHS = {
    'mlp': lambda ds, C: MLP(784 if ds != 'CIFAR10' else 3072, C, width=128, depth=2),
    # small MLP on 10x10 average-pooled images: P ~ 4.4k, so the FULL Hessian can be formed and diagonalised
    'mlps': lambda ds, C: MLP(100 if ds != 'CIFAR10' else 300, C, width=32, depth=2),
    'cnn': lambda ds, C: (SmallCNN(1, C, 28, (16, 32), 64) if ds != 'CIFAR10'
                          else SmallCNN(3, C, 32, (32, 64, 64), 128)),
}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- flat functional wrapper
class Flat:
    """Wrap a model as f(theta_flat, x) -> logits, for torch.func transforms."""
    def __init__(self, model):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.sizes = [p.numel() for _, p in model.named_parameters()]
        self.P = sum(self.sizes)

    def theta(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def unflat(self, th):
        return {n: t.view(s) for n, t, s in zip(self.names, th.split(self.sizes), self.shapes)}

    def f(self, th, x):
        return functional_call(self.model, self.unflat(th), (x,))


class HessianOps:
    """Operators on a fixed dataset (x, y), loss = mean cross-entropy over it.

    H v  : exact Hessian-vector product (forward-over-reverse)
    G v  : Gauss-Newton  J^T (diag p - p p^T) J v / N
    """
    def __init__(self, model, x, y, chunk=2500, dtype=torch.float64):
        self.model = model.to(dtype)
        self.fl = Flat(self.model)
        self.x = x.to(dtype)
        self.y = y
        self.N = len(y)
        self.chunk = chunk
        self.th = self.fl.theta()
        self.P = self.fl.P
        self.dtype = dtype

    def _chunks(self):
        for s in range(0, self.N, self.chunk):
            yield self.x[s:s + self.chunk], self.y[s:s + self.chunk]

    def loss_acc(self):
        L, A = 0., 0.
        with torch.no_grad():
            for xb, yb in self._chunks():
                out = self.fl.f(self.th, xb)
                L += F.cross_entropy(out, yb, reduction='sum').item()
                A += (out.argmax(1) == yb).sum().item()
        return L / self.N, A / self.N

    def Hv(self, v):
        out = torch.zeros_like(v)
        for xb, yb in self._chunks():
            lf = lambda th: F.cross_entropy(self.fl.f(th, xb), yb, reduction='sum')
            _, hv = jvp(grad(lf), (self.th,), (v,))
            out += hv
        return out / self.N

    def Gv(self, v):
        out = torch.zeros_like(v)
        for xb, yb in self._chunks():
            fx = lambda th: self.fl.f(th, xb)
            z, Jv = jvp(fx, (self.th,), (v,))
            p = torch.softmax(z, 1)
            u = p * Jv - p * (p * Jv).sum(1, keepdim=True)
            _, vjp_fn = vjp(fx, self.th)
            out += vjp_fn(u)[0]
        return out / self.N

    def Hv_batched(self, V):
        """(B,P) -> (B,P) exact Hessian columns, vmapped forward-over-reverse."""
        from torch.func import vmap
        out = torch.zeros_like(V)
        for xb, yb in self._chunks():
            lf = lambda th: F.cross_entropy(self.fl.f(th, xb), yb, reduction='sum')
            g = grad(lf)
            out += vmap(lambda v: jvp(g, (self.th,), (v,))[1])(V)
        return out / self.N

    def Gv_batched(self, V):
        from torch.func import vmap
        out = torch.zeros_like(V)
        for xb, yb in self._chunks():
            fx = lambda th: self.fl.f(th, xb)
            z, vjp_fn = vjp(fx, self.th)
            p = torch.softmax(z.detach(), 1)
            def one(v):
                Jv = jvp(fx, (self.th,), (v,))[1]
                u = p * Jv - p * (p * Jv).sum(1, keepdim=True)
                return vjp_fn(u)[0]
            out += vmap(one)(V)
        return out / self.N

    def Ev(self, v):
        return self.Hv(v) - self.Gv(v)

    # ---- Papyan (2019) class / cross-class means of the logit derivatives
    def class_means(self, C):
        """delta_{c,c'} = Ave_{i in c} sqrt(p_{ic'}) J_i^T (e_{c'} - p_i).   Returns (C, C, P)."""
        D = torch.zeros(C, C, self.P, dtype=self.dtype, device=self.th.device)
        for c in range(C):
            xc = self.x[self.y == c]
            Nc = len(xc)
            for s in range(0, Nc, self.chunk):
                xb = xc[s:s + self.chunk]
                fx = lambda th: self.fl.f(th, xb)
                z, vjp_fn = vjp(fx, self.th)
                p = torch.softmax(z.detach(), 1)
                for cp in range(C):
                    e = torch.zeros_like(p); e[:, cp] = 1.
                    u = p[:, cp:cp + 1].sqrt() * (e - p) / Nc
                    D[c, cp] += vjp_fn(u)[0]
        return D


def papyan_decomposition(D):
    """Eigenvalues of G0, G1, G1+2 via Gram matrices (balanced classes).

    G1+2 = (1/C) sum_{c,c'} d_cc' d_cc'^T
    G0   = Ave_c d_cc d_cc^T
    G1   = (C-1) Ave_c d_c d_c^T,  d_c = Ave_{c' != c} d_cc'
    """
    C = D.shape[0]
    flat = D.reshape(C * C, -1)
    g12 = torch.linalg.eigvalsh(flat @ flat.T / C).flip(0)
    dd = torch.stack([D[c, c] for c in range(C)])
    g0 = torch.linalg.eigvalsh(dd @ dd.T / C).flip(0)
    mask = ~torch.eye(C, dtype=torch.bool, device=D.device)
    dc = torch.stack([D[c][mask[c]].mean(0) for c in range(C)])
    g1 = torch.linalg.eigvalsh((C - 1) * dc @ dc.T / C).flip(0)
    # cosine similarity between class centers (structure of the C outliers)
    dcn = dc / dc.norm(dim=1, keepdim=True)
    cos = dcn @ dcn.T
    return dict(G12=g12.cpu().numpy(), G0=g0.cpu().numpy(), G1=g1.cpu().numpy(),
                center_cos=cos.cpu().numpy(), center_norm=dc.norm(dim=1).cpu().numpy())


# ----------------------------------------------------------------------------- Lanczos
@torch.no_grad()
def lanczos(mv, P, m, device, dtype=torch.float64, seed=0, v0=None, reorth=True, return_V=False):
    """m-step Lanczos with full re-orthogonalisation (twice), float64.

    Returns (alpha, beta, first components needed for SLQ are implicit: start vector e1).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    if v0 is None:
        v0 = (torch.randint(0, 2, (P,), generator=g).to(dtype) * 2 - 1)
    v = (v0 / v0.norm()).to(device)
    V = torch.empty(m, P, dtype=dtype, device=device)
    alpha = torch.zeros(m, dtype=dtype)
    beta = torch.zeros(m, dtype=dtype)
    V[0] = v
    k_used = m
    for j in range(m):
        w = mv(V[j])
        a = torch.dot(w, V[j])
        alpha[j] = a
        if reorth:
            for _ in range(2):
                w -= V[:j + 1].T @ (V[:j + 1] @ w)
        else:
            w -= a * V[j]
            if j > 0:
                w -= beta[j - 1] * V[j - 1]
        b = w.norm()
        beta[j] = b
        if j + 1 < m:
            if b < 1e-12:
                k_used = j + 1
                break
            V[j + 1] = w / b
    alpha, beta = alpha[:k_used].double(), beta[:k_used].double()
    if return_V:
        return alpha.numpy(), beta.numpy(), V[:k_used]
    return alpha.numpy(), beta.numpy()


def tridiag_eig(alpha, beta):
    import scipy.linalg
    m = len(alpha)
    theta, S = scipy.linalg.eigh_tridiagonal(alpha, beta[:m - 1])
    tau2 = S[0] ** 2                       # SLQ quadrature weights
    resid = np.abs(beta[m - 1] * S[m - 1])  # Ritz residual bound |beta_m s_{m,j}|
    return theta, tau2, resid


def slq(mv, P, m, nv, device, seed=0):
    nodes, weights = [], []
    for k in range(nv):
        a, b = lanczos(mv, P, m, device, seed=seed + 1000 + k)
        th, t2, _ = tridiag_eig(a, b)
        nodes.append(th); weights.append(t2)
    return nodes, weights


def top_eigs(mv, P, m, device, seed=0, k_vec=0):
    """Extreme Ritz values (ascending) + residual bounds; optionally the top-k_vec Ritz vectors (k, P)."""
    if k_vec:
        a, b, V = lanczos(mv, P, m, device, seed=seed, return_V=True)
        import scipy.linalg
        th, S = scipy.linalg.eigh_tridiagonal(a, b[:len(a) - 1])
        res = np.abs(b[len(a) - 1] * S[len(a) - 1])
        Sk = torch.tensor(S[:, ::-1][:, :k_vec].copy(), dtype=V.dtype, device=V.device)
        return th, res, (V.T @ Sk).T
    a, b = lanczos(mv, P, m, device, seed=seed)
    th, _, res = tridiag_eig(a, b)
    return th, res   # ascending; both ends usable, check residuals


# ----------------------------------------------------------------------------- exact (small nets)
def full_matrix(mv_batched, P, block=256, dtype=torch.float64):
    """Materialise a symmetric operator column-block by column-block. mv_batched: (B,P)->(B,P)."""
    M = torch.empty(P, P, dtype=dtype)
    for s in range(0, P, block):
        E = torch.zeros(min(block, P - s), P, dtype=dtype)
        E[torch.arange(E.shape[0]), torch.arange(s, s + E.shape[0])] = 1.
        M[s:s + E.shape[0]] = mv_batched(E)
    return 0.5 * (M + M.T)


# ----------------------------------------------------------------------------- outlier counting
def count_outliers(lam, kmax):
    """Largest multiplicative gap among the top kmax+1 positive eigenvalues (descending).

    Returns (k, gap_ratio, second_best_k, second_gap_ratio): k eigenvalues sit above the widest log-gap."""
    lam = np.sort(np.asarray(lam))[::-1]
    lam = lam[lam > 0][:kmax + 1]
    r = lam[:-1] / lam[1:]
    o = np.argsort(r)[::-1]
    return int(o[0] + 1), float(r[o[0]]), int(o[1] + 1), float(r[o[1]])
