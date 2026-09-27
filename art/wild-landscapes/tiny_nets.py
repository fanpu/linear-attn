"""Tiny nets, whole weight space.  Exact full-batch loss (every input of the task), deterministic.

Net: f(x) = w2 . tanh(W1 x + b1) + b2,  output o = tanh(f),  targets y in {-1, +1},
L = mean_x (o(x) - y(x))^2  over ALL 2^k inputs x in {-1,+1}^k.  Each wrong sign costs up to 4,
so where the hidden units saturate the loss sits on terraces at 0, 1, 2, 3, 4 (XOR) = the number
of inputs the net gets wrong (x 4/2^k).  Colour: which Boolean function sign(f) computes
(2^(2^k) possible labels).

    python tiny_nets.py --task xor --hidden 2 --res 4096 --span 6 --plane sym  --out cache/tiny/xor_sym.npz
Planes:
  random  theta* + a d1 + b d2, d Gaussian, each hidden unit's incoming (w, b) row and the output
          layer rescaled to the norm of the same piece of theta* (filter normalisation).
  sym     the plane through theta*, P theta* (the two hidden units swapped) and S theta* (hidden unit 0
          sign-flipped): three copies of one solution related by symmetry; coordinates are
          centred on their centroid, axes orthonormalised (Gram-Schmidt), unit = |theta* - centroid|.
  origin  a random plane through the ORIGIN, d normalised to |theta*| (the whole weight space).
"""
import argparse, itertools, json, math, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def task(name):
    if name == "xor":
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], float); y = -X[:, 0] * X[:, 1]
    elif name.startswith("parity"):
        k = int(name[6:]); X = np.array(list(itertools.product([-1, 1], repeat=k)), float); y = X.prod(1)
    return torch.tensor(X), torch.tensor(y)


def unpack(th, k, h):
    """th (..., D) -> W1 (..., h, k), b1 (..., h), w2 (..., h), b2 (...)"""
    i = 0
    W1 = th[..., i:i + h * k].reshape(*th.shape[:-1], h, k); i += h * k
    b1 = th[..., i:i + h]; i += h
    w2 = th[..., i:i + h]; i += h
    return W1, b1, w2, th[..., i]


def fwd(th, X, k, h):
    W1, b1, w2, b2 = unpack(th, k, h)
    z = torch.einsum("...hk,nk->...nh", W1, X) + b1[..., None, :]
    return torch.einsum("...nh,...h->...n", torch.tanh(z), w2) + b2[..., None]


def train(tname, h, seed):
    X, y = task(tname); k = X.shape[1]
    g = torch.Generator().manual_seed(seed)
    D = h * k + 2 * h + 1
    th = (torch.randn(D, generator=g, dtype=torch.float64) * 0.8).requires_grad_()
    opt = torch.optim.Adam([th], lr=0.03)
    for s in range(4000):
        L = ((torch.tanh(fwd(th, X, k, h)) - y) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
    return th.detach(), float(L)


def sym_copies(th, k, h):
    W1, b1, w2, b2 = unpack(th, k, h)
    perm = list(range(h)); perm[0], perm[1] = 1, 0
    P = torch.cat([W1[perm].flatten(), b1[perm], w2[perm], b2[None]])
    S1 = W1.clone(); S1[0] *= -1; sb = b1.clone(); sb[0] *= -1; sw = w2.clone(); sw[0] *= -1
    S = torch.cat([S1.flatten(), sb, sw, b2[None]])
    return P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="xor")
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plane", default="sym")
    ap.add_argument("--dseed", type=int, default=1)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--span", type=float, default=3.0)
    ap.add_argument("--centre", default="0,0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    X, y = task(a.task); k, h = X.shape[1], a.hidden
    th, L0 = train(a.task, h, a.seed)
    D = th.numel()
    g = torch.Generator().manual_seed(a.dseed)
    if a.plane == "sym":
        P, S = sym_copies(th, k, h)
        c = (th + P + S) / 3
        e1 = th - c; unit = e1.norm(); e1 = e1 / unit
        e2 = P - c; e2 = e2 - (e2 @ e1) * e1; e2 = e2 / e2.norm()
        origin, d1, d2 = c, e1 * unit, e2 * unit
        marks = [((v - c) @ e1 / unit, (v - c) @ e2 / unit) for v in (th, P, S)]
    else:
        def rnd():
            d = torch.randn(D, generator=g, dtype=torch.float64)
            if a.plane == "origin":
                return d / d.norm() * th.norm()
            W1, b1, w2, b2 = unpack(d, k, h); tW1, tb1, tw2, tb2 = unpack(th, k, h)
            r = torch.cat([W1, b1[:, None]], 1); tr = torch.cat([tW1, tb1[:, None]], 1)
            r = r * (tr.norm(dim=1) / r.norm(dim=1))[:, None]
            o = torch.cat([w2, b2[None]]); to = torch.cat([tw2, tb2[None]])
            o = o * to.norm() / o.norm()
            return torch.cat([r[:, :k].flatten(), r[:, k], o[:h], o[h:]])
        d1, d2 = rnd(), rnd()
        origin = torch.zeros(D, dtype=torch.float64) if a.plane == "origin" else th
        marks = []
    ca, cb = (float(v) for v in a.centre.split(","))
    ax = np.linspace(-a.span, a.span, a.res)
    Lg = np.zeros((a.res, a.res)); Fn = np.zeros((a.res, a.res), np.int64)
    PC = np.zeros((3, a.res, a.res), np.float32)
    # function-identity basis: PCA of the output vector tanh(f(x)) over all inputs, fitted on a 128^2
    # coarse grid of the same window
    axc = np.linspace(-a.span, a.span, 128)
    Ac, Bc = np.meshgrid(ca + axc, cb + axc)
    thc = origin[None] + torch.tensor(Ac.ravel())[:, None] * d1[None] + torch.tensor(Bc.ravel())[:, None] * d2[None]
    Oc = torch.tanh(fwd(thc, X, k, h)).numpy()
    mu = Oc.mean(0); _, sv, Vt = np.linalg.svd(Oc - mu, full_matrices=False)
    basis = Vt[:3]; evr = (sv[:3] ** 2 / (sv ** 2).sum()).tolist()
    pw = torch.tensor([2 ** i for i in range(len(y))], dtype=torch.long)
    for i, bv in enumerate(cb + ax):
        A = torch.tensor(ca + ax)
        thp = origin[None] + A[:, None] * d1[None] + bv * d2[None]
        f = fwd(thp, X, k, h)
        Lg[i] = ((torch.tanh(f) - y) ** 2).mean(-1).numpy()
        Fn[i] = ((f > 0).long() * pw).sum(-1).numpy()
        PC[:, i] = ((torch.tanh(f).numpy() - mu) @ basis.T).T
    ytrue = int(((y > 0).long() * pw).sum())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, loss=Lg.astype(np.float32), fn=Fn.astype(np.int32), pc=PC, pc_evr=evr, ytrue=ytrue, marks=np.array([[float(u), float(v)] for u, v in marks]).reshape(-1, 2),
                        span=a.span, centre=(ca, cb), theta=th.numpy(), L0=L0)
    print(json.dumps(dict(out=a.out, L0=L0, n_functions=int(len(np.unique(Fn))), frac_solved=float((Fn == ytrue).mean()), pc_evr=[round(e, 3) for e in evr],
                          lmin=float(Lg.min()), lmax=float(Lg.max()))))


if __name__ == "__main__":
    main()
