"""Top-2 Hessian eigenvectors of the ResNet loss (the same fixed images as resnet_plane.py, BN eval),
restricted to conv / linear weights (dim >= 2 tensors; BN and biases held fixed, as in the random
filter-normalised directions).  Lanczos (scipy eigsh, which='LA') on exact Hessian-vector products
(double backprop, fp32, TF32 off).

    python hessian.py --model resnet56 --n 128   -> cache/resnet/hess_resnet56.pt
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/fzeng/ml/research/art/loss-landscape")
import common as LL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet56")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    import pasar_job; pasar_job.apply_memory_limit()
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    net = LL.load_model(a.model)
    X, Y = LL.load_cifar(True)
    idx = torch.tensor(LL.fixed_subset(a.n), device="cuda")
    x, y = X[idx].contiguous(), Y[idx]
    params = list(net.parameters())
    mask = [p.dim() >= 2 for p in params]
    ps = [p for p, m in zip(params, mask) if m]
    sizes = [p.numel() for p in ps]
    D = sum(sizes)
    loss = F.cross_entropy(net(x), y)
    g = torch.autograd.grad(loss, ps, create_graph=True)
    gflat = torch.cat([t.reshape(-1) for t in g])
    count = [0]

    def hvp(v):
        count[0] += 1
        vt = torch.tensor(v, device="cuda", dtype=torch.float32)
        hv = torch.autograd.grad((gflat * vt).sum(), ps, retain_graph=True)
        if count[0] % 10 == 0:
            pasar_job.progress(min(count[0], 199), 200)
        return torch.cat([t.reshape(-1) for t in hv]).double().cpu().numpy()

    from scipy.sparse.linalg import LinearOperator, eigsh
    t0 = time.time()
    op = LinearOperator((D, D), matvec=hvp, dtype=np.float64)
    vals, vecs = eigsh(op, k=a.k, which="LA", tol=1e-4, v0=np.random.default_rng(0).standard_normal(D))
    order = np.argsort(-vals); vals = vals[order]; vecs = vecs[:, order]
    # also the most negative eigenvalue (for the record)
    vneg, _ = eigsh(op, k=1, which="SA", tol=1e-3, v0=np.random.default_rng(1).standard_normal(D))
    # rayleigh check
    ray = [float(vecs[:, i] @ hvp(vecs[:, i])) for i in range(2)]
    out = []
    for i in range(2):
        v = torch.tensor(vecs[:, i], dtype=torch.float32); full, j = [], 0
        for p, m in zip(params, mask):
            if m:
                full.append(v[j:j + p.numel()].view_as(p).clone()); j += p.numel()
            else:
                full.append(torch.zeros_like(p, device="cpu"))
        out.append(full)
    # where does the top eigenvector live?  fraction of its squared norm per parameter tensor
    per = []
    k = 0
    for (n, p), m in zip(net.named_parameters(), mask):
        if m:
            per.append((n, float((out[0][k] ** 2).sum()), float((out[1][k] ** 2).sum())))
        k += 1
    os.makedirs(os.path.join(HERE, "cache", "resnet"), exist_ok=True)
    torch.save(dict(vals=vals.tolist(), neg=float(vneg[0]), vecs=out, n=a.n, ray=ray, per=per),
                os.path.join(HERE, "cache", "resnet", f"hess_{a.model}.pt"))
    top = sorted(per, key=lambda r: -r[1])[:5]
    print(json.dumps(dict(model=a.model, vals=vals.tolist(), most_negative=float(vneg[0]), rayleigh=ray, hvps=count[0],
                          secs=round(time.time() - t0, 1), top1_where=top)), flush=True)


if __name__ == "__main__":
    main()
