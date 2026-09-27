"""ResNet-56 (with / without shortcuts, art/loss-landscape checkpoints) on a 2-D plane of weights,
P perturbed copies at once as ONE grouped convolution per layer.

    python resnet_plane.py --model resnet56_noshort --res 512 --span 1 --n 128 --out cache/resnet/...npz
    python resnet_plane.py --bench                                  # throughput test

Plane: w(a,b) = w* + a d1 + b d2.  Directions (--dirs):
  random   Li et al. filter-normalised Gaussians, seeds 1 and 2, BN/bias entries zero:
           exactly art/loss-landscape/common.get_directions (same plane as its 101^2 heroes).
  hessian  top-2 Hessian eigenvectors from hessian.py (cache/resnet/hess_<model>.pt), restricted to
           conv/linear weights and rescaled to the SAME global norm as the random directions.
Loss: mean cross-entropy on a FIXED set of --n training images (fixed_subset(n), seed 0), BN in
eval mode with the checkpoint's running statistics (as Li et al.).  Also stored per point: accuracy,
agreement of argmax predictions with the unperturbed net, and the number of non-finite logits.
fp32; TF32 off unless --tf32.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/fzeng/ml/research/art/loss-landscape")
import common as LL

EPS = 1e-5


class Grouped:
    """Functional ResNet_cifar for P weight vectors at once."""

    def __init__(self, net, dirs, device="cuda"):
        self.net = net
        self.names = [n for n, _ in net.named_parameters()]
        self.w0 = {n: p.detach().float().clone() for n, p in net.named_parameters()}
        self.d = [{n: d.float().to(device) for n, d in zip(self.names, dd)} for dd in dirs]
        self.buf = {n: b.detach().float().clone() for n, b in net.named_buffers()}

    def weights(self, a, b):
        W = {}
        for n, w in self.w0.items():
            sh = (-1,) + (1,) * w.dim()
            if self.d[0][n].abs().max() == 0 and self.d[1][n].abs().max() == 0:
                W[n] = w                                  # unperturbed (BN params, biases)
            else:
                W[n] = w[None] + a.view(sh) * self.d[0][n][None] + b.view(sh) * self.d[1][n][None]
        return W

    def _conv(self, h, w, P, stride, pad, shared):
        if w.dim() == 4:                                  # shared weight: expand to P groups
            w = w[None].expand(P, *w.shape)
        cout = w.shape[1]
        w = w.reshape(P * cout, *w.shape[2:])
        return F.conv2d(h, w, stride=stride, padding=pad, groups=1 if shared else P)

    def _bn(self, h, pre, P):
        g, bb = self.w0[pre + ".weight"], self.w0[pre + ".bias"]
        rm, rv = self.buf[pre + ".running_mean"], self.buf[pre + ".running_var"]
        s = g * torch.rsqrt(rv + EPS)
        return h * s.repeat(P).view(1, -1, 1, 1) + (bb - rm * s).repeat(P).view(1, -1, 1, 1)

    def forward(self, x, a, b):
        P = a.shape[0]
        W = self.weights(a, b)
        h = F.relu(self._bn(self._conv(x, W["conv1.weight"], P, 1, 1, True), "bn1", P))
        for li, layer in enumerate([self.net.layer1, self.net.layer2, self.net.layer3]):
            for bi, blk in enumerate(layer):
                pre = f"layer{li + 1}.{bi}"
                s = blk.conv1.stride[0]
                o = F.relu(self._bn(self._conv(h, W[pre + ".conv1.weight"], P, s, 1, False), pre + ".bn1", P))
                o = self._bn(self._conv(o, W[pre + ".conv2.weight"], P, 1, 1, False), pre + ".bn2", P)
                if blk.use_short:
                    if len(blk.shortcut) > 0:
                        sc = self._bn(self._conv(h, W[pre + ".shortcut.0.weight"], P, s, 0, False), pre + ".shortcut.1", P)
                    else:
                        sc = h
                    o = o + sc
                h = F.relu(o)
        f = h.mean((2, 3)).view(h.shape[0], P, -1)                      # (N, P, 64)
        fw = W["linear.weight"]
        fw = fw if fw.dim() == 3 else fw[None].expand(P, *fw.shape)
        return torch.einsum("npc,pkc->npk", f, fw) + W["linear.bias"]


def load_dirs(kind, net, model):
    if kind == "random":
        return LL.get_directions(net)
    if kind.startswith("hessian"):
        H = torch.load(os.path.join(HERE, "cache", "resnet", f"hess_{model}.pt"))
        R = LL.get_directions(net)
        out = []
        for k in range(2):
            v = H["vecs"][k]                                 # list aligned with parameters, unit global norm
            tgt = float(torch.sqrt(sum((r.double() ** 2).sum() for r in R[k])))
            out.append([(t.double() * tgt).float().cuda() for t in v])
        return out
    raise KeyError(kind)


@torch.no_grad()
def evaluate(G, x, y, a_all, b_all, P, ref_pred, prog=None, nb=None):
    n = len(a_all)
    L = np.zeros(n); A = np.zeros(n); AG = np.zeros(n); NF = np.zeros(n)
    nb = nb or x.shape[0]
    for c0 in range(0, n, P):
        a = torch.tensor(a_all[c0:c0 + P], device="cuda", dtype=torch.float32)
        b = torch.tensor(b_all[c0:c0 + P], device="cuda", dtype=torch.float32)
        p = a.shape[0]
        tl = torch.zeros(p, device="cuda", dtype=torch.float64); ta = torch.zeros_like(tl); tg = torch.zeros_like(tl)
        tn = torch.zeros_like(tl)
        for i in range(0, x.shape[0], nb):
            lo = G.forward(x[i:i + nb], a, b).double()                  # (N, p, 10)
            fin = torch.isfinite(lo).all(-1)                            # (N, p)
            tn += (~fin).sum(0)
            lp = torch.log_softmax(lo, -1)
            yy = y[i:i + nb]
            nll = -lp.gather(-1, yy[:, None, None].expand(-1, p, 1))[..., 0]
            nll = torch.where(fin, nll, torch.full_like(nll, float("inf")))
            tl += nll.sum(0)
            pred = lo.argmax(-1)
            ta += (pred == yy[:, None]).sum(0); tg += (pred == ref_pred[i:i + nb, None]).sum(0)
        N = x.shape[0]
        L[c0:c0 + p] = (tl / N).cpu().numpy(); A[c0:c0 + p] = (ta / N).cpu().numpy()
        AG[c0:c0 + p] = (tg / N).cpu().numpy(); NF[c0:c0 + p] = (tn / N).cpu().numpy()
        if prog: prog(c0 + p, n)
    return L, A, AG, NF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet56_noshort")
    ap.add_argument("--dirs", default="random")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--centre", default="0,0")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--nb", type=int, default=None)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--rows", default=None, help="r0,r1: only these grid rows (sharding)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pasar_job
    pasar_job.apply_memory_limit()
    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.benchmark = True
    net = LL.load_model(a.model)
    for p_ in net.parameters(): p_.requires_grad_(False)
    X, Y = LL.load_cifar(True)
    idx = torch.tensor(LL.fixed_subset(a.n), device="cuda")
    x, y = X[idx].contiguous(), Y[idx]
    del X, Y
    dirs = load_dirs(a.dirs, net, a.model)
    G = Grouped(net, dirs)
    ref_pred = net(x).argmax(1)
    # check: grouped evaluator at P=2 vs nn.Module
    with torch.no_grad():
        chk = LL.LossEvaluator(net, x, y, dirs[0], dirs[1], batch=a.n)
        l_ref, _ = chk(0.3, -0.2)
        net.load_state_dict(LL.load_model(a.model).state_dict())
        L, _, _, _ = evaluate(G, x, y, np.array([0.3, 0.0]), np.array([-0.2, 0.0]), 2, ref_pred)
    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    print(f"check: module {l_ref:.6f} grouped {L[0]:.6f} centre {L[1]:.6f}", flush=True)
    if a.bench:
        for P in (8, 16, 32, 64):
            aa = np.random.rand(4 * P); bb = np.random.rand(4 * P)
            evaluate(G, x, y, aa[:P], bb[:P], P, ref_pred)
            torch.cuda.synchronize(); t0 = time.time()
            evaluate(G, x, y, aa, bb, P, ref_pred)
            torch.cuda.synchronize(); dt = time.time() - t0
            print(f"P={P} n={a.n}: {4 * P / dt:.1f} points/s, {4 * P * a.n / dt:.0f} img/s, "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return
    ca, cb = (float(v) for v in a.centre.split(","))
    ax = np.linspace(-a.span, a.span, a.res)
    r0, r1 = (0, a.res) if a.rows is None else (int(v) for v in a.rows.split(","))
    B_, A_ = np.meshgrid(cb + ax[r0:r1], ca + ax, indexing="ij")
    out = a.out
    part = out + ".part.npz"
    done = 0; res = [np.zeros(A_.size) for _ in range(4)]
    ck = os.path.join(pasar_job.persist_dir(), os.path.basename(part))
    for cand in (ck, part):
        if os.path.exists(cand):
            z = np.load(cand); done = int(z["done"]); res = [z[k] for k in ("L", "A", "AG", "NF")]
            pasar_job.resumed(done); print("resumed at", done, flush=True); break
    t0 = time.time(); last = t0
    a_all, b_all = A_.ravel(), B_.ravel()
    step = a.P * 64
    for c0 in range(done, A_.size, step):
        r = evaluate(G, x, y, a_all[c0:c0 + step], b_all[c0:c0 + step], a.P, ref_pred, nb=a.nb)
        for k in range(4): res[k][c0:c0 + step] = r[k]
        done = min(c0 + step, A_.size)
        pasar_job.progress(done, A_.size)
        if time.time() - last > 300 or done == A_.size:
            np.savez(ck, done=done, L=res[0], A=res[1], AG=res[2], NF=res[3]); pasar_job.checkpoint(done); last = time.time()
        print(f"{done}/{A_.size} {done / (time.time() - t0 + 1e-9):.1f} pts/s", flush=True)
    sh = (r1 - r0, a.res)
    np.savez_compressed(out, loss=res[0].reshape(sh), acc=res[1].reshape(sh), agree=res[2].reshape(sh),
                        nonfinite=res[3].reshape(sh), span=a.span, centre=(ca, cb), rows=(r0, r1), res=a.res,
                        model=a.model, dirs=a.dirs, n=a.n, tf32=a.tf32, secs=time.time() - t0)
    print(json.dumps(dict(out=out, secs=round(time.time() - t0, 1), lmin=float(np.nanmin(res[0])),
                          lmax=float(np.nanmax(res[0][np.isfinite(res[0])])) if np.isfinite(res[0]).any() else None,
                          nonfinite_pts=int((~np.isfinite(res[0])).sum()))), flush=True)


if __name__ == "__main__":
    main()
