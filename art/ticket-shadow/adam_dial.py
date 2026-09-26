"""Follow-up (2026-09-26 evening): what does the optimiser do to the ticket's shadow?

Same protocol as imp.py (LeNet-300-100, MNIST raw [0,1] pixels, batch 60, 20k steps per round,
layer-wise magnitude pruning 20/20/10%, rewind to init), but many optimiser configurations are
trained side by side as stacked independent MLPs in one process. The optimiser is a vectorised
elementwise update with per-model hyperparameters, so each model trains exactly as it would alone:

  adam   : m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ; w -= lr * mhat / (sqrt(vhat) + eps)
           (+ decoupled weight decay w -= lr*wd*w first, as torch.optim.AdamW)
  sgd    : buf = mu buf + g ; w -= lr * buf            (torch.optim.SGD, dampening 0)
  sign   : w -= lr * sign(g)                           (sign-SGD, no momentum)
  signum : buf = b1 buf + (1-b1) g ; w -= lr * sign(buf)

One process = one family of configurations (a --family below), 3 seeds each.
Output: cache/followup_<family>/round_XX.npz in imp.py's format plus a `cfg` label per model.
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import pasar_job
from imp import load_dataset, keep_counts, glorot_init, next_masks, SIZES

ROOT = "/home/fzeng/ml/research/art"

# (label, kind, lr, b1, b2, eps, wd)
FAMILIES = {
    "eps": [(f"adam_eps{e:g}", "adam", 1.2e-3, 0.9, 0.999, e, 0.0)
            for e in [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]]
,
    "mech": [("adamw_wd0.01", "adam", 1.2e-3, 0.9, 0.999, 1e-8, 0.01),
             ("adamw_wd0.1", "adam", 1.2e-3, 0.9, 0.999, 1e-8, 0.1),
             ("adam_b2_0.9", "adam", 1.2e-3, 0.9, 0.9, 1e-8, 0.0),
             ("adam_b2_0.99", "adam", 1.2e-3, 0.9, 0.99, 1e-8, 0.0),
             ("adam_b2_0.9999", "adam", 1.2e-3, 0.9, 0.9999, 1e-8, 0.0)],
    "sign": [("signsgd_lr1e-4", "sign", 1e-4, 0.0, 0.0, 0.0, 0.0),
             ("signum_lr1e-4", "signum", 1e-4, 0.9, 0.0, 0.0, 0.0),
             ("sgdm0.9_lr0.01", "sgd", 0.01, 0.9, 0.0, 0.0, 0.0),
             ("adam_eps0.1_lr0.012", "adam", 1.2e-2, 0.9, 0.999, 1e-1, 0.0)],
}
FAMILIES["mechsign"] = FAMILIES["mech"] + FAMILIES["sign"]
FAMILIES["signlr"] = [("signum_lr3e-5", "signum", 3e-5, 0.9, 0.0, 0.0, 0.0), ("signum_lr3e-4", "signum", 3e-4, 0.9, 0.0, 0.0, 0.0),
                      ("signsgd_lr3e-5", "sign", 3e-5, 0.0, 0.0, 0.0, 0.0), ("signsgd_lr3e-4", "sign", 3e-4, 0.0, 0.0, 0.0, 0.0),
                      ("signum_lr1e-4", "signum", 1e-4, 0.9, 0.0, 0.0, 0.0), ("signsgd_lr1e-4", "sign", 1e-4, 0.0, 0.0, 0.0, 0.0)]
KIND = {"adam": 0, "sgd": 1, "sign": 2, "signum": 3}


class StackedOpt:
    """Elementwise optimiser over params of shape (M, ...) with per-model hyperparameters."""

    def __init__(self, params, cfgs, dev):
        self.params = params
        M = len(cfgs)
        t = lambda xs: torch.tensor(xs, dtype=torch.float32, device=dev)
        self.lr = t([c[2] for c in cfgs]); self.b1 = t([c[3] for c in cfgs]); self.b2 = t([c[4] for c in cfgs])
        self.eps = t([c[5] for c in cfgs]); self.wd = t([c[6] for c in cfgs])
        k = torch.tensor([KIND[c[1]] for c in cfgs], device=dev)
        self.is_adam = (k == 0).float(); self.is_sgd = (k == 1).float(); self.is_sign = ((k == 2) | (k == 3)).float()
        # momentum accumulation: EMA (1-b1) for adam/signum/sign(b1=0 -> buf=g); plain sum for sgd
        self.c1 = torch.where(k == 1, torch.ones_like(self.b1), 1 - self.b1)
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        bc1 = torch.where(self.is_adam > 0, bc1, torch.ones_like(bc1))
        bc2 = torch.where(self.is_adam > 0, bc2, torch.ones_like(bc2))
        eps = torch.where(self.is_adam > 0, self.eps, torch.ones_like(self.eps))
        for p, m, v in zip(self.params, self.m, self.v):
            sh = (-1,) + (1,) * (p.dim() - 1)
            _update(p, p.grad, m, v, *(x.view(sh) for x in (self.lr, self.b1, self.b2, eps, self.wd, self.is_adam,
                                                              self.is_sgd, self.is_sign, self.c1, bc1, bc2)))


def _update_eager(p, g, m, v, lr, b1, b2, eps, wd, ia, ig, isg, c1, bc1, bc2):
    p.mul_(1 - lr * wd)
    m.mul_(b1).add_(c1 * g)
    v.mul_(b2).addcmul_((1 - b2) * ia, g * g)
    upd = ia * (m / bc1) / (torch.sqrt(v / bc2) + eps) + ig * m + isg * torch.sign(m)
    p.sub_(lr * upd)


_update = _update_eager
if os.environ.get("ADAM_DIAL_COMPILE", "1") == "1":
    _update = torch.compile(_update_eager, dynamic=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        pasar_job.apply_memory_limit()
        torch.backends.cuda.matmul.allow_tf32 = False
    outdir = f"{ROOT}/ticket-shadow/cache/followup_{a.family}{a.tag}"
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(vars(a), configs=FAMILIES[a.family]), open(f"{outdir}/args.json", "w"), indent=1)

    seeds0 = [int(s) for s in a.seeds.split(",")]
    cfgs, seeds, labels = [], [], []
    for c in FAMILIES[a.family]:
        for s in seeds0:
            cfgs.append(c); seeds.append(s); labels.append(c[0])
    M = len(cfgs)
    modes = ["imp"] * M

    xtr, ytr, xte, yte = load_dataset("mnist")  # raw [0,1] pixels
    xva, yva = xtr[55000:], ytr[55000:]
    xtr, ytr = xtr[:55000], ytr[:55000]
    Xtr = torch.tensor(xtr, device=dev)[None]; Xva = torch.tensor(xva, device=dev)[None]; Xte = torch.tensor(xte, device=dev)[None]
    Ytr = torch.tensor(ytr, device=dev); Yva = torch.tensor(yva, device=dev); Yte = torch.tensor(yte, device=dev)
    N = Xtr.shape[1]
    counts = keep_counts(a.rounds)
    perms = np.stack([np.arange(784)] * M)

    theta0 = glorot_init(M, seeds, 0, dev)  # same init per seed as imp.py
    masks = [torch.ones(M, fi, fo, dtype=torch.bool, device=dev) for fi, fo in SIZES]
    start = 0
    done = sorted(f for f in os.listdir(outdir) if f.startswith("round_"))
    if done:
        last = np.load(f"{outdir}/{done[-1]}")
        r_last = int(last["round"])
        cur = [torch.tensor(np.unpackbits(last[f"mask{l}"], axis=-1)[..., :SIZES[l][1]].astype(bool), device=dev) for l in range(3)]
        fw = [torch.tensor(last[f"W{l}"].astype(np.float32), device=dev) for l in range(3)]
        masks = next_masks(cur, fw, theta0, modes, seeds, counts, r_last + 1, M, dev)
        start = r_last + 1
        pasar_job.resumed(start * a.steps)
        print("resumed at round", start, flush=True)

    total = a.rounds * a.steps
    t0 = time.time()
    for r in range(start, a.rounds):
        Ws = [torch.nn.Parameter(w.clone() * mk) for w, mk in zip(theta0, masks)]
        bs = [torch.nn.Parameter(torch.zeros(M, 1, fo, device=dev)) for _, fo in SIZES]
        opt = StackedOpt(Ws + bs, cfgs, dev)
        mf = [mk.float() for mk in masks]

        def fwd(x):
            h = x
            for l in range(3):
                h = torch.baddbmm(bs[l], h, Ws[l] * mf[l])
                if l < 2:
                    h = F.relu(h)
            return h

        @torch.no_grad()
        def evaluate(X, Y):
            acc, los = torch.zeros(M, device=dev), torch.zeros(M, device=dev)
            for i in range(0, X.shape[1], 5000):
                lo = fwd(X[:, i:i + 5000].expand(M, -1, -1)); yb = Y[i:i + 5000]
                los += F.cross_entropy(lo.reshape(-1, 10), yb.repeat(M), reduction="none").view(M, -1).sum(1)
                acc += (lo.argmax(-1) == yb).float().sum(1)
            return (los / X.shape[1]).cpu().numpy(), (acc / X.shape[1]).cpu().numpy()

        gen = torch.Generator(device=dev).manual_seed(12345 + 7919 * r + seeds[0])
        curve = []
        pos, order = N, None
        for step in range(a.steps + 1):
            if step % a.eval_every == 0 or step == a.steps:
                vl, va = evaluate(Xva, Yva); _, ta = evaluate(Xte, Yte)
                curve.append((step, vl, va, ta))
            if step == a.steps:
                break
            if pos + a.batch > N:
                order = torch.argsort(torch.rand(M, N, device=dev, generator=gen), dim=1); pos = 0
            idx = order[:, pos:pos + a.batch]; pos += a.batch
            xb = Xtr[0][idx]; yb = Ytr[idx]
            lo = fwd(xb)
            loss = F.cross_entropy(lo.reshape(-1, 10), yb.reshape(-1), reduction="sum") / a.batch
            for p in opt.params:
                p.grad = None
            loss.backward()
            opt.step()
            if step % 1000 == 0:
                pasar_job.progress(r * a.steps + step, total, loss=float(loss.detach()) / M)
        final_W = [w.detach() * mk for w, mk in zip(Ws, mf)]
        st = np.array([c[0] for c in curve]); vl = np.stack([c[1] for c in curve])
        va = np.stack([c[2] for c in curve]); ta = np.stack([c[3] for c in curve]); es = vl.argmin(0)
        save = dict(round=r, seeds=np.array(seeds), modes=np.array(modes), cfg=np.array(labels), perms=perms,
                    counts=np.array(counts[r]), steps=st, val_loss=vl, val_acc=va, test_acc=ta,
                    es_iter=st[es], es_test_acc=ta[es, np.arange(M)], final_test_acc=ta[-1])
        for l in range(3):
            save[f"mask{l}"] = np.packbits(masks[l].cpu().numpy(), axis=-1)
            save[f"W{l}"] = final_W[l].cpu().numpy().astype(np.float16)
            if r == 0:
                save[f"init_W{l}"] = theta0[l].cpu().numpy()
        np.savez_compressed(f"{outdir}/round_{r:02d}.npz", **save)
        pasar_job.checkpoint(r)
        accs = {labels[m]: round(float(save["es_test_acc"][m]), 4) for m in range(0, M, len(seeds0))}
        print(f"[{a.family}] round {r:2d} keep={counts[r]} es_test(seed0)={accs} t={time.time() - t0:.0f}s", flush=True)
        if r + 1 < a.rounds:
            masks = next_masks(masks, final_W, theta0, modes, seeds, counts, r + 1, M, dev)
    print("done", flush=True)


if __name__ == "__main__":
    main()
