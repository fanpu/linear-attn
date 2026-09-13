"""Teacher/student scaling-law sweep (Sharma & Kaplan 2020/2022, sec. 3.1, MSE variant).

Many independent students are trained *in parallel* as stacked tensors (bmm), one stack per
student width n. Every stacked model m has its own (teacher family, d, seed).

Data manifold: z ~ U[-1/2, 1/2]^d, embedded linearly in R^D (D = 24) by a random orthonormal
matrix Q_d (the paper instead zero-pads a 20-dim input; for a Gaussian-initialised teacher the two
are equal in distribution). Target: a random teacher MLP [D, 600, 600, 1] applied to x = Q_d z.

Teacher families
  relu0 : ReLU, zero biases, W ~ N(0, 1/fan_in)            -- exactly the paper's teacher
  relub : ReLU, W ~ N(0, 1/fan_in), biases ~ N(0, 0.1^2)   -- breaks positive homogeneity

Teacher outputs are standardised (zero mean, unit variance over the manifold) so L is a relative MSE.

Training data: a fixed pool of P fresh teacher samples per (family, d) on the GPU (P >> N);
minibatches are drawn with replacement. Test loss uses a separate 200k-point set.
Student: ReLU MLP [D, n, n, 1], PyTorch-default init, Adam, step schedule with growing batch
(paper's Table 1 uses a step schedule with growing batch; we use a constant lr followed by a
cosine decay, which lowers gradient noise the same way at a fraction of the cost), float32 with TF32 disabled.
"""
import argparse, json, math, time, os
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--widths", type=str, default="8,32")
p.add_argument("--dims", type=str, default="2,3,4,5,6,8,10,12")
p.add_argument("--families", type=str, default="relu0,relub")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--steps", type=int, default=60000)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--lr_final", type=float, default=3e-6)
p.add_argument("--decay_frac", type=float, default=0.5)  # constant lr, then cosine decay over last fraction
p.add_argument("--pool", type=int, default=2_000_000)
p.add_argument("--ntest", type=int, default=200_000)
p.add_argument("--D", type=int, default=24)
p.add_argument("--out", type=str, required=True)
p.add_argument("--log_every", type=int, default=2000)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
D = args.D
dims = [int(v) for v in args.dims.split(",")]
fams = args.families.split(",")
widths = [int(v) for v in args.widths.split(",")]
dmax = max(dims)

# ---------------------------------------------------------------- teachers and embeddings (fixed seed)
g = torch.Generator(device="cpu").manual_seed(1234)
Qs = {}
for d in dims:
    Q, _ = torch.linalg.qr(torch.randn(D, d, generator=g, dtype=torch.float64))
    Qs[d] = Q.float()

def make_teacher(fam, gen):
    sizes = [D, 600, 600, 1]
    Ws, bs = [], []
    for a, b in zip(sizes[:-1], sizes[1:]):
        Ws.append(torch.randn(b, a, generator=gen) / math.sqrt(a))
        bs.append(torch.zeros(b) if fam == "relu0" else 0.1 * torch.randn(b, generator=gen))
    return Ws, bs

teachers = {f: make_teacher(f, torch.Generator().manual_seed(777 + i)) for i, f in enumerate(fams)}

@torch.no_grad()
def teacher_fwd(fam, x):
    Ws, bs = teachers[fam]
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T.to(dev) + b.to(dev)
        if i < len(Ws) - 1:
            h = torch.relu(h)
    return h[:, 0]

@torch.no_grad()
def sample(fam, d, n, gen):
    z = torch.rand(n, d, device=dev, generator=gen) - 0.5
    x = z @ Qs[d].T.to(dev)
    y = torch.cat([teacher_fwd(fam, x[i:i + 200_000]) for i in range(0, n, 200_000)])
    return x, y

gdev = torch.Generator(device=dev).manual_seed(99)
pools, tests, norms = {}, {}, {}
t0 = time.time()
for f in fams:
    for d in dims:
        x, y = sample(f, d, args.pool, gdev)
        mu, sd = y.mean(), y.std()
        pools[(f, d)] = (x, (y - mu) / sd)
        xt, yt = sample(f, d, args.ntest, gdev)
        tests[(f, d)] = (xt, (yt - mu) / sd)
        norms[(f, d)] = (float(mu), float(sd))
print(f"pools built in {time.time()-t0:.1f}s", flush=True)

configs = [(f, d, s) for f in fams for d in dims for s in range(args.seeds)]
M = len(configs)
Xpool = torch.stack([pools[(f, d)][0] for f, d, s in configs])  # (M, P, D)
Ypool = torch.stack([pools[(f, d)][1] for f, d, s in configs])  # (M, P)
Xtest = torch.stack([tests[(f, d)][0] for f, d, s in configs])
Ytest = torch.stack([tests[(f, d)][1] for f, d, s in configs])
del pools, tests


def init_student(n, seed):
    torch.manual_seed(seed)
    layers = [torch.nn.Linear(D, n), torch.nn.Linear(n, n), torch.nn.Linear(n, 1)]
    return [(l.weight.data.clone(), l.bias.data.clone()) for l in layers]


def fwd(params, x, return_hidden=False):
    h = x
    for i, (W, b) in enumerate(params):
        h = torch.baddbmm(b[:, None, :], h, W.transpose(1, 2))
        if i < len(params) - 1:
            h = torch.relu(h)
            hid = h
    return (h[..., 0], hid) if return_hidden else h[..., 0]


@torch.no_grad()
def test_loss(params, chunk=50_000):
    tot = torch.zeros(M, device=dev, dtype=torch.float64)
    for i in range(0, Xtest.shape[1], chunk):
        e = (fwd(params, Xtest[:, i:i + chunk]) - Ytest[:, i:i + chunk]).double()
        tot += (e * e).sum(1)
    return (tot / Xtest.shape[1]).cpu().numpy()


results = {}
for n in widths:
    t0 = time.time()
    per = [init_student(n, 10_000 + 97 * n + j) for j, (f, d, s) in enumerate(configs)]
    params = []
    for li in range(3):
        W = torch.stack([per[j][li][0] for j in range(M)]).to(dev).requires_grad_(True)
        b = torch.stack([per[j][li][1] for j in range(M)]).to(dev).requires_grad_(True)
        params.append((W, b))
    flat = [t for wb in params for t in wb]
    N = sum(t[0].numel() for t in flat)
    opt = torch.optim.Adam(flat, lr=args.lr, fused=True)
    gidx = torch.Generator(device=dev).manual_seed(5 + n)
    curve = []
    P, B, S = Xpool.shape[1], args.batch, args.steps
    t_decay = int(S * (1 - args.decay_frac))
    perm = torch.randperm(P, device=dev, generator=gidx); pos = 0
    for step in range(1, S + 1):
        if step > t_decay:
            u = (step - t_decay) / (S - t_decay)
            lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * u))
            for grp in opt.param_groups:
                grp["lr"] = lr
        if pos + B > P:
            perm = torch.randperm(P, device=dev, generator=gidx); pos = 0
        idx = perm[pos:pos + B]; pos += B
        xb = Xpool[:, idx]
        yb = Ypool[:, idx]
        loss_per = ((fwd(params, xb) - yb) ** 2).mean(1)
        opt.zero_grad(set_to_none=True)
        loss_per.sum().backward()
        opt.step()
        if step % args.log_every == 0 or step == S:
            tl = test_loss(params)
            curve.append((step, tl))
            print(f"n={n} N={N} step {step} t={time.time()-t0:.0f}s "
                  f"median test loss by d: " +
                  " ".join(f"{d}:{np.median([tl[j] for j,c in enumerate(configs) if c[1]==d]):.2e}" for d in dims),
                  flush=True)
    final = test_loss(params)
    # train-pool loss on a 200k subset, to check for pool overfitting
    with torch.no_grad():
        tr = torch.zeros(M, device=dev, dtype=torch.float64)
        for i in range(0, 200_000, 50_000):
            e = (fwd(params, Xpool[:, i:i + 50_000]) - Ypool[:, i:i + 50_000]).double()
            tr += (e * e).sum(1)
        train = (tr / 200_000).cpu().numpy()
    results[n] = dict(
        N=N, test=final, train=train,
        curve_steps=np.array([c[0] for c in curve]), curve=np.stack([c[1] for c in curve]),
        params=[(W.detach().cpu().numpy(), b.detach().cpu().numpy()) for W, b in params],
        wall=time.time() - t0)
    print(f"== n={n} N={N} done in {time.time()-t0:.0f}s", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
save = dict(configs=np.array([(f, str(d), str(s)) for f, d, s in configs]),
            dims=np.array(dims), fams=np.array(fams), widths=np.array(widths), D=D,
            args=json.dumps(vars(args)), norms=json.dumps({f"{k[0]}_{k[1]}": v for k, v in norms.items()}),
            **{f"Q{d}": Qs[d].numpy() for d in dims},
            **{f"teacher_{f}_W{i}": w.numpy() for f in fams for i, w in enumerate(teachers[f][0])},
            **{f"teacher_{f}_b{i}": b.numpy() for f in fams for i, b in enumerate(teachers[f][1])})
for n, r in results.items():
    save[f"w{n}_N"] = r["N"]; save[f"w{n}_test"] = r["test"]; save[f"w{n}_train"] = r["train"]
    save[f"w{n}_curve_steps"] = r["curve_steps"]; save[f"w{n}_curve"] = r["curve"]
    save[f"w{n}_wall"] = r["wall"]
    for li, (W, b) in enumerate(r["params"]):
        save[f"w{n}_W{li}"] = W; save[f"w{n}_b{li}"] = b
np.savez(args.out, **save)
print("saved", args.out)
