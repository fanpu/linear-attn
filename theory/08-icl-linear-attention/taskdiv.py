"""Task-diversity sweep (Raventos, Paul, Chen & Ganguli, NeurIPS 2023) with G GPT-2-style models trained in parallel.

Setup (their Sec. 3 / App. B): D=8, K=16 in-context pairs, noise variance 0.25, tokens interleave x_k and (y_k,0..0),
causal decoder, MSE on every x-token.  Their "small" model = 4 layers, 64-dim embeddings, 2 heads; Adam, one-cycle
triangle LR (50% warm-up, peak 1e-3).  We train all task-pool sizes M in one process: parameters carry a leading
model axis G and each model draws tasks from its own pool.

  .venv/bin/python 08-icl-linear-attention/taskdiv.py --steps 60000 --batch 512
"""
import argparse, math, time, pathlib, json
import numpy as np
import torch
import torch.nn.functional as F
from icl_core import ridge_w, dmmse_prefix_w

p = argparse.ArgumentParser()
p.add_argument("--D", type=int, default=8)
p.add_argument("--K", type=int, default=16)
p.add_argument("--noise_var", type=float, default=0.25)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--width", type=int, default=64)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--steps", type=int, default=60000)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--logM", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,inf")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", type=str, default="small")
p.add_argument("--eval_every", type=int, default=5000)
p.add_argument("--bench", action="store_true")
p.add_argument("--compile", type=int, default=1)
a = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.08)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
torch.manual_seed(a.seed)
D, K, sig = a.D, a.K, math.sqrt(a.noise_var)
logMs = [None if s == "inf" else int(s) for s in a.logM.split(",")]
G = len(logMs)
gen = torch.Generator(device=dev).manual_seed(a.seed)
pools = [None if m is None else torch.randn(2**m, D, device=dev, generator=gen) for m in logMs]
T, W, H, Lyr = 2 * K, a.width, a.heads, a.layers


# ----------------------------------------------------------------------------------------------- model
def init_params():
    def lin(i, o, std=0.02):
        return torch.randn(G, o, i, device=dev) * std
    P = {"inp.w": lin(D, W), "inp.b": torch.zeros(G, W, device=dev), "pos": torch.randn(G, T, W, device=dev) * 0.02,
         "lnf.g": torch.ones(G, W, device=dev), "lnf.b": torch.zeros(G, W, device=dev),
         "out.w": lin(W, 1), "out.b": torch.zeros(G, 1, device=dev)}
    for l in range(Lyr):
        s = 0.02 / math.sqrt(2 * Lyr)  # GPT-2 residual scaling
        P |= {f"{l}.ln1.g": torch.ones(G, W, device=dev), f"{l}.ln1.b": torch.zeros(G, W, device=dev),
              f"{l}.qkv.w": lin(W, 3 * W), f"{l}.qkv.b": torch.zeros(G, 3 * W, device=dev),
              f"{l}.proj.w": lin(W, W, s), f"{l}.proj.b": torch.zeros(G, W, device=dev),
              f"{l}.ln2.g": torch.ones(G, W, device=dev), f"{l}.ln2.b": torch.zeros(G, W, device=dev),
              f"{l}.fc.w": lin(W, 4 * W), f"{l}.fc.b": torch.zeros(G, 4 * W, device=dev),
              f"{l}.fc2.w": lin(4 * W, W, s), f"{l}.fc2.b": torch.zeros(G, W, device=dev)}
    return {k: v.requires_grad_() for k, v in P.items()}


def glin(x, P, name):  # x [G,B,T,i]
    return torch.einsum("gbti,goi->gbto", x, P[name + ".w"]) + P[name + ".b"][:, None, None]


def gln(x, P, name):
    return F.layer_norm(x, (W,)) * P[name + ".g"][:, None, None] + P[name + ".b"][:, None, None]


def forward(P, tok):  # tok [G,B,T,D] -> predictions at x-tokens [G,B,K]
    Gg, B = tok.shape[:2]
    h = glin(tok, P, "inp") + P["pos"][:, None]
    for l in range(Lyr):
        z = gln(h, P, f"{l}.ln1")
        q, k, v = glin(z, P, f"{l}.qkv").split(W, -1)
        sh = lambda t: t.reshape(Gg * B, T, H, W // H).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        o = o.transpose(1, 2).reshape(Gg, B, T, W)
        h = h + glin(o, P, f"{l}.proj")
        z = gln(h, P, f"{l}.ln2")
        h = h + glin(F.gelu(glin(z, P, f"{l}.fc")), P, f"{l}.fc2")
    return glin(gln(h, P, "lnf"), P, "out")[..., 0::2, 0]


# ----------------------------------------------------------------------------------------------- data
def sample(B, which="pretrain"):
    ws = []
    for pool in pools:
        if which == "true" or pool is None:
            ws.append(torch.randn(B, D, device=dev, generator=gen))
        else:
            ws.append(pool[torch.randint(0, pool.shape[0], (B,), device=dev, generator=gen)])
    w = torch.stack(ws)                                            # [G,B,D]
    x = torch.randn(G, B, K, D, device=dev, generator=gen)
    y = torch.einsum("gbkd,gbd->gbk", x, w) + sig * torch.randn(G, B, K, device=dev, generator=gen)
    return x, y, w


def tokens(x, y):
    yt = torch.zeros_like(x); yt[..., 0] = y
    return torch.stack([x, yt], 3).reshape(*x.shape[:2], T, D)


# ----------------------------------------------------------------------------------------------- baselines
def prefix_preds(x, y, fn):
    """Predictions for every x_k from pairs < k, for estimator fn(X,y)->w; x [B,K,D]."""
    B = x.shape[0]
    out = torch.zeros(B, K, device=x.device, dtype=x.dtype)
    for k in range(1, K):
        out[:, k] = (fn(x[:, :k], y[:, :k]) * x[:, k]).sum(-1)
    return out


@torch.no_grad()
def evaluate(P, nb=4, B=1024):
    res = {}
    for which in ["pretrain", "true"]:
        acc = {key: np.zeros((G, K)) for key in ["pt", "ridge", "dmmse", "d_pt_ridge", "d_pt_dmmse", "d_ridge_dmmse"]}
        for _ in range(nb):
            x, y, w = sample(B, which)
            pt = forward(P, tokens(x, y)).double()
            for g in range(G):
                xd, yd = x[g].double(), y[g].double()
                rid = prefix_preds(xd, yd, lambda X, Y: ridge_w(X, Y, a.noise_var))
                if pools[g] is None:
                    dm = rid  # infinite pool: Bayes estimator is ridge
                else:
                    Wd = dmmse_prefix_w(xd, yd, pools[g].double(), sig)          # [B,K+1,D]
                    dm = (Wd[:, :K] * xd).sum(-1)
                se = lambda u, v: ((u - v) ** 2).mean(0).cpu().numpy() / nb
                acc["pt"][g] += se(pt[g], yd); acc["ridge"][g] += se(rid, yd); acc["dmmse"][g] += se(dm, yd)
                acc["d_pt_ridge"][g] += se(pt[g], rid); acc["d_pt_dmmse"][g] += se(pt[g], dm); acc["d_ridge_dmmse"][g] += se(rid, dm)
        res[which] = {k: (v / D).tolist() for k, v in acc.items()}   # per-dimension, per position
    return res


# ----------------------------------------------------------------------------------------------- train
P = init_params()
opt = torch.optim.Adam(P.values(), lr=a.lr, fused=True)
warm = a.steps // 2
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warm, max(0.0, (a.steps - s) / (a.steps - warm))) if s > 0 else 1e-3)
def lossf(P, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return ((forward(P, tokens(x, y)).float() - y) ** 2).mean((1, 2)).sum()  # sum over models


if a.compile:
    lossf = torch.compile(lossf)
out = pathlib.Path(__file__).parent / "cache" / f"taskdiv_{a.tag}_s{a.seed}.json"
hist = {"args": vars(a), "evals": []}
t0 = time.time()
start = 1
ck = out.with_suffix(".ckpt")
if ck.exists():
    c = torch.load(ck)
    with torch.no_grad():
        for k in P:
            P[k].copy_(c["P"][k].to(dev))
    opt.load_state_dict(c["opt"])
    for _ in range(c["step"]):
        sched.step()
    start = c["step"] + 1
    hist = json.load(open(out))
    print("resumed from", c["step"])
for step in range(start, a.steps + 1):
    x, y, _ = sample(a.batch)
    loss = lossf(P, x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); sched.step()
    if a.bench and step == 50:
        torch.cuda.synchronize(); t1 = time.time()
    if a.bench and step == 150:
        torch.cuda.synchronize(); print(f"{(time.time()-t1)/100*1000:.1f} ms/step for G={G}, B={a.batch}"); raise SystemExit
    if step % 1000 == 0:
        print(f"step {step} loss/G {loss.item()/G:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        ev = evaluate(P, nb=2 if step < a.steps else 8)
        ev["step"] = step
        hist["evals"].append(ev)
        m = lambda key, wh: np.mean(ev[wh][key], 1)
        print("  true MSE/D  PT:", np.round(m("pt", "true"), 3), "\n  ridge:", np.round(m("ridge", "true"), 3),
              "\n  dmmse:", np.round(m("dmmse", "true"), 3), flush=True)
        json.dump(hist, open(out, "w"))
        torch.save({"P": {k: v.detach().cpu() for k, v in P.items()}, "opt": opt.state_dict(), "step": step}, out.with_suffix(".ckpt"))
torch.save({k: v.detach().cpu() for k, v in P.items()}, out.with_suffix(".pt"))
print("saved", out)
