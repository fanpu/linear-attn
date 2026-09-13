"""Train an ensemble of independent one-layer transformers on (a+b) mod 113.

Replicates Nanda et al. 2023 (arXiv:2301.05217) / TransformerLens Grokking_Demo:
  d_model 128, 4 heads (d_head 32), d_mlp 512, ReLU, no LayerNorm, learned pos-emb,
  untied embed/unembed, biases frozen at zero (TL demo), GPT-2-style init
  N(0, 0.8/sqrt(d_model)) on all weight matrices (TL default), input "a b =",
  read logits at "=", float64 log-softmax, full-batch AdamW lr 1e-3 wd 1.0
  betas (0.9, 0.98), 30% train fraction.

The S models are stored with a leading ensemble axis.  The total loss is the
SUM of per-model mean losses, so each model's gradient (and AdamW state, which
is elementwise) is exactly what it would be if trained alone.
"""
import argparse, json, math, os, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--init_seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
p.add_argument("--data_seeds", type=str, default="598,598,598,598,598,598,598,598,1,2,3,4")
p.add_argument("--out", type=str, default="cache/run")
p.add_argument("--eval_every", type=int, default=10)
p.add_argument("--full_every", type=int, default=200)
p.add_argument("--print_every", type=int, default=1000)
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(0.10)
dev = "cuda"
P, D, H, DH, M = 113, 128, 4, 32, 512
FRAC = 0.3
init_seeds = [int(s) for s in args.init_seeds.split(",")]
data_seeds = [int(s) for s in args.data_seeds.split(",")]
S = len(init_seeds)
assert len(data_seeds) == S
os.makedirs(args.out, exist_ok=True)

# ---------------- data ----------------
a = torch.arange(P).repeat_interleave(P)
b = torch.arange(P).repeat(P)
eq = torch.full_like(a, P)
all_tokens = torch.stack([a, b, eq], 1)          # (P*P, 3)
all_labels = (a + b) % P
cut = int(P * P * FRAC)                            # 3830
tr_idx, te_idx = [], []
for ds in data_seeds:
    torch.manual_seed(ds)
    perm = torch.randperm(P * P)
    tr_idx.append(perm[:cut]); te_idx.append(perm[cut:])
tr_idx = torch.stack(tr_idx); te_idx = torch.stack(te_idx)   # (S, n)
Xtr = all_tokens[tr_idx].to(dev); Ytr = all_labels[tr_idx].to(dev)
Xte = all_tokens[te_idx].to(dev); Yte = all_labels[te_idx].to(dev)

# ---------------- params ----------------
std = 0.8 / math.sqrt(D)
shapes = dict(W_E=(P + 1, D), W_pos=(3, D), W_Q=(H, D, DH), W_K=(H, D, DH), W_V=(H, D, DH),
              W_O=(H, DH, D), W_in=(D, M), W_out=(M, D), W_U=(D, P))
params = {k: [] for k in shapes}
for s in init_seeds:
    g = torch.Generator().manual_seed(s)
    for k, sh in shapes.items():
        params[k].append(torch.randn(*sh, generator=g) * std)
params = {k: torch.nn.Parameter(torch.stack(v).to(dev)) for k, v in params.items()}
causal = torch.tril(torch.ones(3, 3, dtype=torch.bool, device=dev))


def forward(X):  # X: (S, B, 3) -> logits read at the '=' position, (S, B, P)
    Sx, B = X.shape[:2]
    x = torch.gather(params["W_E"], 1, X.reshape(Sx, -1, 1).expand(-1, -1, D)).view(Sx, B, 3, D)
    x = x + params["W_pos"][:, None]
    xf = x.reshape(Sx, B * 3, D)
    def proj(name):  # (S,H,D,DH) -> (S,B,H,3,DH)
        W = params[name].permute(0, 2, 1, 3).reshape(Sx, D, H * DH)
        return torch.bmm(xf, W).view(Sx, B, 3, H, DH).transpose(2, 3)
    q, k, v = proj("W_Q"), proj("W_K"), proj("W_V")
    att = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    att = att.masked_fill(~causal, float("-inf")).softmax(-1)
    z = (att @ v)[:, :, :, -1]                                  # S B H DH: the '=' query row
    r = x[:, :, -1] + torch.bmm(z.reshape(Sx, B, H * DH), params["W_O"].reshape(Sx, H * DH, D))
    h = torch.relu(torch.bmm(r, params["W_in"]))
    r = r + torch.bmm(h, params["W_out"])
    return torch.bmm(r, params["W_U"])


def loss_acc(logits, Y):
    lp = logits.double().log_softmax(-1)
    l = -lp.gather(-1, Y[..., None])[..., 0].mean(1)                     # (S,)
    acc = (logits.argmax(-1) == Y).float().mean(1)
    return l, acc


opt = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), fused=True)

# checkpoint schedules
def emb_step(t):
    return t <= 50 or (t <= 1000 and t % 10 == 0) or t % 20 == 0
log2 = {0} | {2 ** i for i in range(20)}
def full_step(t):
    return t in log2 or t % args.full_every == 0 or t == args.steps

emb_steps, emb_list = [], []
full_steps, full_list = [], []
tr_loss = np.zeros((args.steps + 1, S)); tr_acc = np.zeros((args.steps + 1, S))
ev_steps, te_loss, te_acc = [], [], []
t0 = time.time()
for t in range(args.steps + 1):
    if emb_step(t):
        emb_steps.append(t); emb_list.append(params["W_E"].detach()[:, :P].float().cpu().numpy().copy())
    if full_step(t):
        full_steps.append(t)
        full_list.append({k: v.detach().cpu().numpy().copy() for k, v in params.items()})
    logits = forward(Xtr)
    l, acc = loss_acc(logits, Ytr)
    tr_loss[t] = l.detach().cpu().numpy(); tr_acc[t] = acc.cpu().numpy()
    if t % args.eval_every == 0:
        with torch.no_grad():
            lt, at = loss_acc(forward(Xte), Yte)
        ev_steps.append(t); te_loss.append(lt.cpu().numpy()); te_acc.append(at.cpu().numpy())
    if t % args.print_every == 0:
        print(f"step {t} {time.time()-t0:.0f}s train {tr_loss[t].round(4).tolist()} test {te_loss[-1].round(3).tolist()}"
              f" test_acc {te_acc[-1].round(3).tolist()}", flush=True)
    if t == args.steps:
        break
    opt.zero_grad(set_to_none=True)
    l.sum().backward()
    opt.step()

# the losses at index t are for params *before* update t, i.e. the params saved at checkpoint t
np.savez(f"{args.out}/metrics.npz", tr_loss=tr_loss, tr_acc=tr_acc, ev_steps=np.array(ev_steps),
         te_loss=np.stack(te_loss), te_acc=np.stack(te_acc), init_seeds=np.array(init_seeds),
         data_seeds=np.array(data_seeds), train_idx=tr_idx.numpy())
np.savez(f"{args.out}/emb.npz", steps=np.array(emb_steps), W_E=np.stack(emb_list))  # (T, S, P, D)
full = {k: np.stack([f[k] for f in full_list]) for k in shapes}
np.savez(f"{args.out}/full.npz", steps=np.array(full_steps), **full)
json.dump(dict(vars(args), wall_s=time.time() - t0), open(f"{args.out}/info.json", "w"))
print("done", time.time() - t0, flush=True)
