"""Spectroscope: train a small GPT on TinyStories and record the eigenvalue spectrum of every weight
matrix (and of two entry-shuffled copies of it) at ~800 log-spaced checkpoints.

    .venv/bin/python spectroscope/train_gpt.py --opt adamw --lr 1e-3 --steps 6000 --name adamw

Optimisers:
  adamw : AdamW on every parameter (decay on 2-D params only).
  muon  : Muon (momentum 0.95, Nesterov, Newton-Schulz-5 orthogonalisation, update scaled by
          sqrt(max(1, rows/cols))) on the 2-D hidden matrices (q, k, v, o, up, down); AdamW at
          --adam-lr on embedding, position embedding, head and LayerNorm gains, as is standard.
  sgd   : SGD with Nesterov momentum 0.9 on every parameter.
  null  : AdamW is run on the model's own gradients to get each parameter's update, then the update
          is replaced by a Gaussian random direction with the SAME Frobenius norm. Moves the weights
          exactly as far as AdamW would, in no particular direction.
All runs: batch 64 x 256 tokens, bf16 autocast, grad-norm clip 1.0, 100-step linear warmup then
cosine decay to 0.1x. --data shuffled permutes the training token stream once (unigram-only data).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pasar_job

p = argparse.ArgumentParser()
p.add_argument('--opt', required=True, choices=['adamw', 'muon', 'sgd', 'null'])
p.add_argument('--lr', type=float, required=True)
p.add_argument('--adam-lr', type=float, default=1e-3)
p.add_argument('--wd', type=float, default=0.0)
p.add_argument('--steps', type=int, default=6000)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--data', default='real', choices=['real', 'shuffled'])
p.add_argument('--nckpt', type=int, default=800)
p.add_argument('--no-spectra', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--name', required=True)
p.add_argument('--compile', type=int, default=1)
p.add_argument('--ckpt-every', type=int, default=2500, help='training-state checkpoint interval (resume after preemption)')
p.add_argument('--stop-at', type=int, default=-1, help='testing only: exit right after the checkpoint at this step')
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
# CPU eigen workers, forked BEFORE CUDA initialises (spawn would re-run this script in each child)
import concurrent.futures as cf, multiprocessing as mproc, eigworker
pool = cf.ProcessPoolExecutor(max_workers=8, mp_context=mproc.get_context('fork'), initializer=eigworker.init)
for _f in [pool.submit(eigworker.init) for _ in range(12)]:
    _f.result()
OUT = os.path.join(HERE, 'cache', 'runs')
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
pasar_job.apply_memory_limit()
torch.manual_seed(args.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True

V, T, D, H, L = 8192, 256, 384, 6, 6
HD = D // H

# ---------------- data
tr = torch.from_numpy(np.load(os.path.join(HERE, 'cache', 'train.npy')).astype(np.int32)).to(dev)
va = torch.from_numpy(np.load(os.path.join(HERE, 'cache', 'val.npy')).astype(np.int32)).to(dev)
if args.data == 'shuffled':
    g = torch.Generator(device=dev); g.manual_seed(1234)
    tr = tr[torch.randperm(len(tr), device=dev, generator=g)]
ar = torch.arange(T + 1, device=dev)


def batch(src, bs, gen=None):
    i = torch.randint(0, len(src) - T - 1, (bs,), device=dev, generator=gen)
    x = src[i[:, None] + ar].long()
    return x[:, :-1], x[:, 1:]


# ---------------- model
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.up, s.down = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(s, x):
        B = x.shape[0]
        h = s.ln1(x)
        q, k, v = (m(h).view(B, T, H, HD).transpose(1, 2) for m in (s.q, s.k, s.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + s.o(a)
        return x + s.down(F.gelu(s.up(s.ln2(x))))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.ModuleList(Block() for _ in range(L))
        s.lnf, s.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)
        for n, w in s.named_parameters():
            if w.dim() == 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(('o.weight', 'down.weight')) else 0.02
                nn.init.normal_(w, 0, std)

    def forward(s, idx, tgt):
        x = s.emb(idx) + s.pos.weight[None]
        for b in s.blocks:
            x = b(x)
        return F.cross_entropy(s.head(s.lnf(x)).float().view(-1, V), tgt.reshape(-1))


model = GPT().to(dev)
fwd = torch.compile(model) if args.compile else model

# ---------------- optimisers
def ns5(G, steps=5):
    """Newton-Schulz quintic iteration (Jordan et al.): approx U V^T of G = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr_ = X.shape[0] > X.shape[1]
    if tr_:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tr_ else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr, momentum=0.95, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(s):
        for gr in s.param_groups:
            for w in gr['params']:
                if w.grad is None:
                    continue
                st = s.state[w]
                if 'buf' not in st:
                    st['buf'] = torch.zeros_like(w)
                buf = st['buf']
                buf.mul_(gr['momentum']).add_(w.grad)
                u = ns5(w.grad + gr['momentum'] * buf)
                w.mul_(1 - gr['lr'] * gr['wd'])
                w.add_(u, alpha=-gr['lr'] * max(1.0, w.shape[0] / w.shape[1]) ** 0.5)


named = dict(model.named_parameters())
hidden = [w for n, w in named.items() if w.dim() == 2 and n.startswith('blocks.')]
other2d = [w for n, w in named.items() if w.dim() == 2 and not n.startswith('blocks.')]
onedim = [w for n, w in named.items() if w.dim() < 2]
if args.opt in ('adamw', 'null'):
    opts = [torch.optim.AdamW([dict(params=hidden + other2d, weight_decay=args.wd),
                               dict(params=onedim, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.95))]
elif args.opt == 'muon':
    opts = [Muon(hidden, lr=args.lr, wd=args.wd),
            torch.optim.AdamW([dict(params=other2d, weight_decay=args.wd * args.lr / args.adam_lr if args.wd else 0.0),
                               dict(params=onedim, weight_decay=0.0)], lr=args.adam_lr, betas=(0.9, 0.95))]
else:
    opts = [torch.optim.SGD([dict(params=hidden + other2d, weight_decay=0.0),
                             dict(params=onedim, weight_decay=0.0)], lr=args.lr, momentum=0.9, nesterov=True)]
for o in opts:
    for gr in o.param_groups:
        gr['base_lr'] = gr['lr']


def lr_mult(step):
    if step < 100:
        return (step + 1) / 100
    t = (step - 100) / max(1, args.steps - 100)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


# ---------------- spectra
def mats():
    """(name, matrix) in the declared row order: per layer q,k,v,o, per-head q/k/v/o, up, down; then emb, pos, head."""
    out = []
    for i, b in enumerate(model.blocks):
        for t in 'qkvo':
            out.append((f'L{i}.{t}', getattr(b, t).weight))
        for t in 'qkvo':
            W = getattr(b, t).weight
            for h in range(H):
                out.append((f'L{i}.{t}.h{h}', W[h * HD:(h + 1) * HD, :] if t != 'o' else W[:, h * HD:(h + 1) * HD]))
        out.append((f'L{i}.up', b.up.weight))
        out.append((f'L{i}.down', b.down.weight))
    out += [('emb', model.emb.weight), ('pos', model.pos.weight), ('head', model.head.weight)]
    return out


NAMES = [n for n, _ in mats()]


WHOLE = [(n, W) for n, W in mats() if '.h' not in n]


@torch.no_grad()
def snapshot():
    """All recorded matrices as float32 numpy: one GPU->CPU copy of the whole matrices, then the
    per-head slices are cut on the CPU."""
    flat = torch.cat([W.detach().float().reshape(-1) for _, W in WHOLE]).cpu().numpy()
    whole, o = {}, 0
    for n, W in WHOLE:
        whole[n] = flat[o:o + W.numel()].reshape(W.shape); o += W.numel()
    out = []
    for n in NAMES:
        if '.h' in n:
            base, h = n.rsplit('.h', 1); h = int(h)
            W = whole[base]
            out.append((n, np.ascontiguousarray(W[:, h * HD:(h + 1) * HD] if base.endswith('.o') else W[h * HD:(h + 1) * HD])))
        else:
            out.append((n, whole[n]))
    return out


def ckpt_steps(S, target):
    K = target
    while True:
        s = np.unique(np.round(np.geomspace(1, S, K)).astype(int))
        if len(s) + 1 >= target or K > 50 * target:
            return np.concatenate([[0], s])
        K = int(K * 1.05) + 1


CK = ckpt_steps(args.steps, args.nckpt) if not args.no_spectra else np.array([], int)
CKset = set(CK.tolist())
EV = np.unique(np.concatenate([[0], np.round(np.geomspace(1, args.steps, 12)).astype(int)]))
futs = []
rec_steps, rec_loss, val_log = [], [], []


@torch.no_grad()
def val_loss(nb=20):
    g = torch.Generator(device=dev); g.manual_seed(7)
    model.eval()
    tot = 0.0
    for _ in range(nb):
        x, y = batch(va, 64, g)
        with torch.autocast('cuda', torch.bfloat16):
            tot += fwd(x, y).item()
    model.train()
    return tot / nb


t0 = time.time(); tspec = 0.0; loss_ema = None
done = {}                      # checkpoint index -> spectra dict, drained from futures at each state checkpoint
CKPT = os.path.join(HERE, 'cache', 'ckpt', args.name + '.pt')
start = 0
if os.path.exists(CKPT) and (pasar_job.resuming() or os.environ.get('SPEC_RESUME')):
    st = torch.load(CKPT, weights_only=False)
    model.load_state_dict(st['model'])
    for o, sd in zip(opts, st['opts']):
        o.load_state_dict(sd)
    torch.set_rng_state(st['cpu_rng']); torch.cuda.set_rng_state(st['cuda_rng'])
    rec_steps, rec_loss, val_log, loss_ema, done, tspec = st['rec_steps'], st['rec_loss'], st['val_log'], st['loss_ema'], st['done'], st['tspec']
    start = st['step'] + 1
    print(f'resumed from step {st["step"]} ({len(done)} spectral checkpoints already done)', flush=True)
    pasar_job.resumed(start)


def save_state(step):
    for f_ in futs:
        ci, sp = f_.result()
        done[ci] = sp
    futs.clear()
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(dict(step=step, model=model.state_dict(), opts=[o.state_dict() for o in opts], cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), rec_steps=rec_steps, rec_loss=rec_loss, val_log=val_log,
                    loss_ema=loss_ema, done=done, tspec=tspec), CKPT + '.tmp')
    os.replace(CKPT + '.tmp', CKPT)
    pasar_job.checkpoint(step)
    print(f'state checkpoint at step {step}', flush=True)


print(f'{args.name}: {len(CK)} spectral checkpoints, {len(NAMES)} matrices', flush=True)
for step in range(start, args.steps + 1):
    if step in CKset:
        ts = time.time()
        futs.append(pool.submit(eigworker.eig, len(rec_steps), snapshot(), 1000003 * args.seed + step))
        while sum(not f_.done() for f_ in futs) > 24:
            time.sleep(0.02)
        rec_steps.append(step); rec_loss.append(loss_ema if loss_ema is not None else float('nan'))
        tspec += time.time() - ts
    if step in EV or step == args.steps:
        vl = val_loss()
        val_log.append((step, vl))
        print(f'step {step} val {vl:.4f} train_ema {loss_ema} elapsed {time.time()-t0:.0f}s spec {tspec:.0f}s', flush=True)
        pasar_job.progress(step, args.steps, val=vl)
    if step == args.steps:
        break
    x, y = batch(tr, args.bs)
    with torch.autocast('cuda', torch.bfloat16):
        loss = fwd(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    m = lr_mult(step)
    for o in opts:
        for gr in o.param_groups:
            gr['lr'] = gr['base_lr'] * m
    if args.opt == 'null':
        with torch.no_grad():
            old = [w.detach().clone() for w in model.parameters()]
        opts[0].step()
        with torch.no_grad():
            for w, w0 in zip(model.parameters(), old):
                d = w - w0
                r = torch.randn_like(w)
                w.copy_(w0 + r * (d.norm() / (r.norm() + 1e-30)))
    else:
        for o in opts:
            o.step()
    for o in opts:
        o.zero_grad(set_to_none=True)
    li = loss.item() if step % 10 == 0 else None
    if li is not None:
        loss_ema = li if loss_ema is None else 0.9 * loss_ema + 0.1 * li
    if step % 200 == 0:
        pasar_job.progress(step, args.steps, loss=loss_ema)
    if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
        save_state(step)
        if step == args.stop_at:
            os._exit(3)

tw = time.time()
rec = {n: ([None] * len(rec_steps), [None] * len(rec_steps), [None] * len(rec_steps)) for n in NAMES}
for f_ in futs:
    ci, sp = f_.result()
    done[ci] = sp
for ci, sp in done.items():
    for n in NAMES:
        for k in range(3):
            rec[n][k][ci] = sp[n][k]
pool.shutdown()
print(f'waited {time.time()-tw:.0f}s for eigen workers', flush=True)
wall = time.time() - t0
print(f'done {args.name}: final val {val_log[-1][1]:.4f}, wall {wall:.0f}s, spectra {tspec:.0f}s', flush=True)
meta = dict(vars(args), names=NAMES, val_log=val_log, wall=wall, spectra_time=tspec,
            job=os.environ.get('PASAR_JOB_ID'), final_val=val_log[-1][1])
if not args.no_spectra:
    arrs = {}
    for n in NAMES:
        for k, tag in enumerate(('lam', 's1', 's2')):
            arrs[f'{n}/{tag}'] = np.stack(rec[n][k]).astype(np.float32)
    np.savez(os.path.join(OUT, args.name + '.npz'), steps=np.array(rec_steps), train_loss=np.array(rec_loss, dtype=float), **arrs)
json.dump(meta, open(os.path.join(OUT, args.name + '.json'), 'w'), indent=1)
