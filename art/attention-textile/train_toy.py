"""Part A compute: train a 2-layer attention-only transformer and log, densely,
everything needed to watch induction heads form.

Writes cache/toy_<tag>.npz (metrics + probe attention patterns every LOG steps)
and cache/toy_<tag>_ckpt.pt (weights every CKPT steps).

  python train_toy.py --steps 6000 --tag main
"""
import argparse, time, json, os
import numpy as np
import torch
import torch.nn.functional as F
from toy_common import AttnOnly, make_bigram, markov_batch, insert_copies, repeated_random, V, T, L, H

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--frac_rep', type=float, default=0.25, help='fraction of batch that is repeated-random')
ap.add_argument('--log', type=int, default=5)
ap.add_argument('--ckpt', type=int, default=20)
ap.add_argument('--ablate_every', type=int, default=25)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='main')
args = ap.parse_args()

dev = 'cuda'
torch.cuda.set_per_process_memory_fraction(0.10)
torch.manual_seed(args.seed)
os.makedirs('cache', exist_ok=True)

Pbig = make_bigram(seed=1234, device=dev)
model = AttnOnly().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
gtrain = torch.Generator(device=dev).manual_seed(args.seed + 1)

# ---------------- fixed evaluation sets ----------------
geval = torch.Generator(device=dev).manual_seed(999)
EV_M2 = markov_batch(Pbig, 512, geval, dev)
EV_M, EV_PRED = insert_copies(EV_M2, geval, n_copies=1)
PERIOD = 32
EV_R = repeated_random(256, PERIOD, geval, dev)             # 4 repeats of 32
PROBE = repeated_random(1, PERIOD, torch.Generator(device=dev).manual_seed(7), dev)

# prefix-matching mask on Markov eval data: M[b,i,j] = (x[j-1]==x[i]) & (j<=i) & j>=1
xm = EV_M
pm_mask = torch.zeros(xm.shape[0], T, T, dtype=torch.bool, device=dev)
pm_mask[:, :, 1:] = xm[:, :, None] == xm[:, None, :-1]
pm_mask &= torch.ones(T, T, dtype=torch.bool, device=dev).tril()[None]
has_pm = pm_mask[:, :, 1:].any(-1)           # rows with at least one prefix match
pm_mask_f = pm_mask.float()

idx = torch.arange(T, device=dev)


def per_pos_loss(logits, x):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1),
                           reduction='none').view(x.shape[0], T - 1)


@torch.no_grad()
def evaluate(step, do_ablate):
    model.eval()
    out = {}
    lm, pats_m = model(EV_M, return_patterns=True)
    lr_, pats_r = model(EV_R, return_patterns=True)
    plm = per_pos_loss(lm, EV_M)
    plr = per_pos_loss(lr_, EV_R)
    out['loss_pos_markov'] = plm.mean(0).cpu().numpy()
    out['loss_pos_rep'] = plr.mean(0).cpu().numpy()
    tgt_pred = EV_PRED[:, 1:]
    out['loss_markov'] = plm.mean().item()
    out['loss_markov_copied'] = plm[tgt_pred].mean().item()
    out['loss_markov_fresh'] = plm[~tgt_pred].mean().item()
    out['loss_rep_first'] = plr[:, :PERIOD - 1].mean().item()
    out['loss_rep_later'] = plr[:, PERIOD:].mean().item()
    # Olsson-style ICL score: late-context loss minus early-context loss
    out['icl_score'] = (plm[:, 99:120].mean() - plm[:, 7:16].mean()).item()
    prev, ind, pm, pm_base = [], [], [], []
    for li in range(L):
        am, ar = pats_m[li], pats_r[li]
        # previous-token score on Markov data: mean a[i,i-1], i>=1
        prev.append(am[:, :, idx[1:], idx[:-1]].mean((0, 2)))
        # induction (stripe) score on repeated random: a[i, i-P+1], i>=P
        ii = idx[PERIOD:]
        ind.append(ar[:, :, ii, ii - PERIOD + 1].mean((0, 2)))
        # Olsson prefix-matching on Markov data: attention mass on {j: x[j-1]==x[i]}
        mass = (am * pm_mask_f[:, None]).sum(-1)            # B,h,T
        sel = has_pm[:, None, :].expand_as(mass)
        pm.append((mass * sel).sum((0, 2)) / sel.sum((0, 2)))
        # uniform-attention baseline for the same quantity
        base = (pm_mask_f.sum(-1) / (idx + 1).float())[has_pm].mean()
        pm_base.append(base.expand(H))
    out['prev_score'] = torch.stack(prev).cpu().numpy()
    out['ind_score'] = torch.stack(ind).cpu().numpy()
    out['pm_score'] = torch.stack(pm).cpu().numpy()
    out['pm_baseline'] = torch.stack(pm_base).cpu().numpy()
    # Olsson copying score from the direct OV circuit W_U W_O W_V W_E (LayerNorm ignored)
    WE = model.emb.weight            # V,D
    WU = model.unemb.weight          # V,D
    cps = np.zeros((L, H))
    dh = model.attn[0].dh
    for li in range(L):
        at = model.attn[li]
        for h in range(H):
            WV = at.v.weight[h * dh:(h + 1) * dh]                 # dh,D
            WO = at.o.weight[:, h * dh:(h + 1) * dh]              # D,dh
            M = (WU @ WO @ WV @ WE.T).double()                    # V,V
            ev = torch.linalg.eigvals(M)
            cps[li, h] = (ev.real.sum() / ev.abs().sum()).item()
    out['copy_score'] = cps
    # probe attention pattern (single repeated-random sequence)
    _, pp = model(PROBE, return_patterns=True)
    out['probe_attn'] = torch.stack([p[0] for p in pp]).half().cpu().numpy()   # L,H,T,T
    # mean pattern over the repeated-random eval set (denoised view)
    out['mean_attn_rep'] = torch.stack([p.mean(0) for p in pats_r]).half().cpu().numpy()
    if do_ablate:
        base = out['loss_rep_later']
        abl = np.zeros((L, H))
        for li in range(L):
            for h in range(H):
                la = model(EV_R, ablate=[(li, h)])
                abl[li, h] = per_pos_loss(la, EV_R)[:, PERIOD:].mean().item() - base
        out['ablate_dloss'] = abl
    model.train()
    return out


log = {}
ckpts = {}
t0 = time.time()
step_loss = np.zeros(args.steps)
last_abl = np.zeros((L, H))
for step in range(args.steps + 1):
    if step % args.log == 0:
        do_ab = step % args.ablate_every == 0
        ev = evaluate(step, do_ab)
        if do_ab:
            last_abl = ev['ablate_dloss']
        ev['ablate_dloss'] = last_abl
        ev['step'] = step
        for k, v in ev.items():
            log.setdefault(k, []).append(v)
        if step % (args.log * 40) == 0:
            print(f"step {step:6d} markov {ev['loss_markov']:.3f} copied {ev['loss_markov_copied']:.3f} "
                  f"rep_later {ev['loss_rep_later']:.3f} icl {ev['icl_score']:.3f} "
                  f"ind_max {ev['ind_score'].max():.3f} prev_max {ev['prev_score'].max():.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    if step % args.ckpt == 0:
        ckpts[step] = {k: v.detach().half().cpu().clone() for k, v in model.state_dict().items()}
    if step == args.steps:
        break
    nrep = int(args.bs * args.frac_rep)
    xm = markov_batch(Pbig, args.bs - nrep, gtrain, dev)
    xm, _ = insert_copies(xm, gtrain, n_copies=2)
    periods = torch.randint(6, 49, (1,), generator=gtrain, device=dev).item()
    xr = repeated_random(nrep, periods, gtrain, dev)
    x = torch.cat([xm, xr])
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    step_loss[step] = loss.item()

arrs = {k: np.array(v) for k, v in log.items()}
arrs['train_loss'] = step_loss
arrs['probe_tokens'] = PROBE.cpu().numpy()
arrs['period'] = PERIOD
np.savez_compressed(f'cache/toy_{args.tag}.npz', **arrs)
torch.save({'ckpts': ckpts, 'args': vars(args)}, f'cache/toy_{args.tag}_ckpt.pt')
json.dump({'args': vars(args), 'wall_s': time.time() - t0}, open(f'cache/toy_{args.tag}_meta.json', 'w'))
print('done', time.time() - t0)
