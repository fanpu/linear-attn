"""Post-hoc measurements for the toy run (CPU only, from cache/).

From cache/toy_<tag>.npz + cache/toy_<tag>_ckpt.pt:
  * offset spectrum per logged step: S[t, l, h, o] = mean_i a[i, i-o] over the
    repeated-random eval set (period 32), queries i >= 32   (the "loom record")
  * probe patterns on a short 64-token probe (period 16 x 4) at every checkpoint
  * Olsson copying score from the OV circuit W_U W_O W_V W_E (LayerNorm ignored)
  * phase-change step estimates

  python analyze_toy.py --tag main   -> cache/toy_<tag>_analysis.npz, cache/toy_<tag>_summary.json
"""
import argparse, json
import numpy as np
import torch
from toy_common import AttnOnly, V, T, H

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='main')
ap.add_argument('--copy_every', type=int, default=50)
args = ap.parse_args()
torch.set_num_threads(4)

log = np.load(f'cache/toy_{args.tag}.npz')
ck = torch.load(f'cache/toy_{args.tag}_ckpt.pt', map_location='cpu')
ckpts = ck['ckpts']
L = int(ck['args'].get('layers', 2))
steps = log['step']
P = int(log['period'])

# ---- offset spectrum ("loom record") from mean attention on repeated-random eval set
M = log['mean_attn_rep'].astype(np.float32)          # N, L, H, T, T
N = M.shape[0]
S = np.zeros((N, L, H, T), np.float32)
ii = np.arange(P, T)
for o in range(T):
    sel = ii[ii - o >= 0]
    S[:, :, :, o] = M[:, :, :, sel, sel - o].mean(-1)
sink = M[:, :, :, P:, 0].mean(-1)

# ---- probe patterns (period 16 x 4) at every checkpoint
g = torch.Generator().manual_seed(11)
seg = torch.randint(0, V, (1, 16), generator=g)
probe = seg.repeat(1, 4)
ck_steps = sorted(ckpts.keys())
model = AttnOnly(L=L)
probe_pats = np.zeros((len(ck_steps), L, H, 64, 64), np.float16)
copy_steps, copy_scores = [], []
for k, st in enumerate(ck_steps):
    sd = {kk: v.float() for kk, v in ckpts[st].items()}
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        _, pats = model(probe, return_patterns=True)
    probe_pats[k] = torch.stack([p[0] for p in pats]).numpy()
    if st % args.copy_every == 0:
        WE, WU = sd['emb.weight'].double(), sd['unemb.weight'].double()
        dh = model.attn[0].dh
        cs = np.zeros((L, H))
        for li in range(L):
            for h in range(H):
                WV = sd[f'attn.{li}.v.weight'][h * dh:(h + 1) * dh].double()
                WO = sd[f'attn.{li}.o.weight'][:, h * dh:(h + 1) * dh].double()
                ev = torch.linalg.eigvals(WU @ WO @ WV @ WE.T)
                cs[li, h] = (ev.real.sum() / ev.abs().sum()).item()
        copy_steps.append(st)
        copy_scores.append(cs)

# ---- phase change estimates
rep = log['loss_rep_later']
ind = log['ind_score']                                # N, L, H
lastL = ind[:, L - 1] if L > 1 else ind[:, 0]
best_head = int(np.argmax(lastL[-1]))
drop = np.diff(rep)
k_fast = int(np.argmin(np.convolve(drop, np.ones(10) / 10, mode='same')))
lo, hi = rep[:50].mean(), rep[-50:].mean()
k_half = int(np.argmax(rep < (lo + hi) / 2))
k_10 = int(np.argmax(rep < lo - 0.1 * (lo - hi)))
k_90 = int(np.argmax(rep < lo - 0.9 * (lo - hi)))
summary = dict(
    tag=args.tag, layers=L, period=P,
    phase_step_half_drop=int(steps[k_half]), phase_step_10pct=int(steps[k_10]), phase_step_90pct=int(steps[k_90]),
    phase_step_steepest=int(steps[k_fast]),
    loss_rep_later_before=float(lo), loss_rep_later_after=float(hi),
    icl_score_before=float(log['icl_score'][max(0, k_10 - 20):k_10].mean()),
    icl_score_final=float(log['icl_score'][-20:].mean()),
    loss_markov_copied_before=float(log['loss_markov_copied'][max(0, k_10 - 20):k_10].mean()),
    loss_markov_copied_final=float(log['loss_markov_copied'][-20:].mean()),
    loss_markov_fresh_final=float(log['loss_markov_fresh'][-20:].mean()),
    final_ind_score=log['ind_score'][-1].round(3).tolist(),
    final_prev_score=log['prev_score'][-1].round(3).tolist(),
    final_pm_score=log['pm_score'][-1].round(3).tolist(),
    pm_uniform_baseline=float(log['pm_baseline'][-1][0][0]),
    final_ablate_dloss=log['ablate_dloss'][-1].round(3).tolist(),
    final_copy_score=np.array(copy_scores[-1]).round(3).tolist(),
    induction_head=[L - 1, best_head],
    max_prev_score_layer0_final=float(log['prev_score'][-1][0].max()),
)
print(json.dumps(summary, indent=1))
json.dump(summary, open(f'cache/toy_{args.tag}_summary.json', 'w'), indent=1)
np.savez_compressed(f'cache/toy_{args.tag}_analysis.npz', offset_spectrum=S, sink=sink, probe_steps=np.array(ck_steps),
                    probe_pats=probe_pats, probe_tokens=probe.numpy(), copy_steps=np.array(copy_steps),
                    copy_scores=np.array(copy_scores))
