"""Measure everything the renders need, from the training caches.  Writes cache/analysis.npz.

Measured quantities (all seeds, S = 12):
  spec_E   (T_emb, S, 57)   ||DFT of W_E along token axis||_2 over d_model, per frequency k=0..56
                           (k>0 combines the cos and sin components)
  keys     per seed         key frequencies (from the final neuron-logit map W_L = W_out W_U, cross-checked with W_E)
  ring coords (T_emb, S, K, P, 2): W_E(t) projected onto the final 2D plane span(u_k, v_k) of each key freq k
  R_k(t)                   ring order parameter |sum_a z_a e^{-i s 2 pi k a/p}| / sqrt(P sum_a |z_a|^2)
  progress measures at full checkpoints: train/test loss, restricted loss, excluded loss, Gini, ||theta||^2
  final MLP neuron activations over the full (a,b) grid, and argmax / correct-logprob tables over training
"""
import glob, json, math, time
import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.10)
dev = "cuda"
P, D, H, DH, M = 113, 128, 4, 32, 512
t0 = time.time()


def load(name):
    runs = [np.load(f"cache/{r}/{name}.npz") for r in ("runA", "runB")]
    return runs


# ---------------------------------------------------------------- metrics
mets = load("metrics")
tr_loss = np.concatenate([m["tr_loss"] for m in mets], 1)
tr_acc = np.concatenate([m["tr_acc"] for m in mets], 1)
ev_steps = mets[0]["ev_steps"]
te_loss = np.concatenate([m["te_loss"] for m in mets], 1)
te_acc = np.concatenate([m["te_acc"] for m in mets], 1)
init_seeds = np.concatenate([m["init_seeds"] for m in mets])
data_seeds = np.concatenate([m["data_seeds"] for m in mets])
train_idx = np.concatenate([m["train_idx"] for m in mets], 0)
S = len(init_seeds)

embs = load("emb")
emb_steps = embs[0]["steps"]
W_E = np.concatenate([e["W_E"] for e in embs], 1).astype(np.float64)   # (T, S, P, D)
fulls = load("full")
full_steps = fulls[0]["steps"]
full = {k: np.concatenate([f[k] for f in fulls], 1) for k in fulls[0].files if k != "steps"}
print("loaded", W_E.shape, len(full_steps), time.time() - t0, flush=True)

# ---------------------------------------------------------------- Fourier basis
a = np.arange(P)
F = [np.ones(P) / math.sqrt(P)]
for k in range(1, P // 2 + 1):
    F.append(np.cos(2 * np.pi * k * a / P)); F[-1] /= np.linalg.norm(F[-1])
    F.append(np.sin(2 * np.pi * k * a / P)); F[-1] /= np.linalg.norm(F[-1])
F = np.stack(F)                                                         # (113, 113) orthonormal rows
assert np.allclose(F @ F.T, np.eye(P))


def freq_norms(coef):  # coef (..., 113) squared -> (..., 57)
    sq = coef ** 2
    out = np.empty(sq.shape[:-1] + (P // 2 + 1,))
    out[..., 0] = sq[..., 0]
    out[..., 1:] = sq[..., 1::2] + sq[..., 2::2]
    return np.sqrt(out)


# spectrum of W_E over training: coefficient tensor (T,S,113,D) -> norm over D -> (T,S,113) -> pair
C_E = np.einsum("fa,tsad->tsfd", F, W_E)
spec_E = freq_norms(np.linalg.norm(C_E, axis=-1))                      # (T,S,57)


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    return 1 - 2 * np.sum((n - np.arange(1, n + 1) + 0.5) * x) / (n * x.sum())


# neuron-logit map W_L = W_out @ W_U  (M, P): Fourier along output axis
W_L_final = full["W_out"][-1] @ full["W_U"][-1]                         # (S, M, P)
spec_L = freq_norms(np.linalg.norm(np.einsum("smc,fc->smf", W_L_final, F), axis=1))  # (S,57)
spec_E_final = spec_E[-1]


def pick_keys(spec):
    """Key frequencies = those k>0 whose norm is > 20% of the largest (declared threshold)."""
    s = spec[1:]
    return [int(k + 1) for k in np.argsort(-s) if s[k] > 0.2 * s.max()]


keys = []
for s in range(S):
    kL = pick_keys(spec_L[s]); kE = pick_keys(spec_E_final[s])
    keys.append(sorted(kL))
    print(f"seed {init_seeds[s]} data {data_seeds[s]}: keys(W_L)={sorted(kL)} keys(W_E)={sorted(kE)}", flush=True)
Kmax = max(len(k) for k in keys)

# ---------------------------------------------------------------- ring projections
T = len(emb_steps)
ring = np.full((T, S, Kmax, P, 2), np.nan)
R = np.full((T, S, Kmax), np.nan)
plane_frac = np.full((S, Kmax), np.nan)   # fraction of final ||W_E||^2 in each key plane
aniso = np.full((S, Kmax), np.nan)        # ratio of singular values of final ring (1 = circle)
orient = np.zeros((S, Kmax))              # +1 / -1 reflection chosen
for s in range(S):
    WEf = W_E[-1, s]                                                    # (P, D) final
    for j, k in enumerate(keys[s]):
        u = C_E[-1, s, 2 * k - 1]; v = C_E[-1, s, 2 * k]                # D-vectors
        Q, _ = np.linalg.qr(np.stack([u, v], 1))                         # (D,2) orthonormal
        z = W_E[:, s] @ Q                                                # (T,P,2)
        zc = z[..., 0] + 1j * z[..., 1]
        th = 2 * np.pi * k * a / P
        # choose reflection + rotation so final token a sits at angle ~ 2 pi k a / p  (declared framing)
        best = None
        for sg in (1, -1):
            zz = zc if sg == 1 else np.conj(zc)
            c = np.sum(zz[-1] * np.exp(-1j * th))
            if best is None or abs(c) > abs(best[1]):
                best = (sg, c)
        sg, c = best
        zz = zc if sg == 1 else np.conj(zc)
        zz = zz * np.exp(-1j * np.angle(c))
        ring[:, s, j, :, 0] = zz.real; ring[:, s, j, :, 1] = zz.imag
        R[:, s, j] = np.abs(np.sum(zz * np.exp(-1j * th)[None], 1)) / np.sqrt(P * np.sum(np.abs(zz) ** 2, 1))
        orient[s, j] = sg
        plane_frac[s, j] = np.sum((WEf @ Q) ** 2) / np.sum(WEf ** 2)
        sv = np.linalg.svd((WEf - WEf.mean(0)) @ Q, compute_uv=False)
        aniso[s, j] = sv[1] / sv[0]
    print(f"seed {s}: R_final={np.round(R[-1, s, :len(keys[s])], 3)} plane_frac={np.round(plane_frac[s, :len(keys[s])], 3)}"
          f" aniso={np.round(aniso[s, :len(keys[s])], 2)}", flush=True)

# ---------------------------------------------------------------- forward passes on full grid (GPU)
causal = torch.tril(torch.ones(3, 3, dtype=torch.bool, device=dev))
aa = torch.arange(P).repeat_interleave(P); bb = torch.arange(P).repeat(P)
X_all = torch.stack([aa, bb, torch.full_like(aa, P)], 1).to(dev)       # (P^2, 3)
Y_all = ((aa + bb) % P).to(dev)


def forward_one(pr, X, want_h=False):
    x = pr["W_E"][X] + pr["W_pos"][None]                                 # B 3 D
    q = torch.einsum("bpd,hde->bhpe", x, pr["W_Q"]); k = torch.einsum("bpd,hde->bhpe", x, pr["W_K"])
    v = torch.einsum("bpd,hde->bhpe", x, pr["W_V"])
    att = (q @ k.transpose(-1, -2) / math.sqrt(DH)).masked_fill(~causal, float("-inf")).softmax(-1)
    z = (att @ v)[:, :, -1]                                              # B H DH
    r = x[:, -1] + torch.einsum("bhe,hed->bd", z, pr["W_O"])
    h = torch.relu(r @ pr["W_in"])
    r = r + h @ pr["W_out"]
    lg = (r @ pr["W_U"]).double()
    return (lg, h) if want_h else lg


Fg = torch.tensor(F, device=dev)
Tf = len(full_steps)
restricted = np.zeros((Tf, S)); excluded = np.zeros((Tf, S)); gini_E = np.zeros((Tf, S)); gini_L = np.zeros((Tf, S))
sumsq = np.zeros((Tf, S)); full_loss = np.zeros((Tf, S)); full_tr = np.zeros((Tf, S))
pred_tab = np.zeros((Tf, S, P, P), np.uint8)
lp_tab = np.zeros((Tf, S, P, P), np.float16)
MAIN_NEURONS = 64
acts_final = np.zeros((S, P, P, M), np.float16)
for s in range(S):
    tr_mask = torch.zeros(P * P, dtype=torch.bool, device=dev); tr_mask[torch.tensor(train_idx[s], device=dev)] = True
    key_idx = sorted(sum([[2 * k - 1, 2 * k] for k in keys[s]], []))
    keep = torch.zeros(P, P, dtype=torch.bool, device=dev); keep[0, 0] = True
    for i in key_idx:
        for j in key_idx:
            # only same-frequency 2x2 blocks (cos/sin a x cos/sin b), 4 terms per key frequency
            if (i + 1) // 2 == (j + 1) // 2:
                keep[i, j] = True
    excl_mask = keep.clone(); excl_mask[0, 0] = False
    for ti in range(Tf):
        pr = {kk: torch.tensor(full[kk][ti, s], device=dev) for kk in full}
        lg, h = forward_one(pr, X_all, want_h=True)
        L = lg.view(P, P, P)
        lp = L.log_softmax(-1)
        corr = lp.gather(-1, Y_all.view(P, P, 1))[..., 0]
        full_loss[ti, s] = -corr.mean().item(); full_tr[ti, s] = -corr.view(-1)[tr_mask].mean().item()
        pred_tab[ti, s] = L.argmax(-1).cpu().numpy().astype(np.uint8)
        lp_tab[ti, s] = corr.float().cpu().numpy().astype(np.float16)
        C = torch.einsum("ia,abc,jb->ijc", Fg, L, Fg)
        Lr = torch.einsum("ia,ijc,jb->abc", Fg, C * keep[..., None], Fg)
        Le = torch.einsum("ia,ijc,jb->abc", Fg, C * (~excl_mask)[..., None], Fg)
        restricted[ti, s] = -Lr.log_softmax(-1).gather(-1, Y_all.view(P, P, 1)).mean().item()
        excluded[ti, s] = -Le.log_softmax(-1).gather(-1, Y_all.view(P, P, 1))[..., 0].view(-1)[tr_mask].mean().item()
        WL = (pr["W_out"] @ pr["W_U"]).double().cpu().numpy()
        cE = freq_norms(np.linalg.norm(F @ full["W_E"][ti, s, :P].astype(np.float64), axis=1))
        cL = freq_norms(np.linalg.norm(WL @ F.T, axis=0))
        gini_E[ti, s] = gini(cE[1:]); gini_L[ti, s] = gini(cL[1:])
        sumsq[ti, s] = sum(float((full[kk][ti, s].astype(np.float64) ** 2).sum()) for kk in full)
        if ti == Tf - 1:
            acts_final[s] = h.view(P, P, M).cpu().numpy().astype(np.float16)
    print(f"seed {s} progress done {time.time()-t0:.0f}s  final full-grid loss {full_loss[-1, s]:.2e} restricted {restricted[-1, s]:.2e}"
          f" excluded {excluded[-1, s]:.2f}", flush=True)

# neuron frequency clustering at the final checkpoint: dominant 2D frequency of each neuron's (a,b) activation
acts_c = torch.tensor(acts_final, device=dev, dtype=torch.float64)
C2 = torch.einsum("ia,sabm,jb->sijm", Fg, acts_c, Fg)                   # (S,113,113,M)
pw = (C2 ** 2).cpu().numpy()
# power per frequency for the "a-only", "b-only" and product blocks; assign neuron to k maximizing total power
nfreq = P // 2 + 1
neuron_k = np.zeros((S, M), int); neuron_frac = np.zeros((S, M)); neuron_var = np.zeros((S, M))
for s in range(S):
    tot = pw[s, 1:, :, :].sum((0, 1)) + pw[s, 0, 1:, :].sum(0)           # all non-DC power
    per_k = np.zeros((nfreq, M))
    for k in range(1, nfreq):
        idx = [2 * k - 1, 2 * k]
        blk = pw[s][np.ix_([0] + idx, [0] + idx)].sum((0, 1)) - pw[s, 0, 0]
        per_k[k] = blk
    neuron_k[s] = per_k.argmax(0)
    neuron_frac[s] = per_k.max(0) / np.maximum(tot, 1e-30)
    neuron_var[s] = tot
    frac_in_keys = np.mean([neuron_k[s, m] in keys[s] and neuron_frac[s, m] > 0.85 for m in range(M) if neuron_var[s, m] > 0])
    print(f"seed {s}: fraction of live neurons with >85% non-DC power in one key frequency: {frac_in_keys:.3f}", flush=True)

np.savez_compressed(
    "cache/analysis.npz",
    P=P, init_seeds=init_seeds, data_seeds=data_seeds, tr_loss=tr_loss, tr_acc=tr_acc, ev_steps=ev_steps,
    te_loss=te_loss, te_acc=te_acc, emb_steps=emb_steps, spec_E=spec_E, spec_L=spec_L, full_steps=full_steps,
    keys=np.array([k + [0] * (Kmax - len(k)) for k in keys]), nkeys=np.array([len(k) for k in keys]),
    ring=ring.astype(np.float32), R=R, plane_frac=plane_frac, aniso=aniso, orient=orient,
    restricted=restricted, excluded=excluded, gini_E=gini_E, gini_L=gini_L, sumsq=sumsq,
    full_loss=full_loss, full_tr=full_tr, train_idx=train_idx, neuron_k=neuron_k, neuron_frac=neuron_frac, neuron_var=neuron_var,
)
np.savez("cache/tables.npz", full_steps=full_steps, pred_tab=pred_tab, lp_tab=lp_tab)
np.save("cache/acts_final.npy", acts_final)
print("done", time.time() - t0, flush=True)
