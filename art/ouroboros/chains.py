"""Batched self-consuming retraining chains for GMM (EM) and KDE (LOO-CV bandwidth).

Protocol (declared; see README):
  * Real dataset D0 of size n (fixed per seed; nested prefix of one master stream).
  * Generation 0 model is fit to D0.
  * Generation g >= 1: n_r = floor(lambda * n) real points (the first n_r of D0, fixed)
    and n_s = n - n_r fresh samples from model g-1.
      replace:    train on  D0[:n_r] ∪ S_g                      (size n)
      accumulate: train on  D0 ∪ S_1 ∪ ... ∪ S_g                (size n + g * n_s)
    (In accumulate the real data are already in the pool, so each generation adds only its
     synthetic share; lambda = 0 is Gerstgrasser et al.'s accumulate protocol.)
  * Common random numbers: synthetic sample j of generation g of seed s uses entry j of the
    (s, g) uniform/normal pools for every (lambda, n). Evaluation samples use a separate pool.
"""
import math

import numpy as np
import torch

from common import (DEV, DT, crn_pools, gmm_em, gmm_init, gmm_sample, hist_overlap, kde_bandwidth_loocv,
                    kde_sample, real_pool, reference_sample, ref_hist, ring_mode_mass, sliced_w2)

N_EVAL = 2048
KDE_Q = 1024


class Evaluator:
    def __init__(self, target):
        self.target = target
        R = reference_sample(target, 200_000)
        self.p_ref = ref_hist(R)
        rng = np.random.default_rng(7)
        self.R = torch.tensor(R[rng.choice(len(R), N_EVAL, replace=False)], dtype=DT, device=DEV)
        self.var_ref = float(np.var(R, 0).sum())

    def __call__(self, Y):
        out = {
            "sw2": sliced_w2(Y, self.R),
            "var_ratio": Y.var(1).sum(1) / self.var_ref,
            "overlap": hist_overlap(Y, self.p_ref),
        }
        if self.target == "ring":
            mm = ring_mode_mass(Y)
            out["modes"] = (mm >= 1.0 / (8 * 4)).sum(1).double()  # mode kept if >= 1/4 of its share
            out["mode_mass"] = mm
        return out


def run_batch(target, model, regime, lam, n, seeds, G, K=8, em_iters0=200, em_iters=30,
              keep_samples_every=0, n_display=0, log=None):
    """Run B chains sharing the same n. lam [B] float, seeds [B] int (python lists)."""
    B = len(lam)
    lam_t = torch.tensor(lam, dtype=DT, device=DEV)
    n_r = torch.floor(lam_t * n + 1e-9).long()
    n_s = n - n_r
    uniq = sorted(set(seeds))
    sidx = torch.tensor([uniq.index(s) for s in seeds], device=DEV)
    D0 = torch.stack([real_pool(target, s, n) for s in uniq])[sidx]  # [B,n,2]
    ev = Evaluator(target)
    ar = torch.arange(n, device=DEV)

    cap = n + (G * n if regime == "accumulate" else 0)
    P = torch.zeros(B, cap + 1, 2, dtype=DT, device=DEV)
    P[:, :n] = D0
    cnt = torch.full((B,), n, dtype=torch.long, device=DEV)

    hist = {}
    samples = []

    def pools(g, tag, m):
        us, zs = zip(*[crn_pools(s, g, m, tag) for s in uniq])
        return torch.stack(us)[sidx], torch.stack(zs)[sidx]

    state = None
    for g in range(G + 1):
        if g > 0:
            u, z = pools(g, "train", n)
            if model == "gmm":
                S = gmm_sample(*state, u, z)
            else:
                S = kde_sample(P, cnt, state, u, z)
            if regime == "replace":
                # position p < n_r: real D0[p]; else synthetic S[p - n_r]
                j = (ar[None] - n_r[:, None]).clamp_min(0)
                Sg = torch.gather(S, 1, j[..., None].expand(-1, -1, 2))
                P[:, :n] = torch.where((ar[None] < n_r[:, None])[..., None], D0, Sg)
            else:
                # append first n_s[b] synthetic samples at positions cnt[b] ... (slot `cap` is a trash slot)
                pos = cnt[:, None] + ar[None]
                pos = torch.where(ar[None] < n_s[:, None], pos, torch.full_like(pos, cap))
                P.scatter_(1, pos[..., None].expand(-1, -1, 2), S)
                cnt = cnt + n_s
        Nmax = int(cnt.max())
        X = P[:, :Nmax]
        w = (torch.arange(Nmax, device=DEV)[None] < cnt[:, None]).to(DT)
        if model == "gmm":
            if state is None:
                state = gmm_init(X, K)
                state = gmm_em(X, w, *state, iters=em_iters0)
            else:
                state = gmm_em(X, w, *state, iters=em_iters)
        else:
            qu, _ = pools(g, "kdeq", KDE_Q)
            state = kde_bandwidth_loocv(X, cnt, qu)
        ue, ze = pools(g, "eval", N_EVAL)
        if model == "gmm":
            Y = gmm_sample(*state, ue, ze)
        else:
            Y = kde_sample(P, cnt, state, ue, ze)
        m = ev(Y)
        if model == "kde":
            m["h"] = state
        for k, v in m.items():
            hist.setdefault(k, []).append(v.cpu().numpy())
        if n_display and (g % max(keep_samples_every, 1) == 0):
            ud, zd = pools(g, "display", n_display)
            Yd = gmm_sample(*state, ud, zd) if model == "gmm" else kde_sample(P, cnt, state, ud, zd)
            samples.append(Yd.float().cpu().numpy())
        if log is not None and (g % 25 == 0 or g == G):
            log(f"{target}/{model}/{regime} n={n} g={g} N={Nmax} sw2={m['sw2'].mean():.4f}")
    out = {k: np.stack(v, 1) for k, v in hist.items()}  # [B, G+1, ...]
    if samples:
        out["display"] = np.stack(samples, 1)  # [B, T, n_display, 2]
    if model == "gmm":
        out["final_pi"], out["final_mu"], out["final_cov"] = [t.cpu().numpy() for t in state]
    return out
