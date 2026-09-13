"""Decode-map engine: all pixels of a (T, top-p) or (T, repetition-penalty) grid decoded
at once on a *prefix trie*.

Every pixel runs the same sampler as sample.py (inverse-CDF on the probability-sorted
vocabulary with shared per-position uniforms u_t, or Gumbel-max with shared noise).
Pixels whose generated prefixes are identical share one trie node, so each distinct
prefix goes through the network exactly once. That makes it fast, because big cells cost
one forward pass. It also makes a run self-consistent: two pixels with the same prefix
see bit-identical logits, so batch-dependent kernel noise cannot create speckle inside a
cell. It can only move a boundary coherently.

Memory: node KV rows live in a GPU buffer of Ncap rows. When the trie outgrows that, the
excess nodes (with their pixels) are offloaded to CPU and processed later (DFS stack).
The forward pass always runs in fixed windows of Fb rows (padding rows are scratch).

Outputs are the same format as sample.py.
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from qwen import Qwen3
from sample import EOS, PROMPTS, chat_ids, shared_noise


class Engine:
    def __init__(self, model, prompt_ids, L, rule="icdf", seed=0, K=2048, Ncap=384, Fb=128,
                 use_pen=False, pix_batch=8192, graph=True, fast=True, NB=1024, delta=0.04):
        self.m, self.L, self.rule, self.K = model, L, rule, K
        self.Ncap, self.Fb, self.use_pen, self.pix_batch = Ncap, Fb, use_pen, pix_batch
        self.fast, self.NB, self.delta = fast and rule == "icdf", NB, delta
        self.timing = bool(os.environ.get("DM_TIMING"))
        self.P = len(prompt_ids)
        self.V = model.w["lm_head.weight"].shape[0]
        u, gum = shared_noise(seed, L, self.V, rule)
        self.u = u.cuda()
        self.gum = gum.cuda() if gum is not None else None
        model.alloc(1, self.P)
        ids = torch.tensor(prompt_ids, device="cuda")[None]
        self.logits0 = model.forward(ids, 0)[0][None]
        self.k0, self.v0 = model.cache_k.clone(), model.cache_v.clone()
        self.prompt_mask = torch.zeros(self.V, dtype=torch.bool, device="cuda")
        self.prompt_mask[ids[0]] = True
        model.alloc(Ncap + Fb, self.P + L)
        self.seen = torch.zeros(Ncap + Fb, self.V, dtype=torch.bool, device="cuda") if use_pen else None
        self.eos = torch.tensor(EOS, device="cuda")
        from qwen import StepGraphs
        self.graphs = StepGraphs(model, Fb, -(-Ncap // Fb), self.P + L) if graph else None
        self.stats = dict(forward_rows=0, fallback_keys=0, offloads=0, node_steps=0)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, logits, rowp, T, p, rho, t, tid, rid):
        """logits [n,V] fp32 per node row; rowp [m] node row of each pixel; T,p,rho [m] f64.
        Returns token [m], H_model [m], H_samp [m], logp [m]."""
        V, K, dev = self.V, self.K, logits.device
        m = rowp.shape[0]
        lsm = torch.log_softmax(logits.float(), -1)
        Hm_node = -(lsm.exp() * lsm).sum(-1)
        tok = torch.empty(m, dtype=torch.long, device=dev)
        Hs = torch.empty(m, dtype=torch.float64, device=dev)
        # key = (node row, T, rho): everything but p and u
        # tid, rid: integer ids of the pixel's T and rho values (T, rho are functions of them)
        key1 = (rowp * self.nT + tid) * self.nR + rid
        keys, inv = torch.unique(key1, return_inverse=True)
        nk = keys.shape[0]
        order = torch.argsort(inv)
        counts = torch.bincount(inv, minlength=nk)
        starts = torch.cumsum(counts, 0) - counts
        u = self.u[t]
        a = 0
        cum = torch.cumsum(counts, 0).cpu().numpy()
        while a < nk:
            base = 0 if a == 0 else cum[a - 1]
            b = int(np.searchsorted(cum, base + self.pix_batch, side="right"))
            b = min(max(b, a + 1), a + 256, nk)
            kr = keys[a:b] // (self.nT * self.nR)
            kT = T[order[starts[a:b]]]
            krho = rho[order[starts[a:b]]]
            z = logits[kr].double()
            if self.use_pen:
                pen = krho[:, None]
                z = torch.where(self.seen[kr], torch.where(z > 0, z / pen, z * pen), z)
            zT = z / kT[:, None]
            lse = torch.logsumexp(zT, -1)
            # top-K per distinct (row, rho): the sort order does not depend on T
            okey = kr * self.nR + keys[a:b] % self.nR
            ou, oinv = torch.unique(okey, return_inverse=True)
            orep = torch.zeros(ou.shape[0], dtype=torch.long, device=dev).scatter_(
                0, oinv, torch.arange(b - a, device=dev))
            vals, ix = torch.topk(z[orep], K, -1, sorted=True)
            vals, ix = vals[oinv], ix[oinv]
            lq = vals / kT[:, None] - lse[:, None]
            q = lq.exp()
            C = torch.cumsum(q, -1)
            E = torch.cumsum(q * lq, -1)
            before = C - q
            # pixels of these keys
            pi = order[int(base): int(cum[b - 1])]
            kk = inv[pi] - a
            pp = p[pi]
            trunc = pp < 1.0
            n = torch.searchsorted(before[kk], pp[:, None])[:, 0].clamp_min(1)       # #(before < p)
            fb = trunc & (n == K) & (C[kk, K - 1] < pp)
            nm1 = (n - 1).clamp_max(K - 1)
            Z = torch.where(trunc, C[kk, nm1], torch.ones_like(pp))
            anyfull = bool((~trunc).any())
            if anyfull:
                full_mean = (torch.softmax(zT, -1) * zT).sum(-1)
                Hfull = lse - full_mean
            if self.rule == "icdf":
                r = torch.searchsorted(C[kk], (u * Z)[:, None])[:, 0]
                r = torch.where(trunc, torch.minimum(r, nm1), r)
                fb = fb | (~trunc & (r >= K))
                tk = ix[kk, r.clamp_max(K - 1)]
            else:
                y = vals / kT[:, None] + self.gum[t][ix]
                _, cmi = torch.cummax(y, -1)
                tk = ix[kk, cmi[kk, nm1]]
                if anyfull:
                    fullarg = (zT + self.gum[t][None]).argmax(-1)
                    tk = torch.where(trunc, tk, fullarg[kk])
            hs = torch.where(trunc, -E[kk, nm1] / Z + torch.log(Z), Hfull[kk] if anyfull else Z)
            if fb.any():
                fbi = fb.nonzero()[:, 0]
                fk = torch.unique(kk[fbi])
                self.stats["fallback_keys"] += len(fk)
                remap = torch.full((b - a,), -1, dtype=torch.long, device=dev)
                remap[fk] = torch.arange(len(fk), device=dev)
                for c0 in range(0, len(fk), 16):
                    fkc = fk[c0:c0 + 16]
                    zsf, ixf = torch.sort(zT[fkc], -1, descending=True)
                    lqf = zsf - lse[fkc][:, None]
                    qf = lqf.exp()
                    Cf = torch.cumsum(qf, -1)
                    Ef = torch.cumsum(qf * lqf, -1)
                    Bf = Cf - qf
                    if self.rule == "gumbel":
                        _, cmf = torch.cummax(zsf + self.gum[t][ixf], -1)
                    rk = remap[kk[fbi]]
                    selall = fbi[(rk >= c0) & (rk < c0 + 16)]
                    for s0 in range(0, len(selall), 128):
                        sel = selall[s0:s0 + 128]
                        j = remap[kk[sel]] - c0
                        ps = pp[sel]
                        tr = ps < 1.0
                        ns = torch.searchsorted(Bf[j], ps[:, None])[:, 0].clamp(1, V)
                        Zs = torch.where(tr, Cf[j, ns - 1], torch.ones_like(ps))
                        if self.rule == "icdf":
                            rs = torch.searchsorted(Cf[j], (u * Zs)[:, None])[:, 0].clamp_max(V - 1)
                            rs = torch.where(tr, torch.minimum(rs, ns - 1), rs)
                            tk[sel] = ixf[j, rs]
                        else:
                            tk[sel] = ixf[j, cmf[j, ns - 1]]
                        hs[sel] = torch.where(tr, -Ef[j, ns - 1] / Zs + torch.log(Zs), hs[sel])
                    del zsf, ixf, lqf, qf, Cf, Ef, Bf
            tok[pi] = tk
            Hs[pi] = hs
            a = b
        return tok, Hm_node[rowp], Hs, lsm[rowp, tok]

    @torch.no_grad()
    def unit_logits(self, urow, urho):
        z = self.cur_logits[urow].double()
        if self.use_pen:
            pen = urho[:, None]
            z = torch.where(self.seen[urow], torch.where(z > 0, z / pen, z * pen), z)
        return z

    @torch.no_grad()
    def row_stats(self, z, K):
        """z [n,V] (fp32 logits, or fp64 penalised). Top-K (sorted) and a histogram of ALL V tokens
        in NB logit bins of width delta below the row max (last bin: everything lower).
        Returns vals64 [n,K], ix [n,K], vbin [n,K], cnt [n,NB], ssum [n,NB], zmin [n]."""
        n, NB = z.shape[0], self.NB
        vals, ix = torch.topk(z, K, -1, sorted=True)
        zmax = vals[:, :1]
        b = (zmax - z).div_(self.delta).floor_().clamp_(0, NB - 1).long()
        b += NB * torch.arange(n, device=z.device)[:, None]
        cnt = torch.bincount(b.view(-1), minlength=n * NB).view(n, NB).double()
        ssum = torch.bincount(b.view(-1), weights=z.reshape(-1).double(), minlength=n * NB).view(n, NB)
        vbin = b.gather(1, ix) - NB * torch.arange(n, device=z.device)[:, None]
        zmin = z.amin(-1).double()
        return vals.double(), ix, vbin, cnt, ssum, zmin

    @torch.no_grad()
    def sample_fast(self, logits, rowp, T, p, rho, t, tid, rid):
        """Same rule as sample(), with a certified top-K prefilter (tiers K/8, K, 16K, full sort).

        Per unit (node row, rho): float64 top-K of the (penalised) logits, plus a histogram of all
        tokens in logit bins of width delta (count, float64 sum); the top-K' tokens are subtracted
        to get the tail. Per key (unit, T), with w = exp((z - z_max)/T), the tail mass of each bin
        is bracketed exactly by Jensen (lower: c exp(mean/T)) and the convexity chord between the
        bin edges (upper), so the full mass W lies in [W_lo, W_hi]. A pixel's top-p cutoff
        n = #(C_{k-1} < pW) and its inverse-CDF token are accepted only when they agree for W_lo
        and W_hi and lie inside the top K'; otherwise the pixel goes to the next tier, and finally
        to a full float64 sort (the reference computation). H_samp of untruncated pixels uses the
        bin means for the tail (quadrature)."""
        self.cur_logits = logits
        m, dev = rowp.shape[0], logits.device
        lsm = torch.log_softmax(logits.float(), -1)
        Hm_node = -(lsm.exp() * lsm).sum(-1)
        tok = torch.empty(m, dtype=torch.long, device=dev)
        Hs = torch.empty(m, dtype=torch.float64, device=dev)
        todo = torch.arange(m, device=dev)
        K = self.K
        tiers = ((K // 8, K, 1024, 4096, 1 << 16), (K, K, 256, 1024, 16384), (16 * K, 16 * K, 8, 64, 512),
                 (None, None, 8, 8, 128))
        for Kp, Ks, UB, KB, PB in tiers:
            if len(todo) == 0:
                break
            if self.timing:
                torch.cuda.synchronize(); t0 = time.time()
            todo = self._tier(todo, rowp, T, p, rho, tid, rid, t, Kp, Ks, UB, KB, PB, tok, Hs)
            self.stats[f"fail{Kp}"] = self.stats.get(f"fail{Kp}", 0) + len(todo)
            if self.timing:
                torch.cuda.synchronize(); self.stats[f"sec{Kp}"] = round(self.stats.get(f"sec{Kp}", 0) + time.time() - t0, 2)
        return tok, Hm_node[rowp], Hs, lsm[rowp, tok]

    def _tier(self, sub, rowp, T, p, rho, tid, rid, t, K, Ks, UB, KB, PB, tok, Hs):
        """Pixels `sub` with top-K certification using stats of top-Ks (K=None: exact full sort).
        Returns the pixels that could not be certified."""
        NB, dev, V, d = self.NB, rowp.device, self.V, self.delta
        u = self.u[t]
        units, uinv = torch.unique(rowp[sub] * self.nR + rid[sub], return_inverse=True)
        keys, kinv = torch.unique(uinv * self.nT + tid[sub], return_inverse=True)
        nk, nu = keys.shape[0], units.shape[0]
        kT = torch.empty(nk, dtype=torch.float64, device=dev).scatter_(0, kinv, T[sub])
        urho = torch.empty(nu, dtype=torch.float64, device=dev).scatter_(0, uinv, rho[sub])
        kunit = keys // self.nT
        porder = torch.argsort(kinv)
        pcum = torch.cumsum(torch.bincount(kinv, minlength=nk), 0).cpu().numpy()
        kcum = torch.cumsum(torch.bincount(kunit, minlength=nu), 0).cpu().numpy()
        bad_all = []
        edge_hi = -(torch.arange(NB, device=dev, dtype=torch.float64) * d) + 1e-4     # relative to z_max
        edge_lo = -(torch.arange(1, NB + 1, device=dev, dtype=torch.float64) * d) - 1e-4
        for ua in range(0, nu, UB):
            ub = min(nu, ua + UB)
            urow = units[ua:ub] // self.nR
            if K is None:
                z = self.unit_logits(urow, urho[ua:ub])
                vals, ix = torch.sort(z, -1, descending=True)
                del z
                Kc = V
            else:
                Kc = K
                if Ks == self.K and not self.use_pen:
                    vals, ix, vbin = self.rs_vals[urow], self.rs_ix[urow], self.rs_vbin[urow]
                    cnt, ssum, zmin = self.rs_cnt[urow], self.rs_ssum[urow], self.rs_zmin[urow]
                else:
                    zu = self.unit_logits(urow, urho[ua:ub]) if self.use_pen else self.cur_logits[urow]
                    vals, ix, vbin, cnt, ssum, zmin = self.row_stats(zu, Ks)
                    del zu
                vals, ix, vbin = vals[:, :K], ix[:, :K], vbin[:, :K]
                cnt = cnt.scatter_add(1, vbin, -torch.ones_like(vals))
                ssum = ssum.scatter_add(1, vbin, -vals)
                has = cnt > 0.5
                zmx = vals[:, :1]
                hi_e = torch.minimum(zmx + edge_hi, zmx)
                lo_e = (zmx + edge_lo).clone()
                lo_e[:, -1] = zmin - 1e-4
                lo_e = torch.minimum(lo_e, hi_e)
                mean = torch.where(has, torch.minimum(torch.maximum(ssum / cnt.clamp_min(1), lo_e), hi_e), float("-inf"))
                cnt = torch.where(has, cnt, 0.0)
                th = torch.where(has, (mean - lo_e) / (hi_e - lo_e).clamp_min(1e-300), 0.0).clamp(0, 1)
                lo_e = torch.where(has, lo_e, float("-inf"))
                hi_e = torch.where(has, hi_e, float("-inf"))
                del ssum, vbin
            ka, kend = (0 if ua == 0 else int(kcum[ua - 1])), int(kcum[ub - 1])
            while ka < kend:
                base = 0 if ka == 0 else int(pcum[ka - 1])
                kb = int(np.searchsorted(pcum, base + PB, side="right"))
                kb = min(max(kb, ka + 1), ka + KB, kend)
                ul = kunit[ka:kb] - ua
                Tk = kT[ka:kb][:, None]
                v0 = vals[ul, :1]
                lw = (vals[ul] - v0) / Tk
                w = lw.exp()
                C = torch.cumsum(w, -1)
                S = torch.cumsum(w * lw, -1)
                before = C - w
                del w
                if K is None:
                    Wlo = Whi = Wmid = C[:, -1]
                    Smid = S[:, -1]
                else:
                    cu = cnt[ul]
                    lm = (mean[ul] - v0) / Tk
                    lo = cu * lm.exp()
                    e0, e1 = ((lo_e[ul] - v0) / Tk).exp(), ((hi_e[ul] - v0) / Tk).exp()
                    tu = th[ul]
                    hi = cu * ((1 - tu) * e0 + tu * e1)
                    Ctop = C[:, -1]
                    Wlo = (Ctop + lo.sum(-1)) * (1 - 1e-8)
                    Whi = (Ctop + hi.sum(-1)) * (1 + 1e-8)
                    Wmid = 0.5 * (Wlo + Whi)
                    Smid = S[:, -1] + torch.where(cu > 0, lo * lm, 0.0).sum(-1)
                    del lm, cu, lo, e0, e1, tu, hi
                pl = porder[base:int(pcum[kb - 1])]
                pi = sub[pl]
                kk = kinv[pl] - ka
                pp = p[pi]
                trunc = pp < 1.0
                Ck = C[kk]
                Bk = before[kk]
                nlo = torch.searchsorted(Bk, (pp * Wlo[kk])[:, None])[:, 0].clamp_min(1)
                nhi = torch.searchsorted(Bk, (pp * Whi[kk])[:, None])[:, 0].clamp_min(1)
                del Bk
                nm1 = (nhi - 1).clamp_max(Kc - 1)
                Z = C[kk, nm1]
                r_tr = torch.minimum(torch.searchsorted(Ck, (u * Z)[:, None])[:, 0], nm1)
                rlo = torch.searchsorted(Ck, (u * Wlo[kk])[:, None])[:, 0]
                rhi = torch.searchsorted(Ck, (u * Whi[kk])[:, None])[:, 0]
                del Ck
                if K is None:
                    ok = torch.ones_like(trunc)
                else:
                    ok = torch.where(trunc, (nlo == nhi) & (nhi < K), (rlo == rhi) & (rhi < K))
                r = torch.where(trunc, r_tr, rhi).clamp_max(Kc - 1)
                tok[pi] = ix[ul[kk], r]
                Sn = S[kk, nm1]
                Hs[pi] = torch.where(trunc, torch.log(Z) - Sn / Z, torch.log(Wmid[kk]) - Smid[kk] / Wmid[kk])
                bad_all.append(pi[~ok])
                del C, S, before, lw
                ka = kb
        return torch.cat(bad_all) if bad_all else sub[:0]

    # ------------------------------------------------------------------ trie bookkeeping
    def forward_rows(self, last, n, t):
        m, Fb = self.m, self.Fb
        out = torch.empty(n, self.V, device="cuda")
        if self.fast and not self.use_pen:
            K, NB = self.K, self.NB
            self.rs_vals = torch.empty(n, K, dtype=torch.float64, device="cuda")
            self.rs_ix = torch.empty(n, K, dtype=torch.long, device="cuda")
            self.rs_vbin = torch.empty(n, K, dtype=torch.long, device="cuda")
            self.rs_cnt = torch.empty(n, NB, dtype=torch.float64, device="cuda")
            self.rs_ssum = torch.empty(n, NB, dtype=torch.float64, device="cuda")
            self.rs_zmin = torch.empty(n, dtype=torch.float64, device="cuda")
        for r0 in range(0, n, Fb):
            if self.timing:
                torch.cuda.synchronize(); tq = time.time()
            ids = torch.zeros(Fb, 1, dtype=torch.long, device="cuda")
            k = min(Fb, n - r0)
            ids[:k, 0] = last[r0:r0 + k]
            n_out = min(Fb, -(-k // 16) * 16)                 # head on a multiple of 16 rows
            if self.graphs is not None:
                out[r0:r0 + k] = self.graphs.run(ids, self.P + t - 1, r0)[:k]
            else:
                out[r0:r0 + k] = m.forward(ids, self.P + t - 1, r0, n_out)[:k]
            if self.timing:
                torch.cuda.synchronize(); self.stats["sec_fwd"] = round(self.stats.get("sec_fwd", 0) + time.time() - tq, 2); tq = time.time()
            if self.fast and not self.use_pen:
                st = self.row_stats(out[r0:r0 + k], self.K)
                for buf, x in zip((self.rs_vals, self.rs_ix, self.rs_vbin, self.rs_cnt, self.rs_ssum, self.rs_zmin), st):
                    buf[r0:r0 + k] = x
                if self.timing:
                    torch.cuda.synchronize(); self.stats["sec_rs"] = round(self.stats.get("sec_rs", 0) + time.time() - tq, 2)
        self.stats["forward_rows"] += n
        return out

    def copy_rows(self, src, dst, length):
        ck, cv = self.m.cache_k, self.m.cache_v
        for c in range(0, len(src), 64):
            s, d = src[c:c + 64], dst[c:c + 64]
            ck[:, d, :, :length] = ck[:, s, :, :length]
            cv[:, d, :, :length] = cv[:, s, :, :length]
            if self.use_pen:
                self.seen[d] = self.seen[s]

    def run(self, T, p, rho, chunks, log_every=20, tid=None, rid=None):
        """T,p,rho: [N] f64 cuda. chunks: list of pixel-index tensors (initial states)."""
        N, L, V = T.shape[0], self.L, self.V
        if tid is None:
            _, tid = torch.unique(T, return_inverse=True)
        if rid is None:
            _, rid = torch.unique(rho, return_inverse=True)
        self.tid, self.rid = tid, rid
        self.nT, self.nR = int(tid.max()) + 1, int(rid.max()) + 1
        self.out_tok = torch.full((N, L), -1, dtype=torch.int32, device="cuda")
        self.out_Hm = torch.full((N, L), float("nan"), dtype=torch.float16, device="cuda")
        self.out_Hs = torch.full((N, L), float("nan"), dtype=torch.float16, device="cuda")
        self.out_lp = torch.full((N, L), float("nan"), dtype=torch.float16, device="cuda")
        stack = [dict(pix=c.cuda(), row=torch.zeros(len(c), dtype=torch.long, device="cuda"), n=1, t=0)
                 for c in reversed(chunks)]
        t0, done_states = time.time(), 0
        while stack:
            st = stack.pop()
            self.run_state(st, stack, T, p, rho)
            done_states += 1
            if done_states % log_every == 0 or not stack:
                pend = sum(s["pix"].numel() for s in stack)
                print(f"states {done_states} pending {len(stack)} ({pend} px)  {self.stats}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    def run_state(self, st, stack, T, p, rho):
        P, L, V, Ncap = self.P, self.L, self.V, self.Ncap
        ck, cv = self.m.cache_k, self.m.cache_v
        pix, row, n, t = st["pix"], st["row"], st["n"], st["t"]
        length = P + max(t - 1, 0)
        if n > Ncap:                                     # split oversized state
            h = Ncap
            hi = row >= h
            stack.append(dict(pix=pix[hi], row=row[hi] - h, n=n - h, t=t,
                              k=st["k"][:, h:], v=st["v"][:, h:], last=st["last"][h:],
                              seen=st["seen"][h:] if self.use_pen else None))
            pix, row, n = pix[~hi], row[~hi], h
            st = dict(st, k=st["k"][:, :h], v=st["v"][:, :h], last=st["last"][:h],
                      seen=st["seen"][:h] if self.use_pen else None)
        if t == 0:
            ck[:, 0, :, :P] = self.k0[:, 0]
            cv[:, 0, :, :P] = self.v0[:, 0]
            if self.use_pen:
                self.seen[0] = self.prompt_mask
            logits = self.logits0
            if self.fast and not self.use_pen:
                (self.rs_vals, self.rs_ix, self.rs_vbin, self.rs_cnt, self.rs_ssum,
                 self.rs_zmin) = self.row_stats(logits, self.K)
        else:
            ck[:, :n, :, :length] = st["k"].cuda(non_blocking=True)
            cv[:, :n, :, :length] = st["v"].cuda(non_blocking=True)
            if self.use_pen:
                self.seen[:n] = st["seen"].cuda()
            logits = self.forward_rows(st["last"].cuda(), n, t)
        while True:
            self.stats["node_steps"] += n
            if self.timing:
                torch.cuda.synchronize(); tq = time.time()
            tok, hm, hs, lp = (self.sample_fast if self.fast else self.sample)(logits, row, T[pix], p[pix], rho[pix], t,
                                          self.tid[pix], self.rid[pix])
            if self.timing:
                torch.cuda.synchronize(); self.stats["sec_samp"] = round(self.stats.get("sec_samp", 0) + time.time() - tq, 2)
            self.out_tok[pix, t] = tok.int()
            self.out_Hm[pix, t] = hm.half()
            self.out_Hs[pix, t] = hs.half()
            self.out_lp[pix, t] = lp.half()
            alive = ~torch.isin(tok, self.eos)
            t += 1
            if t == L or not alive.any():
                return
            pix, row, tok = pix[alive], row[alive], tok[alive]
            ckey, cinv = torch.unique(row * V + tok, return_inverse=True)
            prow, ctok = ckey // V, ckey % V
            nc = len(ckey)
            first = torch.ones(nc, dtype=torch.bool, device="cuda")
            first[1:] = prow[1:] != prow[:-1]
            used = torch.zeros(n, dtype=torch.bool, device="cuda")
            used[prow] = True
            holes = (~used).nonzero()[:, 0]
            nx = int((~first).sum())
            extra_rows = torch.cat([holes, torch.arange(n, n + max(0, nx - len(holes)), device="cuda")])[:nx]
            crow = prow.clone()
            crow[~first] = extra_rows
            length = P + t - 1
            n_rows = int(crow.max()) + 1
            if n_rows > Ncap:                            # offload children in rows >= h
                h = Ncap // 2
                off = crow >= h
                oi = off.nonzero()[:, 0]
                src = prow[oi]
                ks, vs = [], []
                for c in range(0, len(src), 64):
                    ks.append(ck[:, src[c:c + 64], :, :length].cpu())
                    vs.append(cv[:, src[c:c + 64], :, :length].cpu())
                seen_off = None
                if self.use_pen:
                    seen_off = self.seen[src].clone()
                    seen_off[torch.arange(len(oi), device="cuda"), ctok[oi]] = True
                    seen_off = seen_off.cpu()
                local = torch.full((nc,), -1, dtype=torch.long, device="cuda")
                local[oi] = torch.arange(len(oi), device="cuda")
                pmask = off[cinv]
                stack.append(dict(pix=pix[pmask], row=local[cinv[pmask]], n=len(oi), t=t,
                                  k=torch.cat(ks, 1), v=torch.cat(vs, 1), last=ctok[oi].cpu(),
                                  seen=seen_off))
                self.stats["offloads"] += 1
                keep = ~off
                ki = keep.nonzero()[:, 0]
                newidx = torch.full((nc,), -1, dtype=torch.long, device="cuda")
                newidx[ki] = torch.arange(len(ki), device="cuda")
                pk = keep[cinv]
                pix, cinv = pix[pk], newidx[cinv[pk]]
                prow, ctok, crow, first = prow[ki], ctok[ki], crow[ki], first[ki]
                if len(ki) == 0:
                    return
                n_rows = int(crow.max()) + 1
            mv = ~first
            if mv.any():
                self.copy_rows(prow[mv], crow[mv], length)
            if self.use_pen:
                self.seen[crow, ctok] = True
            row = crow[cinv]
            n = n_rows
            last = torch.zeros(n, dtype=torch.long, device="cuda")
            last[crow] = ctok
            logits = self.forward_rows(last, n, t)


def make_grid(args):
    R = args.res
    xs = args.x[0] + (np.arange(R) + 0.5) / R * (args.x[1] - args.x[0])
    ys = args.y[0] + (np.arange(R) + 0.5) / R * (args.y[1] - args.y[0])
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    Tall = XX.ravel()
    if args.grid == "tp":
        Pall, Rall = YY.ravel(), np.ones(R * R)
    else:
        Pall, Rall = np.full(R * R, args.p_fixed), YY.ravel()
    return xs, ys, Tall, Pall, Rall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="story")
    ap.add_argument("--grid", default="tp", choices=["tp", "tr"])
    ap.add_argument("--rule", default="icdf", choices=["icdf", "gumbel"])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--x", type=float, nargs=2, default=[0.0, 2.0])
    ap.add_argument("--y", type=float, nargs=2, default=None)
    ap.add_argument("--p_fixed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--Ncap", type=int, default=384)
    ap.add_argument("--Fb", type=int, default=128)
    ap.add_argument("--chunk_cols", type=int, default=0, help="initial states = blocks of columns (0: all)")
    ap.add_argument("--shuffle", action="store_true", help="random pixel order in initial chunks (placement test)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--nograph", action="store_true", help="eager forward (no CUDA graphs)")
    ap.add_argument("--slow", action="store_true", help="old uncertified-free sampler (V-wide per key)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.y is None:
        args.y = [0.0, 1.0] if args.grid == "tp" else [1.0, 2.0]
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    pids = chat_ids(tok, PROMPTS[args.prompt])
    model = Qwen3(dtype=torch.float32 if args.fp32 else torch.bfloat16)
    E = Engine(model, pids, args.L, args.rule, args.seed, args.K, args.Ncap, args.Fb,
               use_pen=args.grid == "tr", graph=not args.nograph, fast=not args.slow)
    xs, ys, Tall, Pall, Rall = make_grid(args)
    R = args.res
    f = lambda a: torch.tensor(a, dtype=torch.float64, device="cuda")
    idx = np.arange(R * R).reshape(R, R)
    if args.shuffle:
        perm = np.random.default_rng(123).permutation(R * R)
        cc = max(1, args.chunk_cols) * R
        chunks = [torch.tensor(perm[i:i + cc]) for i in range(0, R * R, cc)]
    elif args.chunk_cols:
        chunks = [torch.tensor(idx[:, j:j + args.chunk_cols].ravel()) for j in range(0, R, args.chunk_cols)]
    else:
        chunks = [torch.tensor(idx.ravel())]
    t0 = time.time()
    E.run(f(Tall), f(Pall), f(Rall), chunks)
    wall = time.time() - t0
    toks = E.out_tok.cpu().numpy()
    iseos = np.isin(toks, EOS)
    first = np.where(iseos.any(1), iseos.argmax(1), args.L)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, tokens=toks.reshape(R, R, args.L), H_model=E.out_Hm.cpu().numpy().reshape(R, R, args.L),
        H_samp=E.out_Hs.cpu().numpy().reshape(R, R, args.L), logp=E.out_lp.cpu().numpy().reshape(R, R, args.L),
        eos_pos=first.reshape(R, R).astype(np.int16), xs=xs, ys=ys, grid=args.grid, rule=args.rule,
        prompt=PROMPTS[args.prompt], prompt_key=args.prompt, prompt_ids=np.array(pids),
        u=E.u.cpu().numpy(), p_fixed=args.p_fixed, seed=args.seed, L=args.L, wall=wall,
        fp32_body=args.fp32, K=args.K, Ncap=args.Ncap, Fb=args.Fb, stats=str(E.stats),
        engine="trie", graph=not args.nograph, fast=not args.slow, x=args.x, y=args.y)
    print("saved", args.out, f"{wall:.0f}s", E.stats)


if __name__ == "__main__":
    main()
