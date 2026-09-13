"""Shared pieces for the toy induction-head experiment: model + data.

Model: 2-layer, attention-only transformer (no MLPs), pre-LayerNorm,
learned absolute positional embeddings, untied unembedding.
Data : a mixture that rewards induction.
  (a) "Markov text": tokens from a fixed sparse random bigram chain, with
      1-2 earlier spans copied verbatim later in the sequence.
  (b) "repeated random": a uniform random segment of period P, tiled.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V = 256        # vocab
T = 128        # context length
D = 128        # d_model
H = 4          # heads per layer
L = 2          # layers


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, ablate=None, return_pattern=False):
        B, t, d = x.shape
        q = self.q(x).view(B, t, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, t, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, t, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float('-inf'))
        a = s.softmax(-1)                          # B,h,t,t
        z = a @ v                                  # B,h,t,dh
        if ablate is not None:                     # zero-ablate listed heads
            z = z.clone()
            for hh in ablate:
                z[:, hh] = 0
        out = self.o(z.transpose(1, 2).reshape(B, t, d))
        return (out, a) if return_pattern else (out, None)


class AttnOnly(nn.Module):
    def __init__(self, V=V, T=T, D=D, H=H, L=L):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.ln = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.attn = nn.ModuleList([Attn(D, H) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.unemb = nn.Linear(D, V, bias=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx, ablate=None, return_patterns=False):
        t = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
        pats = []
        for li, (ln, at) in enumerate(zip(self.ln, self.attn)):
            ab = None if ablate is None else [h for (l_, h) in ablate if l_ == li]
            o, a = at(ln(x), ablate=ab, return_pattern=return_patterns)
            x = x + o
            pats.append(a)
        logits = self.unemb(self.lnf(x))
        return (logits, pats) if return_patterns else logits


def make_bigram(seed=0, alpha=0.1, device='cpu'):
    """Fixed sparse random bigram chain: each row ~ Dirichlet(alpha)."""
    import numpy as np
    rs = np.random.default_rng(seed)
    P = rs.dirichlet(np.full(V, alpha), size=V).astype('float32')
    return torch.tensor(P, device=device)


def markov_batch(Pbig, B, gen, device):
    x = torch.empty(B, T, dtype=torch.long, device=device)
    x[:, 0] = torch.randint(0, V, (B,), generator=gen, device=device)
    for t in range(1, T):
        x[:, t] = torch.multinomial(Pbig[x[:, t - 1]], 1, generator=gen).squeeze(1)
    return x


def insert_copies(x, gen, n_copies=2, lmin=8, lmax=32):
    """Copy an earlier span verbatim to a later location. Returns (x, pred)
    where pred marks positions 2..L of each copied span (induction-predictable)."""
    B, T_ = x.shape
    dev = x.device
    pred = torch.zeros_like(x, dtype=torch.bool)
    pos = torch.arange(T_, device=dev)[None]
    for _ in range(n_copies):
        Ls = torch.randint(lmin, lmax + 1, (B, 1), generator=gen, device=dev)
        u1 = torch.rand(B, 1, generator=gen, device=dev)
        u2 = torch.rand(B, 1, generator=gen, device=dev)
        src = (u1 * (T_ - 2 * Ls + 1)).long()
        lo = src + Ls
        dst = lo + (u2 * (T_ - Ls - lo + 1)).long()
        span = (pos >= dst) & (pos < dst + Ls)
        idx = torch.where(span, src + pos - dst, pos)
        x = torch.gather(x, 1, idx)
        pred = pred & ~span
        pred = pred | (span & (pos > dst))
    return x, pred


def repeated_random(B, period, gen, device, T_=T, n_rep=None):
    seg = torch.randint(0, V, (B, period), generator=gen, device=device)
    reps = math.ceil(T_ / period) if n_rep is None else n_rep
    return seg.repeat(1, reps)[:, :T_ if n_rep is None else period * n_rep]
