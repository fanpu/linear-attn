"""Synthetic tasks for probing what a recurrent memory can store and learn.

All generators take a torch.Generator for reproducibility and return plain tensors.

1. In-context linear regression, optionally with a *drifting* task:
       w_1 ~ N(0, I/d),   w_{t+1} = sqrt(1 - q) w_t + sqrt(q) xi_t,  xi_t ~ N(0, I/d)
       x_t ~ N(0, I),     y_t = <w_t, x_t> + sigma * eps_t
   q = 0 is the usual stationary task. With q > 0 the task is a random walk whose marginal stays N(0, I/d),
   so ||w_t|| does not grow along the sequence and only the *speed* of change varies with q.
   Old examples become stale, which is exactly the situation a forget gate is for.

2. Multi-query associative recall (MQAR, Arora et al. 2023 "Zoology"): a prefix of key-value token pairs,
   then queries that repeat earlier keys; the target at each query is the value that key was paired with.

3. Raw key-value recall at the level of the recurrence (no embeddings): random keys with a controllable
   pairwise correlation, for measuring memory capacity and interference directly.
"""
import torch


def icl_regression(batch, n, d, sigma=0.0, drift=0.0, gen=None, dtype=torch.float64, device="cpu"):
    """Returns x [B,n,d], y [B,n], w [B,n,d] (the task in force when each (x_t, y_t) was generated)."""
    kw = dict(generator=gen, dtype=dtype, device=device)
    x = torch.randn(batch, n, d, **kw)
    w0 = torch.randn(batch, d, **kw) / d**0.5
    if drift > 0:
        steps = torch.randn(batch, n, d, **kw) / d**0.5
        ws, w = [], w0
        a, s = (1 - drift) ** 0.5, drift**0.5
        for t in range(n):
            ws.append(w)
            w = a * w + s * steps[:, t]
        w = torch.stack(ws, 1)
    else:
        w = w0[:, None].expand(batch, n, d)
    y = (w * x).sum(-1) + sigma * torch.randn(batch, n, **kw)
    return x, y, w


def mqar(batch, n_pairs, n_queries, vocab, gen=None, device="cpu"):
    """MQAR sequences. Keys come from [0, vocab//2), values from [vocab//2, vocab); keys are distinct within a sequence.

    Returns tokens [B, 2*n_pairs + n_queries] (long), targets (same shape, -100 where no loss),
    so position t's target is the value paired with the key at query position t.
    """
    half = vocab // 2
    assert n_pairs <= half, "need distinct keys"
    keys = torch.argsort(torch.rand(batch, half, generator=gen, device=device), dim=-1)[:, :n_pairs]
    vals = half + torch.randint(0, vocab - half, (batch, n_pairs), generator=gen, device=device)
    prefix = torch.stack([keys, vals], -1).reshape(batch, 2 * n_pairs)
    pick = torch.randint(0, n_pairs, (batch, n_queries), generator=gen, device=device)
    q_keys = torch.gather(keys, 1, pick)
    q_vals = torch.gather(vals, 1, pick)
    tokens = torch.cat([prefix, q_keys], 1)
    targets = torch.full_like(tokens, -100)
    targets[:, 2 * n_pairs :] = q_vals
    return tokens, targets


def correlated_keys(batch, n, d, rho=0.0, gen=None, dtype=torch.float64, device="cpu"):
    """Unit-norm keys [B,n,d] with E[<k_i, k_j>] ~= rho for i != j (a shared component of weight sqrt(rho))."""
    kw = dict(generator=gen, dtype=dtype, device=device)
    shared = torch.randn(batch, 1, d, **kw)
    shared = shared / shared.norm(dim=-1, keepdim=True)
    own = torch.randn(batch, n, d, **kw)
    own = own / own.norm(dim=-1, keepdim=True)
    k = rho**0.5 * shared + (1 - rho) ** 0.5 * own
    return k / k.norm(dim=-1, keepdim=True)
