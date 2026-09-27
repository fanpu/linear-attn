"""A transformer slice: the small GPT of art/spectroscope (6 layers, d 384, 6 heads, vocab 8192,
context 256, TinyStories; checkpoint cache/ckpt/<run>.pt after 10k steps) on a filter-normalised
random plane, loss on ONE fixed token batch.

    python gpt_plane.py --run adamw10k_wd --res 128 --span 1 --out cache/gpt/plane_adamw_r128.npz

Directions: Li et al. filter normalisation with every ROW of every 2-D matrix as a 'filter'
(q, k, v, o, up, down rows; token-embedding, position-embedding and head rows); LayerNorm gains and
biases zero.  Seeds 1 and 2.  Loss: mean next-token cross-entropy over 16 fixed sequences x 256
tokens of the TRAINING stream (offsets from seed 0), fp32 with TF32 matmuls (declared; see
--check).  Model code copied from spectroscope/train_gpt.py (unchanged architecture).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = "/home/fzeng/ml/research/art/spectroscope"
V, T, D, H, L = 8192, 256, 384, 6, 6
HD = D // H


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

    def forward(s, idx, tgt):
        x = s.emb(idx) + s.pos.weight[None]
        for b in s.blocks:
            x = b(x)
        lo = s.head(s.lnf(x)).float().view(-1, V)
        return F.cross_entropy(lo, tgt.reshape(-1)), lo


def directions(model, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, w in model.named_parameters():
        d = torch.randn(w.shape, generator=g, dtype=torch.float64)
        if w.dim() < 2:
            d.zero_()
        else:
            wd = w.detach().double().cpu()
            d *= (wd.norm(dim=1) / d.norm(dim=1).clamp_min(1e-12))[:, None]
        out[n] = d.float().cuda()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="adamw10k_wd")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--nseq", type=int, default=16)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pasar_job; pasar_job.apply_memory_limit()
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32); torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    ck = torch.load(os.path.join(SPEC, "cache", "ckpt", f"{a.run}.pt"), map_location="cuda", weights_only=False)
    model = GPT().cuda()
    sd = ck["model"]; sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tr = np.load(os.path.join(SPEC, "cache", "train.npy"), mmap_mode="r")
    offs = np.random.default_rng(0).integers(0, len(tr) - T - 1, a.nseq)
    seq = torch.tensor(np.stack([np.asarray(tr[o:o + T + 1]) for o in offs]).astype(np.int64), device="cuda")
    x, y = seq[:, :-1], seq[:, 1:]
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    d1, d2 = directions(model, 1), directions(model, 2)
    params = dict(model.named_parameters())
    live = [n for n in w0 if d1[n].abs().max() > 0]

    @torch.no_grad()
    def at(al, be):
        for n in live:
            torch.add(w0[n], d1[n], alpha=al, out=params[n]); params[n].add_(d2[n], alpha=be)
        l, lo = model(x, y)
        return l.item(), (lo.argmax(-1) == y.reshape(-1)).float().mean().item()

    ref = at(0.0, 0.0)
    print("centre loss / acc", ref, "step", ck.get("step"), flush=True)
    if a.check:                                   # TF32 vs fp32 on a line
        ts = np.linspace(-a.span, a.span, 65); res = {}
        for tf in (0, 1):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf)
            res[tf] = np.array([at(float(t), 0.3)[0] for t in ts])
        print(json.dumps(dict(max_abs_log10_diff=float(np.abs(np.log10(res[0]) - np.log10(res[1])).max()),
                              rms_d2_fp32=float(np.sqrt(np.mean(np.diff(np.log10(res[0]), 2) ** 2))))), flush=True)
        return
    ax = np.linspace(-a.span, a.span, a.res)
    Lg = np.zeros((a.res, a.res)); Ag = np.zeros((a.res, a.res))
    ckp = os.path.join(pasar_job.persist_dir(), "gpt_part.npz"); i0 = 0
    if pasar_job.resuming() and os.path.exists(ckp):
        z = np.load(ckp); Lg, Ag, i0 = z["L"], z["A"], int(z["i"]); pasar_job.resumed(i0)
    t0 = time.time(); last = t0
    for i in range(i0, a.res):
        for j in range(a.res):
            Lg[i, j], Ag[i, j] = at(float(ax[j]), float(ax[i]))
        pasar_job.progress(i + 1, a.res)
        if time.time() - last > 300:
            np.savez(ckp, L=Lg, A=Ag, i=i + 1); pasar_job.checkpoint(i + 1); last = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, loss=Lg, acc=Ag, span=a.span, run=a.run, nseq=a.nseq, tf32=a.tf32, centre_loss=ref[0],
                        secs=time.time() - t0)
    print(json.dumps(dict(out=a.out, secs=round(time.time() - t0, 1), lmin=float(Lg.min()), lmax=float(Lg.max()))), flush=True)


if __name__ == "__main__":
    main()
