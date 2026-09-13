"""Part B+ compute: Qwen3-0.6B attention on long SELF-SIMILAR token words.

A word over a small alphabet (Thue-Morse, period-doubling, Fibonacci, Cantor,
nested repetition) is turned into tokens by mapping every letter to a fixed
block of k random tokens (Cantor filler letters get *fresh* random tokens each
time, so only the Cantor-set blocks ever repeat). Controls: an i.i.d. random
a/b word, and a Cantor word whose repeated blocks are placed at random positions.

Attention weights are grabbed layer by layer with forward hooks (no full
output_attentions tensor is ever materialised), so T up to ~2048 fits.

Stores for each word:
  top_attn   (n_top, T, T) float16  token-level attention of the top induction heads
  grid       (28, 16, G, G) float16 all heads, area-averaged to G x G (G=256)
  loss       (T-1,) per-token loss
  letters, block, tokens

  python compute_qwen_selfsim.py -> cache/qwen_selfsim.npz
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch.cuda.set_per_process_memory_fraction(0.10)
dev = 'cuda'
NORMAL_VOCAB = 151643
G = 256
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True, dtype=torch.float32,
                                             attn_implementation='eager').to(dev).eval()
NL, NH = model.config.num_hidden_layers, model.config.num_attention_heads
main = np.load('cache/qwen_main.npz')
ind = main['score_ind']
order = np.argsort(-ind.ravel())[:8]
TOP = [(int(i // NH), int(i % NH)) for i in order]
print('top heads', TOP)


def thue_morse(n):
    return ''.join('ab'[bin(i).count('1') % 2] for i in range(n))


def period_doubling(n):
    s = 'a'
    while len(s) < n:
        s = ''.join('ab' if c == 'a' else 'aa' for c in s)
    return s[:n]


def fibonacci(n):
    a, b = 'a', 'ab'
    while len(b) < n:
        a, b = b, b + a
    return b[:n]


def cantor(depth):
    s = 'a'
    for _ in range(depth):
        s = s + 'z' * len(s) + s          # 'z' = fresh filler
    return s


def nested(depth):
    s = 'a'
    for c in 'bcdefgh'[:depth]:
        s = s + s + s + c
    return s


def to_tokens(word, k, rs):
    letters = sorted(set(word) - {'z'})
    alph = rs.choice(NORMAL_VOCAB, size=(len(letters) + 1) * k, replace=False)
    m = {c: alph[i * k:(i + 1) * k] for i, c in enumerate(letters)}
    out = []
    for c in word:
        out.append(rs.choice(NORMAL_VOCAB, size=k) if c == 'z' else m[c])
    return np.concatenate(out).astype(np.int64)


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=8, help='tokens per letter block')
ap.add_argument('--draws', type=int, default=4, help='independent random token assignments per word')
ap.add_argument('--out', default='cache/qwen_selfsim.npz')
args = ap.parse_args()
K = args.k
NLET = 2048 // K                                             # letters so that T <= 2048

rs = np.random.default_rng(31415)


def depth_for(grow, n):
    d, L = 0, 1
    while grow(L) * K <= 2048:
        L = grow(L); d += 1
    return d


def sierpinski(depth):
    s = 'a'
    for _ in range(depth):
        s = s + s + 'z'                    # X_{j+1} = X_j X_j (fresh filler)
    return s


words = []
words.append(('thue-morse', thue_morse(NLET)))
words.append(('period-doubling', period_doubling(NLET)))
words.append(('fibonacci', fibonacci(NLET)))
words.append(('cantor', cantor(depth_for(lambda L: 3 * L, NLET))))
words.append(('sierpinski-doubling', sierpinski(depth_for(lambda L: 2 * L + 1, NLET))))
words.append(('nested', nested(depth_for(lambda L: 3 * L + 1, NLET))))
words.append(('control iid a/b', ''.join(rs.choice(list('ab'), NLET))))
cw = cantor(depth_for(lambda L: 3 * L, NLET))
na = cw.count('a')
pos = np.sort(rs.choice(len(cw), na, replace=False))
cshuf = ['z'] * len(cw)
for p in pos:
    cshuf[p] = 'a'
words.append(('control cantor-shuffled', ''.join(cshuf)))

cap = {}


def make_hook(li):
    def hook(mod, inp, out):
        a = out[1][0].float()                                  # H,T,T
        T = a.shape[-1]
        cap[f'grid{li}'] = F.adaptive_avg_pool2d(a[None], G)[0].half().cpu()
        for j, (l, h) in enumerate(TOP):
            if l == li:
                cap[f'top{j}'] = a[h].half().cpu()
    return hook


hooks = [model.model.layers[li].self_attn.register_forward_hook(make_hook(li)) for li in range(NL)]
save, meta = {}, []
for wi, (name, word) in enumerate(words):
    k = K
    n = len(word)
    letter_top = []
    for dr in range(args.draws):
        tk = to_tokens(word, k, rs)
        cap.clear()
        with torch.no_grad():
            out = model(torch.tensor(tk[None], device=dev), use_cache=False)
            lp = F.cross_entropy(out.logits[0, :-1].float(), torch.tensor(tk[1:], device=dev), reduction='none')
        del out
        top = torch.stack([cap[f'top{j}'] for j in range(len(TOP))]).float()     # 8,T,T
        # letter level: mean over query tokens in block, sum over key tokens in block
        lt = top.view(len(TOP), n, k, n, k).sum(4).mean(2)
        letter_top.append(lt.half().numpy())
        if dr == 0:
            save[f'w{wi}_top'] = top.half().numpy()
            save[f'w{wi}_grid'] = torch.stack([cap[f'grid{li}'] for li in range(NL)]).numpy()
            save[f'w{wi}_tokens'] = tk
        save[f'w{wi}_loss{dr}'] = lp.cpu().numpy()
        torch.cuda.empty_cache()
    save[f'w{wi}_letter'] = np.stack(letter_top)                                 # draws,8,n,n
    meta.append(dict(name=name, word=word, block=k, T=int(n * k), draws=args.draws))
    print(wi, name, 'letters', n, 'T', n * k, 'mean loss', round(float(lp.mean()), 3), f't={time.time()-t0:.0f}s', flush=True)
save['meta'] = np.array(json.dumps(meta))
save['top_heads'] = np.array(TOP)
np.savez_compressed(args.out, **save)
print('done', time.time() - t0)
