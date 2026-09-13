"""Part B compute: attention patterns of Qwen3-0.6B on repeated random tokens.

Writes
  cache/qwen_main.npz      full 28x16 attention grid for the main probe
                           (P=50 x 4 repeats), per-head scores averaged over
                           N random sequences, per-position loss.
  cache/qwen_book.npz      pattern-book variants (period / repeats / length /
                           quasi-periodic words / controls), full grids.

  python compute_qwen.py
"""
import time, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch.cuda.set_per_process_memory_fraction(0.10)
dev = 'cuda'
MODEL = 'Qwen/Qwen3-0.6B'
NORMAL_VOCAB = 151643          # ids >= this are special tokens

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32,
                                             attn_implementation='eager').to(dev).eval()
NL, NH = model.config.num_hidden_layers, model.config.num_attention_heads


def rand_alphabet(n, rs):
    return rs.choice(NORMAL_VOCAB, size=n, replace=False)


@torch.no_grad()
def run(tokens):
    """tokens: (B, T) int array -> attn (B, L, H, T, T) float32 cpu, per-pos loss (B, T-1)."""
    x = torch.tensor(tokens, device=dev)
    out = model(x, output_attentions=True)
    att = torch.stack(out.attentions, 1).float()            # B,L,H,T,T
    lp = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), x[:, 1:], reduction='none')
    return att, lp


def periodic(P, reps, rs, B=1):
    return np.stack([np.tile(rand_alphabet(P, rs), reps) for _ in range(B)])


def head_scores(att, P, T):
    idx = torch.arange(T, device=att.device)
    ii = idx[P:]
    ind = att[:, :, :, ii, ii - P + 1].mean((0, 3))     # L,H  token after previous occurrence
    dup = att[:, :, :, ii, ii - P].mean((0, 3))         # L,H  previous occurrence itself
    prev = att[:, :, :, idx[1:], idx[:-1]].mean((0, 3)) # L,H  previous token
    sink = att[:, :, :, 1:, 0].mean((0, 3))             # L,H  first-token sink
    # all-previous-occurrence induction mass: sum_k a[i, i-kP+1]
    allind = torch.zeros(att.shape[1], att.shape[2], device=att.device)
    cnt = 0
    for i in range(P, T):
        js = [j for j in range(i - P + 1, 0, -P)]
        allind += att[:, :, :, i, js].sum(-1).mean(0)
        cnt += 1
    allind /= cnt
    return dict(ind=ind.cpu().numpy(), dup=dup.cpu().numpy(), prev=prev.cpu().numpy(),
                sink=sink.cpu().numpy(), ind_all=allind.cpu().numpy())


# ---------------- main probe ----------------
rs = np.random.default_rng(2022)
P, R, NSEQ = 50, 4, 24
toks = periodic(P, R, rs, B=NSEQ)
T = P * R
scores_acc = None
lps, mean_att = [], None
for b0 in range(0, NSEQ, 4):
    att, lp = run(toks[b0:b0 + 4])
    s = head_scores(att, P, T)
    scores_acc = {k: v * 4 for k, v in s.items()} if scores_acc is None else \
        {k: scores_acc[k] + v * 4 for k, v in s.items()}
    lps.append(lp.cpu().numpy())
    mean_att = att.sum(0) if mean_att is None else mean_att + att.sum(0)
    if b0 == 0:
        att0 = att[0].half().cpu().numpy()
    del att
scores = {k: v / NSEQ for k, v in scores_acc.items()}
mean_att = (mean_att / NSEQ).half().cpu().numpy()
lps = np.concatenate(lps)
# null: same length, no repetition
null_toks = np.stack([rand_alphabet(T, rs) for _ in range(8)])
att_n, lp_n = run(null_toks)
null_scores = head_scores(att_n, P, T)
null_att0 = att_n[0].half().cpu().numpy()
del att_n
np.savez_compressed('cache/qwen_main.npz', attn=att0, mean_attn=mean_att, tokens=toks, period=P, reps=R,
                    loss_pos=lps, null_loss_pos=lp_n.cpu().numpy(), null_attn=null_att0,
                    **{f'score_{k}': v for k, v in scores.items()},
                    **{f'null_score_{k}': v for k, v in null_scores.items()})
top = np.dstack(np.unravel_index(np.argsort(-scores['ind'].ravel()), scores['ind'].shape))[0][:12]
print('main done', time.time() - t0)
print('loss per repeat:', [float(lps[:, max(0, r * P - 1):(r + 1) * P - 1].mean()) for r in range(R)])
print('top induction heads (L,H,score):', [(int(l), int(h), round(float(scores['ind'][l, h]), 3)) for l, h in top])
print('top prev-token heads:', [(int(l), int(h), round(float(scores['prev'][l, h]), 3)) for l, h in
      np.dstack(np.unravel_index(np.argsort(-scores['prev'].ravel()), scores['prev'].shape))[0][:6]])
print('null max ind score:', float(null_scores['ind'].max()))


# ---------------- pattern book ----------------
def word_to_tokens(word, block, rs):
    """Map a word over a small alphabet to tokens; each letter -> fixed random block."""
    letters = sorted(set(word))
    alph = rand_alphabet(block * len(letters), rs).reshape(len(letters), block)
    m = {c: alph[i] for i, c in enumerate(letters)}
    return np.concatenate([m[c] for c in word])


def thue_morse(n):
    return ''.join('ab'[bin(i).count('1') % 2] for i in range(n))


def fibonacci_word(n):
    a, b = 'a', 'ab'
    while len(b) < n:
        a, b = b, b + a
    return b[:n]


def nested(depth):
    # x0 = 'a'; x_{k+1} = x_k x_k x_k c_k  (hierarchical repetition)
    s = 'a'
    for k, c in enumerate('bcdefg'[:depth]):
        s = s + s + s + c
    return s


book = []
rsb = np.random.default_rng(88)
for P_ in [5, 8, 13, 21, 34, 55]:                       # vary period at T ~ 220
    R_ = max(2, 220 // P_)
    book.append((f'period {P_} x{R_}', np.tile(rand_alphabet(P_, rsb), R_), dict(kind='periodic', P=P_, R=R_)))
for R_ in [2, 3, 6, 10]:                                  # vary repeats at fixed P
    book.append((f'period 20 x{R_}', np.tile(rand_alphabet(20, rsb), R_), dict(kind='periodic', P=20, R=R_)))
for Tn in [64, 128, 256]:                                 # vary length at P=16
    book.append((f'period 16, T={Tn}', np.tile(rand_alphabet(16, rsb), Tn // 16), dict(kind='periodic', P=16, R=Tn // 16)))
book.append(('Thue-Morse word, blocks of 4', word_to_tokens(thue_morse(56), 4, rsb), dict(kind='thue-morse', block=4)))
book.append(('Fibonacci word, blocks of 4', word_to_tokens(fibonacci_word(56), 4, rsb), dict(kind='fibonacci', block=4)))
book.append(('nested (aaab)^3c..., blocks of 4', word_to_tokens(nested(3), 4, rsb)[:220], dict(kind='nested', block=4)))
seg = rand_alphabet(40, rsb)
noisy = np.tile(seg, 5)
mut = rsb.random(noisy.shape) < 0.15
mut[:40] = False
noisy[mut] = rsb.choice(NORMAL_VOCAB, size=mut.sum())
book.append(('period 40 x5, 15% tokens mutated', noisy, dict(kind='noisy', P=40, R=5, p_mut=0.15)))
book.append(('control: no repetition', rand_alphabet(200, rsb), dict(kind='control')))

save = {}
meta = []
for i, (name, tk, info) in enumerate(book):
    att, lp = run(tk[None])
    save[f'attn_{i}'] = att[0].half().cpu().numpy()
    save[f'tokens_{i}'] = tk
    save[f'loss_{i}'] = lp[0].cpu().numpy()
    meta.append(dict(name=name, T=int(len(tk)), **info))
    print(i, name, len(tk), f'mean loss {lp.mean().item():.2f}', flush=True)
    del att
save['meta'] = np.array(json.dumps(meta))
np.savez_compressed('cache/qwen_book.npz', **save)
json.dump({'wall_s': time.time() - t0, 'model': MODEL, 'dtype': 'float32', 'attn': 'eager'},
          open('cache/qwen_meta.json', 'w'))
print('all done', time.time() - t0)
