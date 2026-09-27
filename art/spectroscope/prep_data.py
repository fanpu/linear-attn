"""Tokenise one TinyStories shard (CPU). Run with an ephemeral env:
    uv run --no-project --with pyarrow --with tiktoken --with numpy python spectroscope/prep_data.py

GPT-2 BPE, then the vocabulary is cut to the 8191 most frequent tokens in the training shard plus
one UNK id (8191) -- TinyStories uses a small vocabulary and a 50k embedding would dominate the
model. <|endoftext|> separates stories. Outputs uint16 arrays in spectroscope/cache/.
"""
import os
import numpy as np
import pyarrow.parquet as pq
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'cache', 'raw')
V = 8192

enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token


def tok(fn):
    texts = pq.read_table(os.path.join(RAW, fn)).column('text').to_pylist()
    ids = enc.encode_ordinary_batch(texts, num_threads=16)
    out = []
    for x in ids:
        out.append(np.asarray(x, np.int32))
        out.append(np.array([EOT], np.int32))
    return np.concatenate(out)


tr = tok('train-00000-of-00004-2d5a1467fff1081b.parquet')
va = tok('validation-00000-of-00001-869c898b519ad725.parquet')
cnt = np.bincount(tr, minlength=enc.n_vocab)
keep = np.argsort(-cnt)[:V - 1]
remap = np.full(enc.n_vocab, V - 1, np.int64)
remap[keep] = np.arange(V - 1)
trm, vam = remap[tr].astype(np.uint16), remap[va].astype(np.uint16)
print('train tokens', len(trm), 'val tokens', len(vam))
print('UNK rate train %.5f val %.5f' % ((trm == V - 1).mean(), (vam == V - 1).mean()))
np.save(os.path.join(HERE, 'cache', 'train.npy'), trm)
np.save(os.path.join(HERE, 'cache', 'val.npy'), vam)
np.save(os.path.join(HERE, 'cache', 'vocab_gpt2_ids.npy'), keep)
