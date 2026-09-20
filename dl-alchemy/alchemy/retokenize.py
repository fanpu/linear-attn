"""One-time CPU job: re-encode the GPT-2-tokenized FineWeb-Edu shards with a small byte-level BPE vocabulary.

Why: with the 50k GPT-2 vocabulary the output layer is ~55% of the baseline's FLOPs (measured 26k vs 75k tok/s).
    python -m alchemy.retokenize ../day3/data/fineweb_edu data/fineweb_edu_bpe8k 8192
Output shards keep the llm.c format (256 int32 header, uint16 tokens); <|endoftext|> (id 0) starts every document."""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

GPT2_EOT = 50256
HEADER_BYTES = 1024


def read_docs(path, max_tokens=None):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
    if max_tokens:
        toks = toks[:max_tokens]
    cuts = np.flatnonzero(toks == GPT2_EOT)
    pieces = np.split(np.asarray(toks), cuts)
    docs = [p[1:] if len(p) and p[0] == GPT2_EOT else p for p in pieces]
    return enc.decode_batch([d.tolist() for d in docs if len(d)])


def train_tokenizer(src_dir, out_path, vocab_size):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    sample = read_docs(sorted(glob.glob(os.path.join(src_dir, "*_train_*.bin")))[0], max_tokens=40_000_000)
    tok.train_from_iterator(sample, trainer)
    tok.save(out_path)


def convert(args):
    src, dst, tok_path = args
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    docs = read_docs(src)
    out = []
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]):
            out.append(np.asarray([0] + e.ids, dtype=np.uint16))
    toks = np.concatenate(out)
    header = np.zeros(256, dtype=np.int32)
    header[:3] = (20240520, 1, len(toks))
    with open(dst + ".tmp", "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())
    os.replace(dst + ".tmp", dst)
    return os.path.basename(dst), len(toks)


def main():
    src_dir, dst_dir, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(dst_dir, exist_ok=True)
    tok_path = os.path.join(dst_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        train_tokenizer(src_dir, tok_path, vocab)
        print("tokenizer trained", flush=True)
    jobs = [(p, os.path.join(dst_dir, os.path.basename(p)), tok_path)
            for p in sorted(glob.glob(os.path.join(src_dir, "*.bin")))
            if not os.path.exists(os.path.join(dst_dir, os.path.basename(p)))]
    with Pool(8) as pool:
        for name, n in pool.imap_unordered(convert, jobs):
            print(name, n, flush=True)
    print("retokenize finished", flush=True)


if __name__ == "__main__":
    main()
