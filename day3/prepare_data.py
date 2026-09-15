"""
Day 3: tokenize FineWeb-Edu (sample-10BT) into llm.c-format shards.

Ownership: Claude (loud). If this is wrong it crashes, hangs, or writes shards whose token
histogram / decode is visibly garbage. Spot-check by decoding the first 200 tokens of a shard.

Format per shard (same as karpathy/llm.c dev/data/fineweb.py, so llm.c tooling can read it):
    256 x int32 header: [magic=20240520, version=1, ntok, 0, 0, ...]
    ntok x uint16 tokens
Shard 0 (`*_val_000000.bin`) is validation. Every document is written as [EOT] + tokens
(GPT-2 tokenizer, EOT = 50256). Documents may straddle a shard boundary; llm.c does the same.

Usage:
    python prepare_data.py --out data/fineweb_edu --num-shards 16
    # 16 shards x 100M tokens = 1 val + 15 train (1.5B train tokens), ~3.2 GB on disk.
"""
import argparse, os, sys, time
import multiprocessing as mp

import numpy as np

MAGIC, VERSION, HEADER_INTS = 20240520, 1, 256
EOT = 50256  # GPT-2 <|endoftext|>


def write_shard(path, tokens_u16):
    assert tokens_u16.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens_u16)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_u16.tobytes())


def read_shard(path):
    """Memory-mapped uint16 view of one shard's tokens (no copy)."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC and header[1] == VERSION, f"bad header in {path}"
    ntok = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(ntok,))


def shard_paths(data_dir):
    """(val_path, [train paths sorted])."""
    names = sorted(os.listdir(data_dir))
    val = [n for n in names if "_val_" in n and n.endswith(".bin")]
    train = [n for n in names if "_train_" in n and n.endswith(".bin")]
    assert len(val) == 1, f"expected exactly one val shard, found {val}"
    assert train, "no train shards"
    return os.path.join(data_dir, val[0]), [os.path.join(data_dir, n) for n in train]


_enc = None


def _init_worker():
    global _enc
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")


def _tokenize_batch(texts):
    out = []
    for t in texts:
        ids = [EOT] + _enc.encode_ordinary(t)
        out.append(np.asarray(ids, dtype=np.uint16))
    toks = np.concatenate(out)
    assert toks.max() < 50257
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--num-shards", type=int, default=16, help="total incl. 1 val shard")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--batch-docs", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    def doc_batches():
        buf = []
        for ex in ds:
            buf.append(ex["text"])
            if len(buf) == a.batch_docs:
                yield buf
                buf = []
        if buf:
            yield buf

    shard_idx, fill = 0, 0
    buf = np.empty(a.shard_tokens, dtype=np.uint16)
    t0, total = time.time(), 0
    with mp.Pool(a.workers, initializer=_init_worker) as pool:
        for toks in pool.imap(_tokenize_batch, doc_batches(), chunksize=4):
            pos = 0
            while pos < len(toks):
                take = min(a.shard_tokens - fill, len(toks) - pos)
                buf[fill : fill + take] = toks[pos : pos + take]
                fill += take
                pos += take
                total += take
                if fill == a.shard_tokens:
                    split = "val" if shard_idx == 0 else "train"
                    path = os.path.join(a.out, f"edu_fineweb_{split}_{shard_idx:06d}.bin")
                    write_shard(path, buf)
                    dt = time.time() - t0
                    print(f"wrote {path}  ({total/1e6:.0f}M tokens, {total/dt/1e6:.2f}M tok/s)", flush=True)
                    shard_idx, fill = shard_idx + 1, 0
                    if shard_idx == a.num_shards:
                        print("done", flush=True)
                        return
    print(f"stream exhausted after {shard_idx} full shards ({fill} tokens in a partial shard, discarded)")


if __name__ == "__main__":
    main()
