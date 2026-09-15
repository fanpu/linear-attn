#!/usr/bin/env python
"""Stream FineWeb-Edu (sample-10BT), tokenize with the GPT-2 tokenizer, and write
llm.c-format shards.  [AI-owned]

    python scripts/prepare_data.py --out data/fineweb_edu --num-shards 16

Writes `edu_fineweb_val_000000.bin` (shard 0, validation) and
`edu_fineweb_train_000001.bin` ... `_000015.bin` (training), 100M tokens each.
Every document is written as [EOT] + tokens, EOT = 50256. Idempotent: existing
shards are skipped, so an interrupted run can be restarted.

Runtime estimate on the DGX Spark: 20-40 minutes for 16 shards, dominated by
tokenization (about 1-2M tokens/s across the worker pool) and by the download
(1.6B tokens is about a sixth of sample-10BT's parquet files).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.data import EOT, write_shard  # noqa: E402

_enc = None


def _tokenize(doc):
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("gpt2")
    ids = [EOT] + _enc.encode_ordinary(doc["text"])
    arr = np.array(ids, dtype=np.uint16)
    assert (0 <= arr).all() and (arr < 2 ** 16).all()
    return arr


def shard_name(out, i):
    split = "val" if i == 0 else "train"
    return os.path.join(out, f"edu_fineweb_{split}_{i:06d}.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    done = [i for i in range(args.num_shards) if os.path.exists(shard_name(args.out, i))]
    if len(done) == args.num_shards:
        print("all shards present; nothing to do")
        return
    if done:
        print(f"WARNING: shards {done} exist; they will be kept and the stream is NOT skipped past them, "
              f"so delete them first if you want a clean rebuild. Continuing writes the missing ones.")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    n_buf, shard_i, t0, total = 0, 0, time.time(), 0
    with mp.Pool(args.workers) as pool:
        for toks in pool.imap(_tokenize, ds, chunksize=16):
            if shard_i >= args.num_shards:
                break
            if n_buf + len(toks) < args.shard_tokens:
                buf[n_buf:n_buf + len(toks)] = toks
                n_buf += len(toks)
            else:
                room = args.shard_tokens - n_buf
                buf[n_buf:] = toks[:room]
                if shard_i not in done:
                    write_shard(shard_name(args.out, shard_i), buf)
                total += args.shard_tokens
                el = time.time() - t0
                print(f"shard {shard_i} written: {total/1e6:.0f}M tokens, {total/el/1e6:.2f} M tok/s, {el/60:.1f} min",
                      flush=True)
                shard_i += 1
                rest = toks[room:]
                buf[:len(rest)] = rest
                n_buf = len(rest)
    print("done")


if __name__ == "__main__":
    main()
