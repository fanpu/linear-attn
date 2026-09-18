#!/usr/bin/env python
"""Look at the data before training on it.  [AI-owned]

    python scripts/look_at_shards.py --data data/fineweb_edu            # header, first two windows, 10 random windows
    python scripts/look_at_shards.py --data data/fineweb_edu --stats    # also document count and length stats on shard 0

Prints the shard header, decodes the first two aligned windows of the first
training shard (the "first two records" of the file), then ten random windows
from random training shards, each cut to its first 80 tokens.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.data import EOT, HEADER_INTS, read_shard, shard_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    train = shard_paths(args.data, "train")
    val = shard_paths(args.data, "val")
    hdr = np.fromfile(train[0], dtype=np.int32, count=HEADER_INTS)
    print(f"{train[0]}: header magic={hdr[0]} version={hdr[1]} tokens={hdr[2]}  (dtype of stream: uint16)")
    s = read_shard(train[0])
    print(f"first 12 token ids: {s[:12].tolist()}")
    for k in range(2):
        print(f"\n--- window {k} of {train[0]} (first 80 of {args.T} tokens) ---")
        print(repr(enc.decode(s[k * args.T:k * args.T + 80].tolist())))
    rng = np.random.default_rng(args.seed)
    print("\n=== ten random training windows ===")
    for i in range(10):
        p = train[rng.integers(len(train))]
        s = read_shard(p)
        k = int(rng.integers(len(s) // args.T))
        w = s[k * args.T:(k + 1) * args.T]
        print(f"\n[{i}] {os.path.basename(p)} window {k}: {int((w == EOT).sum())} document starts in window")
        print(repr(enc.decode(w[:80].tolist())))
    if args.stats:
        v = read_shard(val[0])
        starts = np.flatnonzero(v == EOT)
        lens = np.diff(starts)
        print(f"\nshard 0 (validation): {len(v)} tokens, {len(starts)} documents, "
              f"median doc length {int(np.median(lens))} tokens, mean {lens.mean():.0f}, "
              f"longest {lens.max()}; fraction of tokens that are EOT: {len(starts)/len(v):.4f}")


if __name__ == "__main__":
    main()
