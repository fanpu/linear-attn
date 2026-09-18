#!/usr/bin/env python
"""Glue: load the real shards and call check_disjoint.  [AI-owned]

    python scripts/check_data.py --data data/fineweb_edu --windows 16384
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.data_check import check_disjoint  # noqa: E402
from testbed.data import load_windows, read_shard, shard_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=16384, help="number of validation windows to check (the final-eval set)")
    args = ap.parse_args()
    train = [read_shard(p) for p in shard_paths(args.data, "train")]
    val = load_windows([read_shard(p) for p in shard_paths(args.data, "val")], args.T, args.windows).numpy()
    n_train = sum(len(s) // args.T for s in train)
    t0 = time.time()
    hits = check_disjoint(train, val, args.T)
    print(f"{hits} of {len(val)} validation windows appear as aligned training windows "
          f"({n_train} training windows fingerprinted, {time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
