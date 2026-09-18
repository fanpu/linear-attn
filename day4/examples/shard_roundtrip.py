#!/usr/bin/env python
"""Worked example of the shard format: write a tiny shard, read it back, cut it
into aligned windows, and print the header.  [complete and runnable]

    python examples/shard_roundtrip.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from testbed.data import EOT, HEADER_INTS, WindowIndex, batch_order, load_windows, read_shard, write_shard

T = 4
docs = [[11, 12, 13], [21, 22], [31, 32, 33, 34, 35]]          # three "documents" of token ids
stream = np.concatenate([[EOT] + d for d in docs]).astype(np.uint16)
print("stream of", len(stream), "tokens:", stream.tolist())

path = os.path.join(tempfile.mkdtemp(), "toy_train_000001.bin")
write_shard(path, stream)
hdr = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
print("header[:3] =", hdr[:3].tolist(), "(magic, version, token count); file size =", os.path.getsize(path), "bytes =",
      HEADER_INTS * 4, "+ 2 *", len(stream))

s = read_shard(path)
print("read back:", np.asarray(s).tolist(), "dtype", s.dtype)
print("aligned windows of T =", T, ":", len(s) // T, "(the tail of", len(s) % T, "tokens is dropped)")
print(load_windows([s], T, len(s) // T).tolist())

idx = WindowIndex([s], T, total=3)
print("seed 0 order:", batch_order(0, 3).tolist(), " seed 1 order:", batch_order(1, 3).tolist())
print("window 2 by id:", idx.get(2).tolist())
