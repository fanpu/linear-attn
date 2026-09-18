"""Train/validation disjointness.  [you]"""

import hashlib

import numpy as np


def check_disjoint(train_shards: list, val: np.ndarray, T: int) -> int:
    """Number of validation windows that appear, token for token, in training.

    train_shards: list of 1-D uint16 token streams (np.memmap is fine).
    val:          int64 [N, T] validation windows.
    A validation window counts if it equals some ALIGNED training window
    shard[k*T:(k+1)*T] for some shard and some integer k. A copy that starts at
    a non-multiple of T does not count (a known weakness of this check).
    Suggested method: put a fingerprint (e.g. the window's bytes) of every
    aligned training window into a set, then look up each validation window.
    Expected on real data: 0.
    """
    train_hashes = set()
    for shard in train_shards:
        for i in range(0, len(shard), T):
            window = shard[i  : i + T]
            fingerprint = int.from_bytes(
                hashlib.blake2b(
                    window.astype(np.uint16).tobytes(), digest_size=8
                ).digest(),
                "little",
            )
            train_hashes.add(fingerprint)

    val_hashes = set()
    for window in val:
        fingerprint = int.from_bytes(
            hashlib.blake2b(window.astype(np.uint16).tobytes(), digest_size=8).digest(),
            "little",
        )
        val_hashes.add(fingerprint)

    intersection_cnt = len(train_hashes.intersection(val_hashes))
    return intersection_cnt
