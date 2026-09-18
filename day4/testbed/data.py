"""llm.c shard format, window loading, and the seeded batch order.  [AI-owned]
# Shard format after karpathy/llm.c dev/data/data_common.py (write_datafile);
# memory-mapped reading and the batch loader after karpathy/nanoGPT train.py (get_batch),
# with the random-offset sampling replaced by a seeded permutation of aligned windows.

Shard file layout (Karpathy's llm.c `edu_fineweb10B` format):
    header : 256 x int32.  header[0] = 20240520 (magic), header[1] = 1 (version),
             header[2] = number of tokens that follow. header[3:] are zero.
    tokens : that many uint16 token ids, one flat stream, documents separated by
             the GPT-2 end-of-text token 50256 which is written at the START of
             every document.

A window is T consecutive tokens starting at a multiple of T within one shard
(aligned, non-overlapping). Windows never cross a shard boundary; the tail of a
shard shorter than T is dropped.
"""
import os
import numpy as np
import torch

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
EOT = 50256


def write_shard(path: str, tokens: np.ndarray) -> None:
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def read_shard(path: str) -> np.ndarray:
    """Memory-mapped uint16 token stream (no copy)."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC, f"{path}: bad magic {header[0]}"
    assert header[1] == VERSION, f"{path}: bad version {header[1]}"
    n = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(n,))


def shard_paths(data_dir: str, split: str) -> list:
    names = sorted(n for n in os.listdir(data_dir) if n.endswith(".bin") and f"_{split}_" in n)
    assert names, f"no {split} shards in {data_dir}"
    return [os.path.join(data_dir, n) for n in names]


def n_windows(shards: list, T: int) -> list:
    return [len(s) // T for s in shards]


def load_windows(shards: list, T: int, n: int) -> torch.Tensor:
    """The first n aligned windows of the shards, in file order, as int64 [n, T].
    Used for the validation set (shard 0) and for the fixed training subset."""
    out = np.empty((n, T), dtype=np.int64)
    filled = 0
    for s in shards:
        k = min(len(s) // T, n - filled)
        if k <= 0:
            continue
        out[filled:filled + k] = np.asarray(s[: k * T]).reshape(k, T)
        filled += k
        if filled == n:
            break
    assert filled == n, f"only {filled} windows available, need {n}"
    return torch.from_numpy(out)


class WindowIndex:
    """Maps a global window id in [0, total) to (shard, offset) without loading anything."""

    def __init__(self, shards: list, T: int, total: int):
        self.shards, self.T = shards, T
        counts = n_windows(shards, T)
        assert sum(counts) >= total, f"budget needs {total} windows, shards hold {sum(counts)}"
        self.total = total
        self.starts = np.cumsum([0] + counts)

    def get(self, wid: int) -> np.ndarray:
        si = int(np.searchsorted(self.starts, wid, side="right") - 1)
        k = wid - self.starts[si]
        return np.asarray(self.shards[si][k * self.T:(k + 1) * self.T])


def batch_order(seed: int, total_windows: int) -> np.ndarray:
    """The permutation of window ids that seed `seed` trains on. Deterministic in
    the seed alone, so a resumed run replays the same order."""
    rng = np.random.default_rng(seed)
    return rng.permutation(total_windows)


def batch_iter(index: WindowIndex, order: np.ndarray, B: int, start_step: int = 0, device: str = "cuda"):
    """Yields (step, x) with x int64 [B, T] on `device`, starting at batch `start_step`.
    Batch s is windows order[s*B:(s+1)*B]; a budget of S steps uses order[:S*B]."""
    T = index.T
    steps = len(order) // B
    for s in range(start_step, steps):
        ids = order[s * B:(s + 1) * B]
        x = np.stack([index.get(int(w)) for w in ids]).astype(np.int64)
        yield s, torch.from_numpy(x).to(device, non_blocking=True)
