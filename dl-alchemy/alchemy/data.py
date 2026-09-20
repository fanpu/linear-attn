"""Token shards in llm.c format: 256 int32 header (magic 20240520, version, ntok) then uint16 tokens.
Batches are stateless functions of (seed, step), so resume needs no loader state."""
import glob
import os

import numpy as np
import torch

HEADER_BYTES = 1024


def _open(path):
    h = np.fromfile(path, dtype=np.int32, count=3)
    assert h[0] == 20240520 and h[1] == 1, f"bad shard header in {path}"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(int(h[2]),))


class Shards:
    def __init__(self, data_dir, split, context):
        paths = sorted(glob.glob(os.path.join(data_dir, f"*_{split}_*.bin")))
        assert paths, f"no {split} shards in {data_dir}"
        self.shards = [_open(p) for p in paths]
        self.context = context
        # non-overlapping windows of context+1 tokens (inputs + shifted targets)
        self.per_shard = [(len(s) - 1) // context for s in self.shards]
        self.starts = np.concatenate([[0], np.cumsum(self.per_shard)])
        self.n_windows = int(self.starts[-1])

    def windows(self, idx):
        out = np.empty((len(idx), self.context + 1), dtype=np.int64)
        shard_of = np.searchsorted(self.starts, idx, side="right") - 1
        for i, (w, s) in enumerate(zip(idx, shard_of)):
            o = (int(w) - int(self.starts[s])) * self.context
            out[i] = self.shards[s][o:o + self.context + 1]
        return torch.from_numpy(out)


class TrainStream:
    """Step k of a run with data seed s reads windows perm_s[k*B:(k+1)*B]."""

    def __init__(self, data_dir, context, batch_seqs, seed):
        self.data = Shards(data_dir, "train", context)
        self.batch_seqs = batch_seqs
        self.perm = np.random.default_rng(seed).permutation(self.data.n_windows)

    def batch(self, step):
        lo = step * self.batch_seqs
        assert lo + self.batch_seqs <= len(self.perm), "run needs more tokens than are on disk"
        return self.data.windows(self.perm[lo:lo + self.batch_seqs])


class ValSet:
    """The first n_windows windows of the validation shard; identical for every run."""

    def __init__(self, data_dir, context, n_windows):
        self.data = Shards(data_dir, "val", context)
        self.n = min(n_windows, self.data.n_windows)

    def batches(self, batch_seqs, limit=None):
        n = self.n if limit is None else min(limit, self.n)
        for lo in range(0, n, batch_seqs):
            yield self.data.windows(np.arange(lo, min(lo + batch_seqs, n)))
