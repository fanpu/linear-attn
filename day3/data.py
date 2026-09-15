"""
Day 3: training / validation data access. Ownership: Claude (loud).

Model of the data: every train shard is a flat token stream, cut into non-overlapping windows
of T tokens. Window i of the run is a global index into (shard, offset) in file order.
A run with budget D tokens uses the FIRST D/T windows in file order (a fixed subset per budget,
identical across seeds); the seed only permutes their order. So "seed" = init + data order,
never data content.

Window length is T (not T+1): fla's ForCausalLM shifts labels internally, so a window of T
tokens yields T-1 predictions. Tokens are counted as B*T per step for the budget.
"""
import numpy as np

from prepare_data import read_shard, shard_paths


class TrainShards:
    def __init__(self, data_dir, T):
        self.T = T
        _, paths = shard_paths(data_dir)
        self.shards = [read_shard(p) for p in paths]
        self.per_shard = np.array([len(s) // T for s in self.shards], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.per_shard)])
        self.total_windows = int(self.cum[-1])

    def window(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1)
        off = int(i - self.cum[s]) * self.T
        return self.shards[s][off : off + self.T]


def make_order(n_windows, shards, seed):
    """Permutation of the first n_windows global window ids, under this seed."""
    assert n_windows <= shards.total_windows, (
        f"budget needs {n_windows} windows, data has {shards.total_windows}; tokenize more shards"
    )
    rng = np.random.default_rng(seed)
    return rng.permutation(n_windows)


def train_batches(shards, order, B, start_batch=0):
    """Yields (batch_index, int64 array [B, T]). Resumable: start_batch skips ahead deterministically."""
    n_batches = len(order) // B
    for b in range(start_batch, n_batches):
        idx = order[b * B : (b + 1) * B]
        yield b, np.stack([shards.window(int(i)) for i in idx]).astype(np.int64)


def val_windows(data_dir, n_tokens, T):
    """The first n_tokens of the val shard as non-overlapping [N, T] windows, int64, fixed order."""
    val_path, _ = shard_paths(data_dir)
    v = read_shard(val_path)
    n = n_tokens // T
    assert n * T <= len(v), f"val shard has {len(v)} tokens, asked for {n*T}"
    return np.asarray(v[: n * T]).reshape(n, T).astype(np.int64)
