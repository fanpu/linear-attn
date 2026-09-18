"""Multi-query associative recall (MQAR) data generator.

after zoology/data/multiquery_ar.py (HazyResearch/zoology, Apache-2.0). The
sequence layout, the key/value vocabulary split, the power-law gap
distribution, and the -100 label convention are Zoology's; the random-number
plumbing is a numpy Generator instead of the global numpy seed.

One example of `input_seq_len = 16`, `num_kv_pairs = 2`, `vocab_size = 12`,
with `random_non_queries=False` (the docstring example of the original):

        Key   Val  Key  Val            Query                         Query
Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0
Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

Keys are drawn from [1, vocab_size // 2), values from [vocab_size // 2,
vocab_size). The first `2 * num_kv_pairs` positions hold the pairs; the
remaining positions hold each key exactly once more (the query), at a gap
drawn from a power law, and the label at the position after a query is the
key's value. Every other label is -100, which the loss and the accuracy
ignore.
"""
from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = -100


def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    random_non_queries: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(inputs, labels)`, both int64 of shape `(num_examples, input_seq_len)`.

    `power_a` is the exponent of the gap distribution `p(g) ∝ g^(power_a - 1)`
    over the available query slots; `0.01` (Zoology's default) puts most
    queries close to the context, `1.0` makes the gaps uniform.
    `random_non_queries=True` replaces the filler zeros by random tokens.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    # Zoology asserts vocab_size > input_seq_len here, which its own docstring example
    # violates; what the construction needs is enough distinct keys and values.
    assert vocab_size // 2 - 1 >= num_kv_pairs, "not enough distinct keys for num_kv_pairs"
    assert num_kv_pairs * 4 <= input_seq_len, "need room for the pairs and their queries"

    rng = np.random.default_rng(seed)

    context_size = num_kv_pairs * 2
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    # each key (and each value) appears exactly once per example
    keys = np.stack([rng.choice(key_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_kv_pairs, replace=False) for _ in range(num_examples)])

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # power-law distribution over the query slots
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack([rng.choice(space, size=num_kv_pairs, replace=False, p=p) for _ in range(num_examples)])

    # queries and answers; one extra position so that the last answer has a slot
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), IGNORE_INDEX, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.from_numpy(examples[:, :-1].copy())
    labels = torch.from_numpy(labels[:, 1:].copy())

    if random_non_queries:
        filler = inputs == 0
        inputs[filler] = torch.from_numpy(rng.integers(vocab_size, size=inputs.shape))[filler]
    return inputs, labels


def query_gaps(inputs: torch.Tensor, labels: torch.Tensor, num_kv_pairs: int) -> list[list[tuple[int, int, int]]]:
    """For each example, the list of `(key_position, query_position, gap)` of
    its queries, where `gap = query_position - key_position`. Used by the
    failure inspection in `scripts/look_at_mqar.py`."""
    out = []
    context_size = 2 * num_kv_pairs
    for x, y in zip(inputs.tolist(), labels.tolist()):
        key_pos = {x[2 * i]: 2 * i for i in range(num_kv_pairs)}
        rows = []
        for t in range(context_size, len(y)):
            if y[t] != IGNORE_INDEX:
                # the label at t answers the query token at position t
                q = x[t]
                rows.append((key_pos[q], t, t - key_pos[q]))
        out.append(rows)
    return out
