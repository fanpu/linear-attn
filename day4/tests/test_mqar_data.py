"""Tests for the MQAR generator (AI-owned code that the whole sprint reuses). [AI]"""
import torch

from testbed.tasks.mqar import IGNORE_INDEX, multiquery_ar, query_gaps


def test_mqar_docstring_example_layout():
    """vocab 12, 2 pairs, 16 tokens: keys in [1, 6), values in [6, 12), the first four
    positions hold the pairs, every query is a key seen in the context, and the label at
    a query position is that key's value."""
    x, y = multiquery_ar(vocab_size=12, num_examples=50, input_seq_len=16, seed=0, num_kv_pairs=2)
    assert x.shape == (50, 16) and y.shape == (50, 16) and x.dtype == torch.int64
    keys, vals = x[:, 0:4:2], x[:, 1:4:2]
    assert ((keys >= 1) & (keys < 6)).all() and ((vals >= 6) & (vals < 12)).all()
    assert (keys[:, 0] != keys[:, 1]).all() and (vals[:, 0] != vals[:, 1]).all()
    assert (y[:, :4] == IGNORE_INDEX).all()
    n_labeled = (y != IGNORE_INDEX).sum(1)
    assert (n_labeled == 2).all()
    for i in range(50):
        kv = {int(keys[i, j]): int(vals[i, j]) for j in range(2)}
        for t in range(4, 16):
            if y[i, t] != IGNORE_INDEX:
                assert int(y[i, t]) == kv[int(x[i, t])]
            else:
                assert x[i, t] == 0  # filler, random_non_queries=False


def test_mqar_is_deterministic_in_seed_and_gaps_are_consistent():
    a = multiquery_ar(8192, 20, 512, seed=7, num_kv_pairs=64)
    b = multiquery_ar(8192, 20, 512, seed=7, num_kv_pairs=64)
    c = multiquery_ar(8192, 20, 512, seed=8, num_kv_pairs=64)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and not torch.equal(a[0], c[0])
    gaps = query_gaps(a[0], a[1], 64)
    assert all(len(g) == 64 for g in gaps)
    assert all(qp - kp == gap > 0 for rows in gaps for (kp, qp, gap) in rows)


def test_mqar_random_non_queries_keeps_labels():
    x0, y0 = multiquery_ar(8192, 5, 128, seed=1, num_kv_pairs=8, random_non_queries=False)
    x1, y1 = multiquery_ar(8192, 5, 128, seed=1, num_kv_pairs=8, random_non_queries=True)
    assert torch.equal(y0, y1)
    assert (x1 != 0).all() and torch.equal(x0[x0 != 0], x1[x0 != 0])
