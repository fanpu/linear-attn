"""Tests for ref_nll and evaluate.  [AI-owned; do not edit]

A stand-in model runs on CPU with no data files: logits[b, t] = W[input_ids[b, t]]
(a bigram model), and when `labels` is given it shifts them exactly as fla's
*ForCausalLM does (drop the first label, pad the end with ignore_index = -100)
and returns the mean cross-entropy over the valid positions in `.loss`.

test_ref_nll   checks your reference against the handout's hand-worked example
               (V = 3, W = diag(1, 2, 3), window [0, 1, 2] -> 1.8955).
test_evaluate  feeds 5 windows with B = 2 so the final batch is short, and
               requires evaluate to match ref_nll to 1e-5.
"""
import contextlib
import math
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import run_evaluate, run_ref_nll


class _ShiftingBigram(nn.Module):
    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.W = nn.Parameter(W.clone().float())

    def forward(self, input_ids, labels=None, use_cache=False, **kw):
        logits = self.W[input_ids]  # [B, T, V]
        out = types.SimpleNamespace(logits=logits, loss=None)
        if labels is not None:
            lab = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], -100)), 1)
            out.loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=-100)
        return out


def test_ref_nll():
    model = _ShiftingBigram(torch.diag(torch.tensor([1.0, 2.0, 3.0])))
    val = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    got = run_ref_nll(model, val)
    want = (math.log(math.e + 2) + math.log(math.e ** 2 + 2)) / 2  # 1.8954947
    assert isinstance(got, float)
    assert got == pytest.approx(want, abs=1e-5)


def test_evaluate():
    g = torch.Generator().manual_seed(3)
    V, T, N, B = 64, 16, 5, 2
    model = _ShiftingBigram(torch.randn(V, V, generator=g) * 1.5)
    val = torch.randint(0, V, (N, T), generator=g, dtype=torch.int64)
    ref = run_ref_nll(model, val)
    got = run_evaluate(model, val, B, contextlib.nullcontext())
    assert isinstance(got, float)
    assert got == pytest.approx(ref, abs=1e-5), f"evaluate={got} ref_nll={ref}"
    assert model.training, "evaluate must restore model.train() before returning"
    # The wrong-but-plausible answer is the unweighted mean of the batch means.
    means = [model(input_ids=val[i:i + B], labels=val[i:i + B]).loss.item() for i in range(0, N, B)]
    unweighted = sum(means) / len(means)
    assert abs(unweighted - ref) > 1e-4, "test setup: batches must differ enough for weighting to matter"
    assert abs(got - unweighted) > 1e-5, "evaluate is averaging batch means without token weights"
