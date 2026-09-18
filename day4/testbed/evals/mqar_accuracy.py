"""MQAR accuracy: fraction of query positions answered correctly. [you]

after zoology/train.py (`compute_metrics`): the accuracy of one example is
the mean over its query positions (labels != -100) of `argmax(logits) ==
label`, and the accuracy of a run is the mean of that over examples.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def mqar_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-example accuracy over query positions.

    Args:
        logits: `[B, T, V]`, the model's output at every position.
        labels: `[B, T]` int64; `-100` at every position that is not the
            answer to a query, the value token at the position after each query.

    Returns:
        acc: `[B]` fp32, for each example the fraction of its positions with
            `labels != -100` at which `logits.argmax(-1) == labels`. Positions
            with label `-100` are excluded from both the numerator and the
            denominator; they are never counted as correct.
    """

    mask = labels != -100  # [B, T]
    pred = logits.argmax(dim=-1)  # [B, T]
    correct = (pred == labels) & mask  # [B, T]
    n_correct = correct.sum(dim=-1).to(torch.float32)  # [B]
    n_labeled = mask.sum(dim=-1).to(torch.float32)  # [B]
    return n_correct / n_labeled.clamp_min(1.0)
