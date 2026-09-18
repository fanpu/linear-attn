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
    predicted = logits.argmax(dim=-1, keepdim=False)
    mask = labels != IGNORE_INDEX

    numerator = ((predicted == labels) & mask).sum(dim=-1)
    denominator = mask.sum(dim=-1)

    per_example_scores = numerator / denominator

    return per_example_scores
