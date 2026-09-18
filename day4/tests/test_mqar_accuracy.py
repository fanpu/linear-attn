"""Tests for the MQAR accuracy metric. [AI harness; you fill the adapter]"""
import torch

from .adapters import run_mqar_accuracy


def test_mqar_accuracy_hand_instance():
    """Example (accuracy) of the handout: two examples, T = 6, V = 5.
    Example 0 has queries at positions 2 and 5, predictions right and wrong: 0.5.
    Example 1 has one query at position 4, right: 1.0. The argmax at unlabeled
    positions is wrong on purpose and must not count."""
    V = 5
    logits = torch.full((2, 6, V), -1.0)
    # example 0: predicted token = argmax; position 2 -> 3 (label 3), position 5 -> 1 (label 4)
    logits[0, 2, 3] = 5.0
    logits[0, 5, 1] = 5.0
    logits[0, 0, 4] = 5.0   # unlabeled position, "correct-looking", must be ignored
    # example 1: position 4 -> 2 (label 2)
    logits[1, 4, 2] = 5.0
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 2], labels[0, 5] = 3, 4
    labels[1, 4] = 2
    acc = run_mqar_accuracy(logits, labels)
    assert acc.shape == (2,)
    torch.testing.assert_close(acc, torch.tensor([0.5, 1.0]), atol=1e-6, rtol=0)


def test_mqar_accuracy_all_wrong_and_all_right():
    logits = torch.zeros(3, 4, 7)
    logits[:, :, 6] = 1.0
    labels = torch.full((3, 4), -100, dtype=torch.long)
    labels[:, 1] = 6
    labels[:, 3] = 0
    acc = run_mqar_accuracy(logits, labels)
    torch.testing.assert_close(acc, torch.full((3,), 0.5), atol=1e-6, rtol=0)
    labels[:, 3] = 6
    torch.testing.assert_close(run_mqar_accuracy(logits, labels), torch.ones(3), atol=1e-6, rtol=0)
