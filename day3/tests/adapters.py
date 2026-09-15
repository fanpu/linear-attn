"""Adapters connecting the tests to your implementations.  [you write the glue]

Each run_<slug> function is called by tests/test_<slug>.py. Replace each
`raise NotImplementedError` with a call into the testbed package. Do not edit the
test files; if a signature differs, the fix is here.
"""

from __future__ import annotations

import numpy as np
import torch

from testbed.data_check import check_disjoint


def run_lr_at(
    step: int, total_steps: int, peak_lr: float, warmup_frac: float, final_frac: float
) -> float:
    """Return the learning rate for batch `step` of a run of `total_steps` batches."""
    raise NotImplementedError


def run_evaluate(model, val: torch.Tensor, B: int, ctx) -> float:
    """Return the token-weighted mean next-token NLL over `val` in batches of B."""
    raise NotImplementedError


def run_ref_nll(model, val: torch.Tensor) -> float:
    """Return the naive, unbatched mean next-token NLL over `val`."""
    raise NotImplementedError


def run_check_disjoint(train_shards: list[np.ndarray], val: np.ndarray, T: int) -> int:
    """Return the number of validation windows that equal an aligned training window."""
    return check_disjoint(train_shards, val, T)


def run_seed_stats(losses_by_size: dict[str, list[float]]) -> dict:
    """Return {"s_pooled", "nu", "lo", "hi", "mdd"} for the given final losses."""
    raise NotImplementedError
