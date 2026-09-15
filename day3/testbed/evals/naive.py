"""The naive reference for evaluate.  [you]

This is the slow, unbatched version. It is the reference precisely because it
performs the next-token shift by hand and so cannot inherit a mistake from the
model's internal shift.
"""
import torch


def ref_nll(model, val: torch.Tensor) -> float:
    """Mean next-token NLL over every window and position, in nats, as a Python float.

    val: int64 [N, T]. For every window w and every position t in 0 .. T-2:
        logits = model(input_ids=w[None]).logits[0, t]        (no labels passed)
        nll    = logsumexp(logits) - logits[w[t + 1]]           (in fp32)
    Return the mean of the N * (T - 1) values.
    """
    raise NotImplementedError
