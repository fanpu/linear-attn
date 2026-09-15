"""Validation loss.  [you]"""
import torch


@torch.no_grad()
def evaluate(model, val: torch.Tensor, B: int, ctx) -> float:
    """Token-weighted mean next-token NLL, in nats, over the windows in `val`.

    val:  int64 [N, T]; windows are fed in batches of B along the first axis.
    ctx:  a context manager the forward pass runs inside (the training script
          passes the SDPA Flash context; the test passes contextlib.nullcontext()).
    Forward call, the same one the training loop makes:
        with ctx, torch.autocast(device_type=val.device.type, dtype=torch.bfloat16,
                                 enabled=val.is_cuda):
            out = model(input_ids=x, labels=x, use_cache=False)
        out.loss is the mean NLL over the B_batch * (T - 1) valid positions of that
        batch; the model shifts the labels itself (do NOT shift them here).
    Weight each batch's loss by its number of valid positions, B_batch * (T - 1),
    so that a short final batch (N not a multiple of B) is counted correctly.
    Call model.eval() on entry and model.train() before returning.
    Return a Python float.
    """
    raise NotImplementedError
