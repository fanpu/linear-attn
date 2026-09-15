"""Deterministic per-element uniforms without torch RNG state: an integer hash of (index, seed, stream)."""
import torch

_M = 0xFFFFFFFF


def hash01(idx, seed=0, stream=0, dtype=torch.float64):
    """u in [0, 1) for each integer in `idx`; the same (idx, seed, stream) always gives the same u."""
    key = (int(seed) * 0x632BE5AB + int(stream) * 0x85157AF5 + 0x9E3779B9) & _M
    x = ((idx.long() & _M) * 0x7FEB352D + key) & _M          # < 2**63: no int64 overflow
    for _ in range(3):
        x = (((x >> 16) ^ x) * 0x45D9F3B) & _M
    x = (x >> 16) ^ x
    return x.to(dtype) / 4294967296.0
