"""Block-scaled 4-bit formats implemented from the published descriptions (not from a library).

MXFP4  (OCP Microscaling Formats spec v1.0, sec. 5-6; Rouhani et al. 2023, Algorithm 1)
  - block size k = 32, elements E2M1, shared scale X in E8M0 (power of two, bias 127, exps -127..127, 0xFF = NaN)
  - shared_exp = floor(log2(max_i |V_i|)) - emax_elem,   emax_elem = 2 for E2M1  (largest E2M1 binade: 4..6)
  - P_i = quantize_to_E2M1(V_i / X), normal values beyond the max (6.0) are clamped to +-6 (sign kept)
  - element rounding: round-to-nearest, ties-to-even (the MX default; the MX paper uses it for inference)
  Consequence (faithful to the spec, and visible in the art): max|V|/X lies in [4, 8), so any block whose
  max lands in (6, 8)*X has its largest elements clipped.

NVFP4  (NVIDIA: "Pretraining LLMs with NVFP4", arXiv:2509.25149; TensorRT Model Optimizer nvfp4_tensor.py)
  - block size 16, elements E2M1, block scale in FP8 E4M3 (E4M3FN: bias 7, max 448, min subnormal 2^-9),
    plus one per-tensor FP32 scale  s2 = amax_tensor / (6 * 448)
  - block scale  s_b = amax_block / (6 * s2), cast to E4M3 with round-to-nearest-even, clamped to [2^-9, 448]
    (all-zero block -> 1.0, as in ModelOpt)
  - element  q_i = E2M1(V_i / (s_b_e4m3 * s2)), saturating at 6, round-to-nearest-even
  - dequant  V_i ~= q_i * s_b_e4m3 * s2

Rounding modes offered for the E2M1 elements: 'nearest' (ties-to-even) and 'stochastic'
(round up with probability (x-lo)/(hi-lo); unbiased inside the range, clamps outside). Scales are always
computed deterministically (RTNE for E4M3, floor rule for E8M0) in both modes.

Blocks run along the LAST axis of the matrix (the in_features axis of an nn.Linear weight), as in
ModelOpt / MX reference implementations. All arithmetic is float64 on CPU.
"""
import numpy as np
import torch

E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
E2M1_MAX = 6.0
E4M3_MAX = 448.0


def e4m3_grid():
    """All non-negative finite float8_e4m3fn values, index = (exponent_bits << 3) | mantissa_bits."""
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:      # 0b1111_111 is NaN in E4M3FN (no infinities)
                continue
            vals.append((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return torch.tensor(vals, dtype=torch.float64)


E4M3_GRID = e4m3_grid()


def round_to_grid(x, grid, mode="nearest", generator=None):
    """x >= 0 (float64 tensor). Returns integer index into sorted non-negative `grid`.
    Saturates at grid[-1]. nearest: ties go to the even index (mantissa LSB = 0 for E2M1 and E4M3)."""
    g = grid.to(x.device)
    x = torch.clamp(x, max=float(g[-1]))
    hi = torch.searchsorted(g, x.contiguous(), right=False).clamp(1, len(g) - 1)  # g[hi] >= x
    lo = hi - 1
    glo, ghi = g[lo], g[hi]
    exact_hi = x == ghi
    if mode == "nearest":
        dlo, dhi = x - glo, ghi - x
        pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    elif mode == "stochastic":
        p = (x - glo) / (ghi - glo)
        u = torch.rand(x.shape, dtype=torch.float64, generator=generator, device=x.device)
        pick_hi = u < p
    else:
        raise ValueError(mode)
    idx = torch.where(pick_hi | exact_hi, hi, lo)
    idx = torch.where(x <= g[0], torch.zeros_like(idx), idx)
    return idx


def _blocks(W, bs):
    W = torch.as_tensor(W, dtype=torch.float64)
    lead = W.shape[:-1]
    n = W.shape[-1]
    pad = (-n) % bs
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    return W.reshape(*lead, (n + pad) // bs, bs), lead, n


def _unblock(B, lead, n):
    return B.reshape(*lead, -1)[..., :n]


def quant_e2m1_elements(Bx, scale, mode, generator):
    """Bx: (..., nb, bs) block values; scale: (..., nb, 1) effective (dequant) scale."""
    y = Bx / scale
    idx = round_to_grid(y.abs(), E2M1_LEVELS, mode, generator)
    sign = torch.signbit(y)
    q = E2M1_LEVELS[idx] * torch.where(sign, -1.0, 1.0)
    clipped = y.abs() > E2M1_MAX
    return q * scale, idx, sign, clipped


def mxfp4(W, block_size=32, mode="nearest", seed=0):
    B, lead, n = _blocks(W, block_size)
    gen = torch.Generator().manual_seed(seed) if mode == "stochastic" else None
    amax = B.abs().amax(-1, keepdim=True)
    FP32_MIN_NORMAL = 2.0 ** -126
    _, e = torch.frexp(torch.where(amax == 0, torch.full_like(amax, FP32_MIN_NORMAL), amax))
    shared_exp = (e - 1).to(torch.float64) - 2.0          # floor(log2 amax) - emax_elem(E2M1)=2 ; frexp is exact
    if (shared_exp > 127).any():
        raise ValueError("shared exponent above E8M0 range -> NaN per spec")
    shared_exp = shared_exp.clamp(min=-127)
    X = torch.pow(2.0, shared_exp)
    deq, idx, sign, clipped = quant_e2m1_elements(B, X, mode, gen)
    return dict(
        deq=_unblock(deq, lead, n), idx=_unblock(idx, lead, n).to(torch.uint8),
        sign=_unblock(sign, lead, n), clipped=_unblock(clipped, lead, n),
        scale=X[..., 0], scale_code=(shared_exp[..., 0] + 127).to(torch.uint8),   # E8M0 byte
        amax=amax[..., 0], block_size=block_size, fmt="MXFP4")


def nvfp4(W, block_size=16, mode="nearest", seed=0, tensor_amax=None):
    B, lead, n = _blocks(W, block_size)
    gen = torch.Generator().manual_seed(seed) if mode == "stochastic" else None
    t_amax = B.abs().max() if tensor_amax is None else torch.tensor(float(tensor_amax), dtype=torch.float64)
    s2 = torch.tensor(np.float32(float(t_amax) / (E2M1_MAX * E4M3_MAX)), dtype=torch.float64)  # FP32 tensor scale
    amax = B.abs().amax(-1, keepdim=True)
    sb = amax / (E2M1_MAX * s2)
    sb = torch.where(amax == 0, torch.ones_like(sb), sb)
    sb = sb.clamp(2.0 ** -9, E4M3_MAX)
    sidx = round_to_grid(sb, E4M3_GRID, "nearest")       # scale cast: RTNE, always deterministic
    sbq = E4M3_GRID[sidx]
    scale = sbq * s2
    deq, idx, sign, clipped = quant_e2m1_elements(B, scale, mode, gen)
    return dict(
        deq=_unblock(deq, lead, n), idx=_unblock(idx, lead, n).to(torch.uint8),
        sign=_unblock(sign, lead, n), clipped=_unblock(clipped, lead, n),
        scale=scale[..., 0], scale_e4m3=sbq[..., 0], scale_code=sidx[..., 0].to(torch.uint8), s2=float(s2),
        amax=amax[..., 0], block_size=block_size, fmt="NVFP4")


def fp8_e4m3_per_tensor(W, mode="nearest", seed=0):
    """Per-tensor scaled FP8 E4M3 (scale = amax/448), for the FP8-code bitplanes."""
    W = torch.as_tensor(W, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed) if mode == "stochastic" else None
    s = float(W.abs().max()) / E4M3_MAX
    idx = round_to_grid(W.abs() / s, E4M3_GRID, mode, gen)
    sign = torch.signbit(W)
    deq = E4M3_GRID[idx] * torch.where(sign, -1.0, 1.0) * s
    byte = (sign.to(torch.int32) << 7) | idx.to(torch.int32)   # index == (exp<<3)|mant == E4M3FN bit pattern
    return dict(deq=deq, byte=byte.to(torch.uint8), scale=s)


def fp4_nibbles(res):
    """4-bit E2M1 code: sign<<3 | (exp<<1 | mant) ; the level index IS the 3-bit E2M1 pattern."""
    return (res["sign"].to(torch.uint8) << 3) | res["idx"]


QUANTIZERS = {"NVFP4": nvfp4, "MXFP4": mxfp4}
