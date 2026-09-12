"""First-principles throughput prediction for GB10, from the measured roofline.

The survey's measurements are interesting mainly against a prediction. This module
provides one: for a given model and cell, how fast *should* the hardware go if the
only limits were arithmetic throughput and memory bandwidth?

The core relations, per decode step for a batch of B sequences at context C:

    bytes  = weights_read + B * C * kv_bytes_per_token
    flops  = 2 * active_params * B          (plus attention, small at short C)
    t_step = max(bytes / BW, flops / FLOPS)

Decode arithmetic intensity is therefore approximately B while weights dominate the
read. That is the whole story of batching on this machine: GB10's measured ridge is
~406 FLOP/byte, so decode stays memory-bound until batch is in the hundreds, and
every doubling of batch below that is nearly free throughput.

Prefill is the opposite: it does in_len times more arithmetic for the same weight
read, so it is compute-bound at any realistic length.
"""

from __future__ import annotations

from dataclasses import dataclass

import budget

GIB = 1 << 30


@dataclass(frozen=True)
class Roofline:
    tflops: float          # sustained BF16 matmul, TFLOP/s
    bw_gbps: float         # achievable read bandwidth, GB/s

    def flops(self) -> float:
        return self.tflops * 1e12

    def bw(self) -> float:
        return self.bw_gbps * 1e9

    def ridge_flop_per_byte(self) -> float:
        """Arithmetic intensity above which the machine is compute-bound."""
        return self.flops() / self.bw()

    @classmethod
    def from_json(cls, path: str = "results/roofline.json") -> "Roofline":
        import json
        from pathlib import Path

        d = json.loads(Path(path).read_text())
        return cls(tflops=d["measured_bf16_tflops"], bw_gbps=d["measured_bw_gbps"])


@dataclass(frozen=True)
class Prediction:
    tok_s: float
    bound: str             # "memory" or "compute"
    bytes_per_step: float
    flops_per_step: float
    intensity: float       # FLOP/byte
    t_step_s: float


def _params_from_bytes(weight_bytes: int, dtype_bytes: int = 2) -> float:
    return weight_bytes / dtype_bytes


def predict_decode(
    config: dict,
    weight_bytes: int,
    batch: int,
    ctx: int,
    roof: Roofline,
    active_weight_bytes: int | None = None,
    kv_dtype_bytes: int = 2,
) -> Prediction:
    """Predict decode throughput in output tokens/second.

    `active_weight_bytes` lets an MoE read only its routed experts: at low batch an
    MoE streams far fewer bytes per step than its total size implies. This ignores
    the growth of the routed-expert union as batch rises, so it is an optimistic
    bound for MoE at large batch -- which is exactly the gap worth measuring.
    """
    read_bytes = active_weight_bytes if active_weight_bytes is not None else weight_bytes
    kv_per_tok = budget.kv_bytes_per_token(config, kv_dtype_bytes)

    # Each sequence re-reads its own KV history every step.
    bytes_per_step = read_bytes + batch * ctx * kv_per_tok

    # 2 FLOP per parameter per token (multiply-add), over the active parameters.
    active_params = _params_from_bytes(read_bytes)
    flops_per_step = 2.0 * active_params * batch

    t_mem = bytes_per_step / roof.bw()
    t_cmp = flops_per_step / roof.flops()
    t_step = max(t_mem, t_cmp)

    return Prediction(
        tok_s=batch / t_step,
        bound="memory" if t_mem >= t_cmp else "compute",
        bytes_per_step=bytes_per_step,
        flops_per_step=flops_per_step,
        intensity=flops_per_step / bytes_per_step,
        t_step_s=t_step,
    )


def predict_prefill(
    config: dict,
    weight_bytes: int,
    batch: int,
    in_len: int,
    roof: Roofline,
    active_param_bytes: int | None = None,
    kv_dtype_bytes: int = 2,
) -> Prediction:
    """Predict prefill throughput in input tokens/second."""
    read_bytes = active_param_bytes if active_param_bytes is not None else weight_bytes
    kv_per_tok = budget.kv_bytes_per_token(config, kv_dtype_bytes)

    tokens = batch * in_len
    # Weights are read once and reused across every token in the batch; KV is
    # written rather than read, but costs the same traffic.
    bytes_total = read_bytes + tokens * kv_per_tok
    active_params = _params_from_bytes(read_bytes)
    flops_total = 2.0 * active_params * tokens

    t_mem = bytes_total / roof.bw()
    t_cmp = flops_total / roof.flops()
    t = max(t_mem, t_cmp)

    return Prediction(
        tok_s=tokens / t,
        bound="memory" if t_mem >= t_cmp else "compute",
        bytes_per_step=bytes_total,
        flops_per_step=flops_total,
        intensity=flops_total / bytes_total,
        t_step_s=t,
    )


def ridge_batch(roof: Roofline) -> float:
    """Batch size at which decode arithmetic intensity reaches the ridge.

    Since intensity is approximately batch (weights-dominated), this is just the
    ridge itself -- the headline number for how batch-hungry this machine is.
    """
    return roof.ridge_flop_per_byte()


if __name__ == "__main__":
    roof = Roofline(tflops=95.9, bw_gbps=236.4)
    print(f"measured roofline: {roof.tflops} TFLOP/s, {roof.bw_gbps} GB/s")
    print(f"ridge: {roof.ridge_flop_per_byte():.0f} FLOP/byte "
          f"-> decode stays memory-bound until batch ~{ridge_batch(roof):.0f}\n")
    cfg = {"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128}
    print(f"{'batch':>6} {'ctx':>7} {'pred tok/s':>11} {'bound':>8} {'FLOP/byte':>10}")
    for batch in (1, 4, 16, 64, 256, 1024):
        for ctx in (1024, 8192):
            p = predict_decode(cfg, 15 * GIB, batch, ctx, roof)
            print(f"{batch:>6} {ctx:>7} {p.tok_s:>11.1f} {p.bound:>8} {p.intensity:>10.1f}")


def active_weight_bytes(config: dict, weight_bytes: int) -> int:
    """Bytes an MoE actually reads per token, counting only routed experts.

    A sparse model's decode speed is set by its *active* parameters. Qwen3-30B-A3B
    holds 128 experts per layer but routes each token to 8, so it streams roughly a
    tenth of its 57 GiB. Returns weight_bytes unchanged for dense models.
    """
    n_exp = config.get("num_experts")
    k = config.get("num_experts_per_tok")
    if not n_exp or not k:
        return weight_bytes

    h = int(config["hidden_size"])
    L = int(config["num_hidden_layers"])
    m = int(config.get("moe_intermediate_size", config["intermediate_size"]))
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    hd = budget.head_dim(config)
    vocab = int(config["vocab_size"])

    # gate, up and down projections per expert
    expert = 3 * h * m
    attn = h * heads * hd * 2 + 2 * h * kv_heads * hd   # q, o, then k and v
    embed = h * vocab * (1 if config.get("tie_word_embeddings") else 2)

    total_params = L * (attn + n_exp * expert) + embed
    active_params = L * (attn + k * expert) + embed
    return int(weight_bytes * active_params / total_params)


def effective_bytes_per_step(decode_tok_s: float, batch: int, roof: Roofline) -> float:
    """Invert the bandwidth bound to recover what a measured cell actually read.

    If decode is memory-bound -- which every cell on this machine is -- then
    tok/s = batch / (bytes / BW), so bytes = BW * batch / tok_s. This turns a
    throughput measurement into a statement about traffic, which is what makes the
    MoE's expert-union growth visible.
    """
    return roof.bw() * batch / decode_tok_s


def expert_params(config: dict) -> int:
    """Parameter count held in MoE experts (0 for a dense model)."""
    n_exp = config.get("num_experts")
    if not n_exp:
        return 0
    h = int(config["hidden_size"])
    L = int(config["num_hidden_layers"])
    m = int(config.get("moe_intermediate_size", config["intermediate_size"]))
    return L * n_exp * 3 * h * m


def experts_touched(decode_tok_s: float, batch: int, config: dict,
                    weight_bytes: int, roof: Roofline, ctx: int = 0,
                    dtype_bytes: int = 2) -> float:
    """How many of the layer's experts a measured cell effectively read.

    Derived, not assumed: take the measured traffic, subtract what every step must
    read regardless -- the non-expert weights *and* the KV cache -- and express the
    remainder as a fraction of the full expert bank.

    Subtracting KV matters at large batch: at batch 256 with a 256-token context
    this model's KV traffic is ~6 GB/step, and attributing that to experts pushes
    the derived count above the 128 that physically exist.
    """
    e_bytes = expert_params(config) * dtype_bytes
    if e_bytes == 0:
        return 0.0
    nonexpert = weight_bytes - e_bytes
    kv = batch * ctx * kv_bytes_per_token_for(config, dtype_bytes)
    eff = effective_bytes_per_step(decode_tok_s, batch, roof)
    return (eff - nonexpert - kv) / e_bytes * int(config["num_experts"])


def kv_bytes_per_token_for(config: dict, dtype_bytes: int = 2) -> int:
    return budget.kv_bytes_per_token(config, dtype_bytes)


def experts_touched_if_random(batch: int, config: dict) -> float:
    """Expected distinct experts if every token routed independently at random.

    The reference curve: n * (1 - (1 - k/n)^batch). Real routing is correlated, so
    measurements running below this line are evidence of that correlation.
    """
    n = int(config["num_experts"])
    k = int(config["num_experts_per_tok"])
    return n * (1 - (1 - k / n) ** batch)
