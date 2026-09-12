"""Pre-flight memory model for GB10 inference benchmarking.

GB10 has *unified* memory: the 121.7 GiB pool is shared between host and GPU, and
there is no separate framebuffer (`nvidia-smi` reports `memory.used = N/A`). An
over-allocation therefore starves the host rather than raising a clean CUDA OOM,
which on this machine can mean an OOM-kill hang.

So every benchmark cell is costed *before* any engine is constructed, and a cell
that does not fit under the ceiling is skipped rather than attempted. Nothing here
touches the GPU or imports torch; it is pure arithmetic so it can be tested and
trusted independently of the benchmark itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30

# Fraction of total system memory any single run may claim. 0.60 of 121.7 GiB is
# ~73 GiB, leaving the host ~48 GiB. Deliberately conservative: the cost of being
# wrong is a wedged machine, and the cost of being too careful is a few skipped
# cells at the top of the batch sweep.
CEILING_FRAC = 0.60

# Memory vLLM needs beyond weights and KV cache: activation peak during profiling,
# CUDA graph pools, NCCL/cuBLAS workspaces, and the CUDA context itself. Measured
# empirically at ~2-4 GiB on this stack; 6 GiB is a safety-biased default, since
# over-estimating only costs us skipped cells while under-estimating risks the host.
DEFAULT_OVERHEAD_BYTES = 6 * GIB


def total_memory_bytes() -> int:
    """Total system memory, which on unified-memory GB10 is also the GPU pool."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def available_memory_bytes() -> int:
    """MemAvailable from /proc/meminfo.

    Used instead of NVML because GB10 exposes no framebuffer usage counter.
    """
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def head_dim(config: dict) -> int:
    """Per-head dimension.

    Qwen3 sets `head_dim` explicitly and it is *not* hidden_size // num_heads
    (0.6B: 1024/16 = 64, but head_dim is 128), so the explicit field must win.
    Getting this wrong understates KV cache by 2x on exactly the models we care about.
    """
    if config.get("head_dim"):
        return int(config["head_dim"])
    return int(config["hidden_size"]) // int(config["num_attention_heads"])


def kv_bytes_per_token(config: dict, dtype_bytes: int = 2) -> int:
    """Bytes of KV cache per token of context, for one sequence.

    Counts key *and* value, and uses num_key_value_heads (GQA) rather than the
    query head count.
    """
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
    return 2 * layers * kv_heads * head_dim(config) * dtype_bytes


def kv_bytes(config: dict, batch: int, total_len: int, dtype_bytes: int = 2) -> int:
    """Total KV cache for `batch` sequences each holding `total_len` tokens."""
    return kv_bytes_per_token(config, dtype_bytes) * batch * total_len


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: dict
    weight_bytes: int

    @classmethod
    def from_hf_cache(cls, repo_id: str) -> "ModelSpec":
        """Build a spec from an already-downloaded HF snapshot.

        Weight size is taken from the actual safetensors files on disk rather than
        derived from the config: that is exact, and it handles tied embeddings,
        MoE expert counts and pre-quantised checkpoints without special cases.
        """
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id, local_files_only=True))
        config = json.loads((path / "config.json").read_text())
        weight_bytes = sum(f.stat().st_size for f in path.glob("*.safetensors"))
        if weight_bytes == 0:
            raise RuntimeError(f"no safetensors found for {repo_id} in {path}")
        return cls(repo_id, config, weight_bytes)


@dataclass(frozen=True)
class Budget:
    weight_bytes: int
    kv_bytes: int
    overhead_bytes: int
    total_bytes: int
    total_mem_bytes: int
    required_util: float
    gpu_memory_utilization: float
    fits: bool
    reason: str

    def summary(self) -> str:
        return (
            f"weights={self.weight_bytes/GIB:.1f}G kv={self.kv_bytes/GIB:.1f}G "
            f"overhead={self.overhead_bytes/GIB:.1f}G total={self.total_bytes/GIB:.1f}G "
            f"util={self.gpu_memory_utilization:.3f} fits={self.fits}"
        )


def plan_cell(
    spec: ModelSpec,
    batch: int,
    total_len: int,
    total_mem_bytes: int | None = None,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
    kv_dtype_bytes: int = 2,
    ceiling_frac: float = CEILING_FRAC,
) -> Budget:
    """Cost one benchmark cell and decide whether it may run.

    A cell that does not fit is *refused*, never silently shrunk to fit: quietly
    reducing the batch would emit a results row labelled with a batch size that
    was never actually executed.
    """
    if total_mem_bytes is None:
        total_mem_bytes = total_memory_bytes()

    kv = kv_bytes(spec.config, batch, total_len, kv_dtype_bytes)
    total = spec.weight_bytes + kv + overhead_bytes
    required_util = total / total_mem_bytes
    ceiling_bytes = int(ceiling_frac * total_mem_bytes)

    fits = total <= ceiling_bytes
    reason = (
        "ok"
        if fits
        else (
            f"exceeds ceiling: needs {total/GIB:.1f} GiB > "
            f"{ceiling_bytes/GIB:.1f} GiB ({ceiling_frac:.0%} of "
            f"{total_mem_bytes/GIB:.1f} GiB)"
        )
    )

    # vLLM interprets gpu_memory_utilization as a fraction of *total* memory and
    # carves KV cache out of whatever remains after weights and activations. Ask
    # for what this cell needs plus a small margin, clamped to the ceiling: asking
    # for less than weights+activation makes vLLM refuse to start.
    util = min(ceiling_frac, required_util * 1.05) if fits else 0.0

    return Budget(
        weight_bytes=spec.weight_bytes,
        kv_bytes=kv,
        overhead_bytes=overhead_bytes,
        total_bytes=total,
        total_mem_bytes=total_mem_bytes,
        required_util=required_util,
        gpu_memory_utilization=util,
        fits=fits,
        reason=reason,
    )


if __name__ == "__main__":
    import sys

    total = total_memory_bytes()
    print(f"total memory      {total/GIB:.1f} GiB")
    print(f"available now     {available_memory_bytes()/GIB:.1f} GiB")
    print(f"ceiling ({CEILING_FRAC:.0%})     {CEILING_FRAC*total/GIB:.1f} GiB\n")
    for repo in sys.argv[1:]:
        try:
            spec = ModelSpec.from_hf_cache(repo)
        except Exception as e:
            print(f"{repo:24s} unavailable: {type(e).__name__}")
            continue
        per_tok = kv_bytes_per_token(spec.config)
        print(f"{repo:24s} weights={spec.weight_bytes/GIB:5.1f} GiB  "
              f"kv={per_tok/1024:6.1f} KiB/token")
        for batch in (1, 8, 32, 128, 256):
            p = plan_cell(spec, batch=batch, total_len=4096)
            print(f"    b={batch:<4d} len=4096  {p.summary()}")
