"""Measure GB10's actual roofline: sustained compute and achievable bandwidth.

Benchmark results are reported as a fraction of *these* numbers rather than of
vendor spec sheets. Spec bandwidth (273 GB/s) and derived peak BF16 (~125 TFLOP/s)
are what the hardware is sold as; what a kernel can sustain is what a served model
is actually competing for.

Writes results/roofline.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

GIB = 1 << 30
DEV = "cuda"

# Vendor/derived reference points for context in the writeup.
SPEC_BW_GBPS = 273.0            # 256-bit LPDDR5X @ 8533 MT/s
SPEC_BF16_TFLOPS = 125.0        # 48 SMs x 1024 FLOP/clk/SM x ~2.5 GHz (derived)


def _sync():
    torch.cuda.synchronize()


def _timed(fn, warmup: int, iters: int) -> float:
    """Median seconds per iteration, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def bench_gemm(n: int, dtype, warmup=5, iters=20) -> dict:
    """Sustained GEMM throughput at one square size."""
    a = torch.randn(n, n, device=DEV, dtype=torch.float32).to(dtype)
    b = torch.randn(n, n, device=DEV, dtype=torch.float32).to(dtype)
    if dtype in (torch.float8_e4m3fn,):
        scale = torch.tensor(1.0, device=DEV)
        fn = lambda: torch._scaled_mm(a, b.t().contiguous().t(), scale_a=scale,
                                      scale_b=scale, out_dtype=torch.bfloat16)
    else:
        fn = lambda: a @ b
    sec = _timed(fn, warmup, iters)
    flops = 2.0 * n**3
    del a, b
    torch.cuda.empty_cache()
    return {"n": n, "dtype": str(dtype).split(".")[-1],
            "sec": sec, "tflops": flops / sec / 1e12}


def bench_bandwidth(nbytes: int, warmup=3, iters=10) -> dict:
    """Achievable bandwidth for a large device-to-device copy.

    A copy touches each byte once for read and once for write, so effective
    traffic is 2 x nbytes.
    """
    n = nbytes // 2  # bf16 elements
    src = torch.empty(n, device=DEV, dtype=torch.bfloat16).normal_()
    dst = torch.empty_like(src)
    sec = _timed(lambda: dst.copy_(src), warmup, iters)
    gbps = 2 * nbytes / sec / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return {"bytes": nbytes, "sec": sec, "gbps": gbps}


def bench_read_bandwidth(nbytes: int, warmup=3, iters=10) -> dict:
    """Read-dominated bandwidth, via a full reduction.

    This is the traffic pattern that decoding actually performs: stream the
    weights in, write almost nothing out.
    """
    n = nbytes // 2
    src = torch.empty(n, device=DEV, dtype=torch.bfloat16).normal_()
    sec = _timed(lambda: src.sum(), warmup, iters)
    gbps = nbytes / sec / 1e9
    del src
    torch.cuda.empty_cache()
    return {"bytes": nbytes, "sec": sec, "gbps": gbps}


def main():
    props = torch.cuda.get_device_properties(0)
    out = {
        "device": props.name,
        "sm": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "spec_bw_gbps": SPEC_BW_GBPS,
        "spec_bf16_tflops": SPEC_BF16_TFLOPS,
    }

    print(f"{props.name}  sm_{props.major}{props.minor}  "
          f"{props.multi_processor_count} SMs  {props.total_memory/GIB:.1f} GiB\n")

    print("GEMM (bf16):")
    out["gemm_bf16"] = []
    for n in (1024, 2048, 4096, 8192, 16384):
        r = bench_gemm(n, torch.bfloat16)
        out["gemm_bf16"].append(r)
        print(f"  n={n:<6d} {r['tflops']:7.1f} TFLOP/s")

    print("\nGEMM (fp16):")
    out["gemm_fp16"] = []
    for n in (4096, 8192):
        r = bench_gemm(n, torch.float16)
        out["gemm_fp16"].append(r)
        print(f"  n={n:<6d} {r['tflops']:7.1f} TFLOP/s")

    print("\nGEMM (fp8 e4m3):")
    out["gemm_fp8"] = []
    for n in (4096, 8192):
        try:
            r = bench_gemm(n, torch.float8_e4m3fn)
            out["gemm_fp8"].append(r)
            print(f"  n={n:<6d} {r['tflops']:7.1f} TFLOP/s")
        except Exception as e:
            print(f"  n={n:<6d} unsupported: {type(e).__name__}: {str(e)[:80]}")
            out["gemm_fp8_error"] = f"{type(e).__name__}: {e}"
            break

    print("\nBandwidth (copy, read+write):")
    out["bandwidth_copy"] = []
    for gb in (1, 2, 4):
        r = bench_bandwidth(gb * GIB)
        out["bandwidth_copy"].append(r)
        print(f"  {gb} GiB  {r['gbps']:7.1f} GB/s")

    print("\nBandwidth (read-only reduction):")
    out["bandwidth_read"] = []
    for gb in (1, 2, 4):
        r = bench_read_bandwidth(gb * GIB)
        out["bandwidth_read"].append(r)
        print(f"  {gb} GiB  {r['gbps']:7.1f} GB/s")

    peak_tflops = max(r["tflops"] for r in out["gemm_bf16"])
    peak_bw = max(r["gbps"] for r in out["bandwidth_copy"] + out["bandwidth_read"])
    out["measured_bf16_tflops"] = peak_tflops
    out["measured_bw_gbps"] = peak_bw
    out["ridge_flop_per_byte"] = peak_tflops * 1e12 / (peak_bw * 1e9)

    print(f"\nmeasured peak BF16   {peak_tflops:.1f} TFLOP/s "
          f"({peak_tflops/SPEC_BF16_TFLOPS:.0%} of derived spec)")
    print(f"measured peak BW     {peak_bw:.1f} GB/s "
          f"({peak_bw/SPEC_BW_GBPS:.0%} of spec)")
    print(f"roofline ridge       {out['ridge_flop_per_byte']:.0f} FLOP/byte")

    Path("results").mkdir(exist_ok=True)
    Path("results/roofline.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/roofline.json")


if __name__ == "__main__":
    main()
