"""
Day 2: throughput + roofline table, attention (SDPA) vs Gated DeltaNet, on GB10.

Ownership (per RESEARCH_PROGRAM.md, loud vs silent):
  Claude (loud):   roofline microbench, model plumbing, timing loop, warmup, logging, plotting.
  Fan Pu (silent): flops_per_token, bytes_per_token, state_size_per_layer (stubs below),
                   and verifying the printed param counts against your own counter.

Usage:
  python bench_throughput.py roofline                    # measured matmul TFLOP/s and copy GB/s
  python bench_throughput.py sdpa-check                  # which SDPA backends actually run on sm_121
  python bench_throughput.py run --mixer gdn --size 60M  # one row -> results.jsonl
  python bench_throughput.py sweep                       # 2 mixers x 4 sizes, fresh process per row
  python bench_throughput.py plot                        # log-log tok/s vs params, fitted slopes

Untested on hardware. Failures here should be loud (crash / obviously wrong numbers); report them.
"""
import argparse, json, math, os, subprocess, sys, time

import torch
import torch.nn.functional as F

RESULTS = "results.jsonl"
ROOFLINE = "roofline.json"
DEV = "cuda"

# ----------------------------------------------------------------------------------------------
# PROPOSED shapes. Verify every printed param count with your own counter before trusting a row.
# GDN: head_dim=64, expand_v=2 -> (d_k, d_v) = (64, 128), a Day-1-verified head shape,
#      num_heads = 0.75 * d / 64 so that num_heads * head_dim = 0.75 * d (fla's intended allocation).
# Attention: head_dim = 64 (fla sets head_dim = hidden_size / num_heads).
# NOTE: fixing head_dim=64 across sizes means state/d^2 shrinks as d grows. That is a D2 decision;
#       it is provisional here and irrelevant to throughput.
# ----------------------------------------------------------------------------------------------
SHAPES = {  # size: (hidden_size, num_layers)
    "30M": (512, 10),
    "60M": (768, 9),
    "125M": (768, 18),
    "250M": (1024, 20),
}


def build_config(mixer, size, vocab=32000):
    from fla.models import GatedDeltaNetConfig, TransformerConfig
    d, L = SHAPES[size]
    common = dict(hidden_size=d, num_hidden_layers=L, vocab_size=vocab, max_position_embeddings=4096,
                  tie_word_embeddings=False, fuse_cross_entropy=True)
    if mixer == "attn":
        return TransformerConfig(num_heads=d // 64, **common)
    if mixer == "gdn":
        nh = (3 * d) // 256
        assert nh * 64 == int(0.75 * d), f"0.75*d={0.75 * d} not divisible by 64"
        return GatedDeltaNetConfig(head_dim=64, expand_v=2.0, num_heads=nh, **common)
    raise ValueError(mixer)


def build_model(mixer, size):
    from fla.models import GatedDeltaNetForCausalLM, TransformerForCausalLM
    cfg = build_config(mixer, size)
    cls = TransformerForCausalLM if mixer == "attn" else GatedDeltaNetForCausalLM
    return cls(cfg).to(DEV), cfg


# ================== FAN PU OWNS THESE. Return None until written; the table leaves blanks. ==================
def flops_per_token(cfg, mixer, T):
    """Training FLOPs per token (fwd+bwd). State your convention (causal halving? LM head? GDN intra-chunk?)."""
    return None


def bytes_per_token(cfg, mixer, T, tokens_per_step):
    """Estimated DRAM bytes per token for one training step. Weight/optimizer terms divide by tokens_per_step."""
    return None


def state_size_per_layer(cfg, mixer, T):
    """Recurrent state elements per layer (GDN) / KV-cache elements per layer at length T (attention)."""
    return None
# =================================================================================================================


# ---------------------------------- SDPA shim for fla's Attention ----------------------------------
def _sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    # fla calls flash_attn_func(q, k, v, causal=True, window_size=...) with q,k,v: [B, T, H, D]
    assert tuple(window_size) == (-1, -1), "shim supports full causal attention only"
    assert not kwargs, f"unsupported kwargs: {list(kwargs)}"
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> [B, H, T, D]
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=(q.shape[1] != k.shape[1]))
    return o.transpose(1, 2)  # -> [B, T, H, D]


def install_sdpa_shim():
    import fla.layers.attn as fla_attn
    fla_attn.flash_attn_func = _sdpa_flash_attn_func  # looked up at call time, so patching the module global works


def sdpa_check():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(4, 8, 1024, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    for b in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel([b]):
                F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
            torch.cuda.synchronize()
            print(f"{b.name:22s} ok")
        except Exception as e:
            print(f"{b.name:22s} FAIL {type(e).__name__}: {str(e)[:100]}")


# ---------------------------------- roofline microbenchmarks ----------------------------------
def _time(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def roofline():
    n = 8192
    a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    t_mm = _time(lambda: a @ b, iters=20)
    tflops = 2 * n ** 3 / t_mm / 1e12
    del a, b
    x = torch.empty(2 * 2 ** 30, device=DEV, dtype=torch.uint8)  # 2 GiB, far larger than any cache
    y = torch.empty_like(x)
    t_cp = _time(lambda: y.copy_(x), iters=20)
    gbs = 2 * x.numel() / t_cp / 1e9  # read + write
    out = dict(device=torch.cuda.get_device_name(), bf16_matmul_tflops=tflops, copy_GBps=gbs,
               ridge_flop_per_byte_measured=tflops * 1e12 / (gbs * 1e9),
               ridge_flop_per_byte_nominal=104e12 / 273e9)
    json.dump(out, open(ROOFLINE, "w"), indent=2)
    print(json.dumps(out, indent=2))


# ---------------------------------- one benchmark row ----------------------------------
def param_buckets(model):
    total = emb = head = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if "embeddings" in name:
            emb += p.numel()
        elif "lm_head" in name:
            head += p.numel()
    return dict(params_total=total, params_embed=emb, params_lm_head=head, params_body=total - emb - head)


def run(mixer, size, B, T, warmup, iters):
    install_sdpa_shim()
    torch.manual_seed(0)
    model, cfg = build_model(mixer, size)
    model.train()
    pb = param_buckets(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(warmup):  # includes Triton autotuning on the first calls
        loss = step()
    assert torch.isfinite(loss), "non-finite loss during warmup"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    tokens = B * T
    tps = tokens / dt
    roof = json.load(open(ROOFLINE)) if os.path.exists(ROOFLINE) else {}
    fpt = flops_per_token(cfg, mixer, T)
    bpt = bytes_per_token(cfg, mixer, T, tokens)
    row = dict(mixer=mixer, size=size, d=cfg.hidden_size, L=cfg.num_hidden_layers, B=B, T=T,
               step_s=dt, tok_per_s=tps, us_per_token=1e6 / tps,
               peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9, **pb,
               state_per_layer=state_size_per_layer(cfg, mixer, T),
               flops_per_token=fpt, bytes_per_token=bpt,
               torch=torch.__version__)
    if fpt is not None:
        achieved = fpt * tps
        row["mfu_vs_104T"] = achieved / 104e12
        if roof:
            row["mfu_vs_measured"] = achieved / (roof["bf16_matmul_tflops"] * 1e12)
            if achieved > roof["bf16_matmul_tflops"] * 1e12:
                print("!! implied FLOP rate exceeds measured matmul peak: flops_per_token overcounts")
    if bpt is not None:
        bw = bpt * tps
        row["implied_GBps"] = bw / 1e9
        row["bw_util_vs_273"] = bw / 273e9
        if roof and bw > roof["copy_GBps"] * 1e9:
            print("!! implied bandwidth exceeds measured copy bandwidth: bytes_per_token overcounts")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


def sweep(B, T, warmup, iters):
    for size in SHAPES:
        for mixer in ("attn", "gdn"):
            cmd = [sys.executable, __file__, "run", "--mixer", mixer, "--size", size,
                   "--B", str(B), "--T", str(T), "--warmup", str(warmup), "--iters", str(iters)]
            print(">>", " ".join(cmd), flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!! row {mixer}/{size} failed (rc={r.returncode}); continuing", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [json.loads(l) for l in open(RESULTS)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mixer, c in (("attn", "C0"), ("gdn", "C1")):
        rs = sorted([r for r in rows if r["mixer"] == mixer], key=lambda r: r["params_body"])
        if len(rs) < 2:
            continue
        P = np.array([r["params_body"] for r in rs], float)
        us = np.array([r["us_per_token"] for r in rs], float)
        slope = np.polyfit(np.log(P), np.log(us), 1)[0]
        ax[0].loglog(P / 1e6, us, "o-", color=c, label=f"{mixer}  slope={slope:.2f}")
        ax[1].semilogx(P / 1e6, [r["peak_mem_GB"] for r in rs], "o-", color=c, label=mixer)
    ax[0].set_xlabel("non-embedding, non-head params (M)"); ax[0].set_ylabel("µs per token (train step)")
    ax[0].set_title("time/token vs size: slope ~1 compute-bound, <1 bandwidth-bound, ~0 overhead-bound")
    ax[1].set_xlabel("non-embedding, non-head params (M)"); ax[1].set_ylabel("peak allocated (GB)")
    for a in ax:
        a.legend(); a.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig("day2_throughput.png", dpi=150)
    print("wrote day2_throughput.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["roofline", "sdpa-check", "run", "sweep", "plot"])
    ap.add_argument("--mixer", choices=["attn", "gdn"])
    ap.add_argument("--size", choices=list(SHAPES))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    {"roofline": roofline, "sdpa-check": sdpa_check, "plot": plot,
     "run": lambda: run(a.mixer, a.size, a.B, a.T, a.warmup, a.iters),
     "sweep": lambda: sweep(a.B, a.T, a.warmup, a.iters)}[a.cmd]()
