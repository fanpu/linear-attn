"""
Day 2: throughput + roofline table, attention (SDPA) vs Gated DeltaNet, on GB10.

Ownership (per RESEARCH_PROGRAM.md, loud vs silent):
  Claude (loud):   roofline microbench, model plumbing, timing loop, warmup, logging, plotting.
  Fan Pu (silent): flops_per_token, bytes_per_token, state_size_per_layer (stubs below),
                   and verifying the printed param counts against your own counter.

Usage:
  python bench_throughput.py roofline                        # sustained matmul TFLOP/s, copy GB/s, SM clock
  python bench_throughput.py sdpa-check                      # which SDPA backends run on sm_121
  python bench_throughput.py params                          # param buckets, all 8 configs, no GPU work
  python bench_throughput.py named-params --mixer gdn --size 60M   # every tensor: name, shape, count
  python bench_throughput.py run --mixer gdn --size 60M      # one row -> results.jsonl
  python bench_throughput.py sweep                           # preheat, then 8 rows, fresh process each
  python bench_throughput.py plot                            # log-log µs/token vs params, fitted slopes

Debug loop: `run --warmup-s 5 --time-s 5` gives a number in seconds. Use `sweep` for the real thing.
NOTE: results.jsonl is append-only and `plot` reads every line. Move it aside before a clean sweep.

Attention backend defaults to FLASH_ATTENTION: measured fastest at H in {8, 12, 16}, T=1024, d_h=64
on this box (10-13% over CUDNN, 22-25% over EFFICIENT). Pinned so rows stay reproducible across
torch upgrades, and so bytes_per_token can assume a fused kernel that never writes T x T scores.

Timing is by wall-clock duration, not step count: small models finish 30 steps in seconds, which is
too short to reach thermal steady state and gives unequal measurement windows across sizes.
Per-step times come from CUDA events (no per-step host sync, so CPU launch-ahead is preserved).
"""

import argparse, json, os, subprocess, sys, time

import torch
import torch.nn.functional as F

RESULTS = "results.jsonl"
ROOFLINE = "roofline.json"
DEV = "cuda"

PEAK_BF16_TFLOPS = (
    118.8  # 48 SMs x 1024 FLOP/clk/SM x 2.42 GHz (derived, not NVIDIA-published)
)
PEAK_BW_GBPS = 273.0  # 256-bit LPDDR5x @ 8533 MT/s
DEFAULT_SDPA_BACKEND = "FLASH_ATTENTION"

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
    common = dict(
        hidden_size=d,
        num_hidden_layers=L,
        vocab_size=vocab,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        fuse_cross_entropy=True,
    )
    if mixer == "attn":
        return annotate_shapes(TransformerConfig(num_heads=d // 64, **common), mixer)
    if mixer == "gdn":
        nh = (3 * d) // 256
        assert nh * 64 == int(0.75 * d), f"0.75*d={0.75 * d} not divisible by 64"
        cfg = GatedDeltaNetConfig(head_dim=64, expand_v=2.0, num_heads=nh, **common)
        return annotate_shapes(cfg, mixer)
    raise ValueError(mixer)


def build_model(mixer, size):
    from fla.models import GatedDeltaNetForCausalLM, TransformerForCausalLM

    cfg = build_config(mixer, size)
    cls = TransformerForCausalLM if mixer == "attn" else GatedDeltaNetForCausalLM
    return cls(cfg).to(DEV), cfg


def mlp_intermediate_size(cfg):
    """fla computes this at layer-construction time, so cfg.intermediate_size is usually None."""
    m = cfg.intermediate_size
    if m is None:
        m = int(cfg.hidden_size * (cfg.hidden_ratio or 4) * 2 / 3)
        m = 256 * ((m + 255) // 256)  # round UP to a multiple of 256
    return m


def annotate_shapes(cfg, mixer):
    """Resolve the per-head / MLP shapes that fla only computes at layer-construction time.

    The analytic models below want one object to read shapes off, but fla's configs do not
    carry head_k_dim / head_v_dim / ffn_intermediate -- the layer modules do. These rules
    mirror what the modules actually build, checked against every one of the 8 configs:
      attn: head_dim = hidden_size / num_heads, and dk == dv (no expansion).
      gdn:  dk = head_dim, dv = head_dim * expand_v.
    """
    if mixer == "attn":
        cfg.head_k_dim = cfg.head_v_dim = cfg.hidden_size // cfg.num_heads
        cfg.conv_size = 0  # attention has no short conv; keeps p_conv well-defined
    elif mixer == "gdn":
        cfg.head_k_dim = cfg.head_dim
        cfg.head_v_dim = int(cfg.head_dim * cfg.expand_v)
    else:
        raise ValueError(mixer)
    cfg.ffn_intermediate = mlp_intermediate_size(cfg)
    cfg.tie_embeddings = cfg.tie_word_embeddings
    return cfg


def flops_per_token(cfg, mixer, T, chunk_size=64, return_breakdown=True):
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    H, dk, dv = cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim

    # ---- (a) MLP: identical for both mixers, so it can never explain a gap ----
    # SwiGLU has 3 matrices (gate, up, down), each d x d_ff.
    # Classic 2-matrix MLP: use 2 * d * ffn_intermediate instead.
    p_mlp = 3 * d * cfg.ffn_intermediate

    # ---- (b) token-mixing weights ----
    if mixer == "attn":
        # W_q, W_k : d x (H*dk)     W_v, W_o : d x (H*dv)
        # QK^T and AV have NO weights -- activation x activation. They scale
        # with T and are handled in (c).
        p_mix = 2 * (d * H * dk) + 2 * (d * H * dv)
        p_conv = 0

    elif mixer == "gdn":
        p_mix = 2 * (d * H * dk)  # W_q, W_k
        p_mix += 2 * (d * H * dv)  # W_v, W_o  <- both inflated by expand_v
        p_mix += 1 * (d * H * dv)  # W_gate    <- attention has no analogue
        p_mix += 2 * (d * H)  # a_proj (decay), b_proj (beta): ~1e-4 of total
        # depthwise causal conv over the q, k, v paths
        p_conv = cfg.conv_size * (2 * H * dk + H * dv)
    else:
        raise ValueError(mixer)

    # ---- (c) mixer core: the T-dependent, weight-free part ----
    if mixer == "attn":
        # Per token, per layer, forward:
        #   QK^T : 2 * T * (H*dk)     each query dots against T keys
        #   AV   : 2 * T * (H*dv)     weighted sum over T values
        #   = 4Td  (when H*dk = H*dv = d)
        # fwd+bwd = 12Td.  Causal halving -> 6Td.  (Valid because flash/SDPA
        # skips fully-masked blocks; a dense-mask implementation would not halve.)
        core = 6 * T * d * L
    else:
        # Chunkwise GDN trades FLOPs for parallelism: inside a chunk of size C
        # it does attention-like C x C work; between chunks it passes the state.
        # Per head, per token, forward:
        #   2*C*dk   intra-chunk QK^T
        #   2*C*dk   WY / UT transform (the delta-rule correction)
        #   2*C*dv   intra-chunk AV
        #   4*dk*dv  state readout + state update
        fwd = H * (4 * chunk_size * dk + 2 * chunk_size * dv + 4 * dk * dv)
        core = 3 * fwd * L  # x3 for fwd+bwd

    parts = {
        "token_mix": 6 * L * p_mix,
        "mlp": 6 * L * p_mlp,
        "conv": 6 * L * p_conv,
        "lm_head": 6 * d * V,  # embeddings are a gather: 0 FLOPs
        "core": core,
    }
    return parts if return_breakdown else sum(parts.values())


def bytes_per_token(
    cfg,
    mixer,
    T,
    tokens_per_step,
    chunk_size=64,
    act_elems_per_layer=18,
    attn_block=128,
    return_breakdown=True,
):
    d, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    H, dk, dv = cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim

    # ---- (a) total parameters, INCLUDING embeddings ----
    # Embeddings cost no FLOPs but AdamW still reads and writes all of them
    # every step. At V*d = 25M that is as much optimizer traffic as the whole
    # transformer stack.
    if mixer == "attn":
        p_block = 2 * (d * H * dk) + 2 * (d * H * dv)
    else:
        p_block = (
            2 * d * H * dk
            + 3 * d * H * dv
            + 2 * d * H
            + cfg.conv_size * (2 * H * dk + H * dv)
        )
    p_block += 3 * d * cfg.ffn_intermediate
    n_params = L * p_block + V * d * (1 if cfg.tie_embeddings else 2)

    # ---- (b) weights + optimizer: bf16 params, fp32 AdamW master copy ----
    #   fwd weight read      2      optimizer grad read   2
    #   bwd weight read      2      read m, v (fp32)      8
    #   grad write           2      write m, v            8
    #                               read master (fp32)    4
    #                               write master          4
    #                               write bf16 copy       2
    #   -> 34 bytes per parameter per step
    weights_and_opt = 34 * n_params / tokens_per_step

    # ---- (c) activations: written in fwd, read in bwd ----
    # ~act_elems_per_layer * d elements per layer per token, 2 bytes each,
    # touched twice. This constant is a GUESS and torch.compile fuses away an
    # unknown fraction of it.
    activations = 2 * 2 * act_elems_per_layer * d * L

    # ---- (d) mixer-specific traffic ----
    if mixer == "attn":
        # Flash-style: K and V are re-read from HBM once per query block of
        # size `attn_block`, so per token per layer: 2 * T * d / attn_block
        # elements. LINEAR IN T.
        mixer_traffic = 2 * 2 * (T * H * dk / attn_block) * L
    else:
        # Chunk state is materialized to HBM once per chunk of `chunk_size`
        # tokens, read and written: 2 * H*dk*dv / chunk_size elements per
        # token per layer. INDEPENDENT OF T.
        mixer_traffic = 2 * 2 * (H * dk * dv / chunk_size) * L

    parts = {
        "weights_and_opt": weights_and_opt,
        "activations": activations,
        "mixer": mixer_traffic,
    }
    return parts if return_breakdown else sum(parts.values())


def state_size_per_layer(cfg, mixer, T):
    """Elements of per-layer sequence memory.

    GDN:  the recurrent state S, one matrix per head.  Independent of T.
    Attn: the KV cache at length T.  Linear in T.
    """
    if mixer == "gdn":
        # S_t has shape (d_v, d_k) per head:
        #   S_t = S_{t-1}(alpha_t (I - beta_t k_t k_t^T)) + beta_t v_t k_t^T
        # Nothing in that recurrence depends on t, so the state never grows.
        return cfg.num_heads * cfg.head_k_dim * cfg.head_v_dim

    if mixer == "attn":
        # K and V are both kept for every past position: 2 tensors x T x H x d_h
        return 2 * T * cfg.num_heads * cfg.head_k_dim

    raise ValueError(mixer)


# ---------------------------------- SDPA shim for fla's Attention ----------------------------------
def _sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    # fla calls flash_attn_func(q, k, v, causal=True, window_size=...) with q,k,v: [B, T, H, D]
    assert tuple(window_size) == (-1, -1), "shim supports full causal attention only"
    assert not kwargs, f"unsupported kwargs: {list(kwargs)}"
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> [B, H, T, D]
    o = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=(q.shape[1] != k.shape[1])
    )
    return o.transpose(1, 2)  # -> [B, T, H, D]


def install_sdpa_shim():
    import fla.layers.attn as fla_attn

    fla_attn.flash_attn_func = _sdpa_flash_attn_func  # looked up at call time, so patching the module global works


def _sdpa_backend(name):
    from torch.nn.attention import SDPBackend

    return getattr(SDPBackend, name)


def sdpa_check():
    from torch.nn.attention import sdpa_kernel

    q = torch.randn(
        4, 8, 1024, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True
    )
    for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]:
        try:
            with sdpa_kernel([_sdpa_backend(name)]):
                F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
            torch.cuda.synchronize()
            print(f"{name:22s} ok")
        except Exception as e:
            print(f"{name:22s} FAIL {type(e).__name__}: {str(e)[:100]}")


# ---------------------------------- helpers ----------------------------------
def sm_clock_mhz():
    """Current SM clock via NVML, or None if unavailable (needs pynvml; GB10 may report N/A)."""
    try:
        return torch.cuda.clock_rate(DEV)
    except Exception:
        return None


def _time_for(fn, seconds, min_iters=3):
    """Run fn repeatedly for at least `seconds` of wall clock; return mean seconds per call."""
    fn()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while n < min_iters or time.perf_counter() - t0 < seconds:
        fn()
        n += 1
        if n % 10 == 0:
            torch.cuda.synchronize()  # keep the wall-clock check honest
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def roofline(seconds):
    n = 8192
    a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    burst = 2 * n**3 / _time_for(lambda: a @ b, seconds=0.5) / 1e12
    sustained = 2 * n**3 / _time_for(lambda: a @ b, seconds=seconds) / 1e12
    clk_under_load = sm_clock_mhz()
    del a, b
    x = torch.empty(
        2 * 2**30, device=DEV, dtype=torch.uint8
    )  # 2 GiB, far larger than any cache
    y = torch.empty_like(x)
    gbs = (
        2 * x.numel() / _time_for(lambda: y.copy_(x), seconds=seconds) / 1e9
    )  # read + write
    out = dict(
        device=torch.cuda.get_device_name(),
        bf16_matmul_tflops_burst=burst,
        bf16_matmul_tflops_measured=sustained,  # sustained over `seconds`; this is the ceiling used below
        bf16_tflops_peak=PEAK_BF16_TFLOPS,
        matmul_frac_of_peak=sustained / PEAK_BF16_TFLOPS,
        sm_clock_MHz_under_load=clk_under_load,
        copy_GBps_measured=gbs,
        bw_GBps_peak=PEAK_BW_GBPS,
        copy_frac_of_peak=gbs / PEAK_BW_GBPS,
        ridge_flop_per_byte_measured=sustained * 1e12 / (gbs * 1e9),
        ridge_flop_per_byte_peak=PEAK_BF16_TFLOPS * 1e12 / (PEAK_BW_GBPS * 1e9),
        duration_s=seconds,
    )
    json.dump(out, open(ROOFLINE, "w"), indent=2)
    print(json.dumps(out, indent=2))


def load_roofline():
    if not os.path.exists(ROOFLINE):
        return None
    r = json.load(open(ROOFLINE))
    # accept both the original key names and the renamed ones
    tf = r.get("bf16_matmul_tflops_measured", r.get("bf16_matmul_tflops"))
    bw = r.get("copy_GBps_measured", r.get("copy_GBps"))
    return dict(tflops=tf, GBps=bw)


def param_buckets(model):
    total = emb = head = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if "embeddings" in name:
            emb += p.numel()
        elif "lm_head" in name:
            head += p.numel()
    return dict(
        params_total=total,
        params_embed=emb,
        params_lm_head=head,
        params_body=total - emb - head,
    )


def _meta_model(mixer, size):
    """Build on the meta device: shapes only, no allocation, no GPU needed."""
    from fla.models import GatedDeltaNetForCausalLM, TransformerForCausalLM

    install_sdpa_shim()
    cfg = build_config(mixer, size)
    cls = TransformerForCausalLM if mixer == "attn" else GatedDeltaNetForCausalLM
    with torch.device("meta"):
        model = cls(cfg)
    return model, cfg


def params_table():
    """Parameter buckets for all 8 configs. No timing, no GPU work."""
    rows = []
    for size in SHAPES:
        for mixer in ("attn", "gdn"):
            model, cfg = _meta_model(mixer, size)
            pb = param_buckets(model)
            rows.append(
                dict(
                    mixer=mixer,
                    size=size,
                    d=cfg.hidden_size,
                    L=cfg.num_hidden_layers,
                    mlp_m=mlp_intermediate_size(cfg),
                    state_per_layer=state_size_per_layer(cfg, mixer, 1024),
                    **pb,
                )
            )
    hdr = [
        "mixer",
        "size",
        "d",
        "L",
        "mlp_m",
        "params_body",
        "params_embed",
        "params_lm_head",
        "params_total",
        "state_per_layer",
    ]
    print(" ".join(f"{h:>15s}" for h in hdr))
    for r in rows:
        print(" ".join(f"{str(r[h]):>15s}" for h in hdr))
    print(
        "\nPer-layer body params, as a multiple of d^2 (note: state_per_layer uses T=1024):"
    )
    for r in rows:
        per_layer = r["params_body"] / r["L"]
        print(
            f"  {r['mixer']:4s} {r['size']:5s}  {per_layer:12.0f}"
            f"   = {per_layer / r['d'] ** 2:5.2f} * d^2"
        )


def named_params(mixer, size):
    """Every parameter tensor with shape and count, for one config. Ground truth for Task 3a."""
    model, _ = _meta_model(mixer, size)
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        print(f"{name:60s} {str(tuple(p.shape)):20s} {p.numel():>12,d}")
    print(f"{'TOTAL':60s} {'':20s} {total:>12,d}")


# ---------------------------------- one benchmark row ----------------------------------
def run(mixer, size, B, T, warmup_s, time_s, min_steps, sdpa_backend):
    import contextlib

    install_sdpa_shim()
    torch.manual_seed(0)
    model, cfg = build_model(mixer, size)
    model.train()
    pb = param_buckets(model)
    print(
        f"[{mixer}/{size}] d={cfg.hidden_size} L={cfg.num_hidden_layers} "
        f"mlp_m={mlp_intermediate_size(cfg)} | body={pb['params_body']:,} "
        f"embed={pb['params_embed']:,} lm_head={pb['params_lm_head']:,} "
        f"total={pb['params_total']:,}",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)

    if sdpa_backend and mixer == "attn":
        from torch.nn.attention import sdpa_kernel

        attn_ctx = lambda: sdpa_kernel([_sdpa_backend(sdpa_backend)])
    else:
        attn_ctx = contextlib.nullcontext

    def step():
        with attn_ctx(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    # warmup: Triton autotuning, allocator growth, and thermal settling. By time, not steps.
    n_warm, t0 = 0, time.perf_counter()
    while n_warm < min_steps or time.perf_counter() - t0 < warmup_s:
        loss = step()
        n_warm += 1
        if n_warm % 5 == 0:
            torch.cuda.synchronize()
    assert torch.isfinite(loss), "non-finite loss during warmup"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # timed window: one CUDA event per step boundary; no host sync inside the loop.
    events = [torch.cuda.Event(enable_timing=True)]
    events[0].record()
    t0 = time.perf_counter()
    while len(events) - 1 < min_steps or time.perf_counter() - t0 < time_s:
        step()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        events.append(e)
        if len(events) % 5 == 0:
            torch.cuda.synchronize()  # bound CPU run-ahead so the wall-clock check means something
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    clk = sm_clock_mhz()

    step_ms = torch.tensor(
        [events[i].elapsed_time(events[i + 1]) for i in range(len(events) - 1)]
    )
    n = len(step_ms)
    med = step_ms.median().item()
    third = max(n // 3, 1)
    drift = step_ms[-third:].median().item() / step_ms[:third].median().item()

    tokens = B * T
    tps = tokens / (med / 1e3)
    roof = load_roofline()
    fpt_parts = flops_per_token(cfg, mixer, T)
    bpt_parts = bytes_per_token(cfg, mixer, T, tokens)
    fpt = sum(fpt_parts.values())
    bpt = sum(bpt_parts.values())
    row = dict(
        mixer=mixer,
        size=size,
        d=cfg.hidden_size,
        L=cfg.num_hidden_layers,
        B=B,
        T=T,
        sdpa_backend=(sdpa_backend or "default") if mixer == "attn" else None,
        n_warmup_steps=n_warm,
        n_timed_steps=n,
        timed_wall_s=wall,
        step_ms_median=med,
        step_ms_p10=step_ms.quantile(0.1).item(),
        step_ms_p90=step_ms.quantile(0.9).item(),
        drift_last_over_first_third=drift,  # >1.03 suggests throttling or a warmup that was too short
        tok_per_s=tps,
        tok_per_s_wall=tokens
        * n
        / wall,  # cross-check; should be within a few % of tok_per_s
        us_per_token=1e6 / tps,
        peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
        sm_clock_MHz_end=clk,
        **pb,
        state_per_layer=state_size_per_layer(cfg, mixer, T),
        flops_per_token=fpt,
        flops_breakdown=fpt_parts,
        bytes_per_token=bpt,
        bytes_breakdown=bpt_parts,
        torch=torch.__version__,
    )
    if fpt is not None:
        achieved = fpt * tps
        row["achieved_TFLOPs"] = achieved / 1e12
        row["mfu_vs_peak"] = achieved / (PEAK_BF16_TFLOPS * 1e12)
        if roof:
            row["mfu_vs_measured"] = achieved / (roof["tflops"] * 1e12)
            if achieved > roof["tflops"] * 1e12:
                print(
                    "!! implied FLOP rate exceeds measured matmul ceiling: flops_per_token overcounts"
                )
    if bpt is not None:
        bw = bpt * tps
        row["implied_GBps"] = bw / 1e9
        row["bw_util_vs_peak"] = bw / (PEAK_BW_GBPS * 1e9)
        if roof:
            row["bw_util_vs_measured"] = bw / (roof["GBps"] * 1e9)
            if bw > roof["GBps"] * 1e9:
                print(
                    "!! implied bandwidth exceeds measured copy bandwidth: bytes_per_token overcounts"
                )
    if drift > 1.03:
        print(
            f"!! drift {drift:.3f}: step time rose during the timed window (throttling?)"
        )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


def preheat(seconds):
    """Bring the device to thermal steady state so the first row isn't measured on a cold chip."""
    a = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    _time_for(lambda: a @ a, seconds=seconds)
    print(f"preheated {seconds}s; SM clock now {sm_clock_mhz()} MHz", flush=True)


def sweep(a):
    preheat(a.preheat_s)
    for size in SHAPES:
        for mixer in (
            "attn",
            "gdn",
        ):  # mixers adjacent in time at each size, so they see the same thermal state
            cmd = [
                sys.executable,
                __file__,
                "run",
                "--mixer",
                mixer,
                "--size",
                size,
                "--B",
                str(a.B),
                "--T",
                str(a.T),
                "--warmup-s",
                str(a.warmup_s),
                "--time-s",
                str(a.time_s),
                "--min-steps",
                str(a.min_steps),
                "--sdpa-backend",
                a.sdpa_backend,
            ]
            print(">>", " ".join(cmd), flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(
                    f"!! row {mixer}/{size} failed (rc={r.returncode}); continuing",
                    flush=True,
                )


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [json.loads(l) for l in open(RESULTS)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mixer, c in (("attn", "C0"), ("gdn", "C1")):
        rs = sorted(
            [r for r in rows if r["mixer"] == mixer], key=lambda r: r["params_body"]
        )
        if len(rs) < 2:
            continue
        P = np.array([r["params_body"] for r in rs], float)
        us = np.array([r["us_per_token"] for r in rs], float)
        # p10/p90 step-time band, converted to µs/token (per-row token count, not row 0's)
        tok = np.array([r["B"] * r["T"] for r in rs], float)
        lo = (
            np.array([r.get("step_ms_p10", r["step_ms_median"]) for r in rs])
            * 1e3
            / tok
        )
        hi = (
            np.array([r.get("step_ms_p90", r["step_ms_median"]) for r in rs])
            * 1e3
            / tok
        )
        slope = np.polyfit(np.log(P), np.log(us), 1)[0]
        ax[0].loglog(P / 1e6, us, "o-", color=c, label=f"{mixer}  slope={slope:.2f}")
        ax[0].fill_between(P / 1e6, lo, hi, color=c, alpha=0.2)
        ax[1].semilogx(
            P / 1e6, [r["peak_mem_GB"] for r in rs], "o-", color=c, label=mixer
        )
    ax[0].set_xlabel("non-embedding, non-head params (M)")
    ax[0].set_ylabel("µs per token (train step)")
    ax[0].set_title(
        "slope ~1 compute-bound, <1 bandwidth-bound, ~0 overhead-bound", fontsize=9
    )
    ax[1].set_xlabel("non-embedding, non-head params (M)")
    ax[1].set_ylabel("peak allocated (GB)")
    for x in ax:
        x.legend()
        x.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("day2_throughput.png", dpi=150)
    print("wrote day2_throughput.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "roofline",
            "sdpa-check",
            "params",
            "named-params",
            "run",
            "sweep",
            "plot",
        ],
    )
    ap.add_argument("--mixer", choices=["attn", "gdn"])
    ap.add_argument("--size", choices=list(SHAPES))
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument(
        "--warmup-s", type=float, default=30.0, help="minimum warmup wall-clock seconds"
    )
    ap.add_argument(
        "--time-s", type=float, default=60.0, help="minimum timed wall-clock seconds"
    )
    ap.add_argument(
        "--min-steps",
        type=int,
        default=10,
        help="minimum steps for both warmup and timing",
    )
    ap.add_argument(
        "--preheat-s",
        type=float,
        default=60.0,
        help="sweep only: device preheat before first row",
    )
    ap.add_argument(
        "--roofline-s",
        type=float,
        default=30.0,
        help="roofline only: sustained measurement window",
    )
    ap.add_argument(
        "--sdpa-backend",
        choices=[
            "FLASH_ATTENTION",
            "EFFICIENT_ATTENTION",
            "CUDNN_ATTENTION",
            "MATH",
            "default",
        ],
        default=DEFAULT_SDPA_BACKEND,
        help="SDPA backend for attention rows; 'default' hands the choice to PyTorch dispatch",
    )
    a = ap.parse_args()
    if a.sdpa_backend == "default":
        a.sdpa_backend = None
    {
        "roofline": lambda: roofline(a.roofline_s),
        "sdpa-check": sdpa_check,
        "params": params_table,
        "named-params": lambda: named_params(a.mixer, a.size),
        "plot": plot,
        "run": lambda: run(
            a.mixer, a.size, a.B, a.T, a.warmup_s, a.time_s, a.min_steps, a.sdpa_backend
        ),
        "sweep": lambda: sweep(a),
    }[a.cmd]()
