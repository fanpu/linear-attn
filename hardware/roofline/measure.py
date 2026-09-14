"""measure.py - Roofline constellation: re-measure the roofs on an idle GB10 and time a set of "stars".

    python measure.py --out cache/roofline.npz      (~10 min, needs an idle GPU; logs nvidia-smi before/after)

Roofs: bf16 GEMM at n = 1024..16384 (peak = the max), device copy / read bandwidth on 1-4 GiB buffers.
Stars: kernels with analytic FLOPs and minimal bytes (weights + activations + KV, all in the tensors' dtype):
  GEMM n x n bf16; skinny [B,1024]x[1024,3072] (a decode linear) for B = 1..512; SDPA flash at T = 256..8192;
  Qwen3-0.6B prefill (B=1, T = 128..4096) and decode step (B = 1..512, KV 512) with decode-map's model;
  elementwise add, softmax over 151936, a 2 GiB copy. Each timed by CUDA events, median of reps, warm.
Everything is measured; the analytic byte counts are declared (the minimum traffic if every tensor is read once).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/fzeng/ml/research/art/decode-map")
sys.path.insert(0, "/home/fzeng/ml/research/hardware/fingerprint")
import common as C   # smi(), stack()
C.LOGS = f"{HERE}/logs"; os.makedirs(C.LOGS, exist_ok=True)
dev = "cuda"; bf = torch.bfloat16


def timed(fn, warm=3, reps=10, min_ms=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); n = 0; t = 0.0
        while True:
            fn(); n += 1
            e.record(); torch.cuda.synchronize(); t = s.elapsed_time(e)
            if t >= min_ms or n >= 50:
                break
        ts.append(t / n / 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{HERE}/cache/roofline.npz"); a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    before = C.smi("before_roofline")
    torch.backends.cuda.matmul.allow_tf32 = False
    stars = []

    def star(group, label, flops, bytes_, sec_med, sec_min, **kw):
        stars.append(dict(group=group, label=label, flops=flops, bytes=bytes_, sec=sec_med, sec_min=sec_min, **kw))
        print(f"{group:10s} {label:28s} OI={flops/bytes_:9.2f} F/B  {flops/sec_med/1e12:7.2f} TFLOP/s  {bytes_/sec_med/1e9:7.1f} GB/s  {sec_med*1e3:8.3f} ms", flush=True)

    # roofs
    gemm_peak = 0
    for n in (1024, 2048, 4096, 8192, 12288, 16384):
        A = torch.randn(n, n, device=dev, dtype=bf); B = torch.randn(n, n, device=dev, dtype=bf)
        med, mn = timed(lambda: A @ B, reps=7)
        star("gemm", f"GEMM {n}²", 2.0 * n ** 3, 3 * 2.0 * n * n, med, mn, n=n)
        gemm_peak = max(gemm_peak, 2.0 * n ** 3 / med); del A, B
    bw_peak = 0
    for gib in (1, 2, 4):
        src = torch.empty(gib << 30, dtype=torch.uint8, device=dev); dst = torch.empty_like(src)
        med, mn = timed(lambda: dst.copy_(src), reps=7)
        star("copy", f"copy {gib} GiB", 0.0, 2.0 * (gib << 30), med, mn)
        bw_peak = max(bw_peak, 2.0 * (gib << 30) / med)
        x = src.view(torch.float32)
        med, mn = timed(lambda: x.sum(), reps=7)
        star("read", f"sum {gib} GiB", float(x.numel()), float(gib << 30), med, mn)
        bw_peak = max(bw_peak, (gib << 30) / med); del src, dst, x
    torch.cuda.empty_cache()
    # skinny linear (decode-style): [B,1024] x [3072,1024]^T
    W = torch.randn(3072, 1024, device=dev, dtype=bf)
    for Bsz in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        x = torch.randn(Bsz, 1024, device=dev, dtype=bf)
        med, mn = timed(lambda: F.linear(x, W))
        star("linear", f"linear B={Bsz}", 2.0 * Bsz * 1024 * 3072, 2.0 * (W.numel() + x.numel() + Bsz * 3072), med, mn, B=Bsz)
    # SDPA flash, H=16, d=128, causal
    for T in (256, 512, 1024, 2048, 4096, 8192):
        q = torch.randn(1, 16, T, 128, device=dev, dtype=bf); k = torch.randn_like(q); v = torch.randn_like(q)
        med, mn = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        star("sdpa", f"SDPA causal T={T}", 4.0 * T * T * 128 * 16 / 2, 2.0 * 4 * q.numel(), med, mn, T=T)
        del q, k, v
    # elementwise
    x = torch.randn(1 << 28, device=dev, dtype=bf); y = torch.randn_like(x)
    med, mn = timed(lambda: x + y); star("eltwise", "add 2^28 bf16", float(x.numel()), 3 * 2.0 * x.numel(), med, mn)
    lg = torch.randn(512, 151936, device=dev)
    med, mn = timed(lambda: torch.softmax(lg, -1)); star("eltwise", "softmax 512×151936 fp32", 5.0 * lg.numel(), 2 * 4.0 * lg.numel(), med, mn)
    del x, y, lg; torch.cuda.empty_cache()
    # Qwen3-0.6B via decode-map's model
    from qwen import Qwen3
    m = Qwen3()
    nparam_body = sum(v.numel() for k, v in m.w.items() if "embed" not in k and "lm_head" not in k)
    nhead = m.w["lm_head.weight"].numel()
    nl = 28; nkv = m.nkv; hd = m.hd
    wbytes = 2.0 * nparam_body + 2.0 * nhead      # bf16 body, head stored bf16 (upcast per call in this model: counted once)
    for T in (128, 256, 512, 1024, 2048, 4096):
        m.alloc(1, T + 8); ids = torch.randint(0, 150000, (1, T), device=dev)
        med, mn = timed(lambda: m.forward(ids, 0), reps=5)
        flops = 2.0 * nparam_body * T + 2.0 * nhead * T + 4.0 * nl * T * T * m.nh * hd / 2
        by = wbytes + 2.0 * nl * 2 * T * nkv * hd + 4.0 * T * 151936      # weights once + KV written + fp32 logits; activations not counted (declared)
        star("prefill", f"Qwen3 prefill T={T}", flops, by, med, mn, T=T)
    for Bsz in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        m.alloc(Bsz, 520); ids = torch.randint(0, 150000, (Bsz, 512), device=dev); m.forward(ids, 0)
        one = torch.randint(0, 150000, (Bsz, 1), device=dev)
        med, mn = timed(lambda: m.forward(one, 512), reps=5)
        flops = 2.0 * (nparam_body + nhead) * Bsz + 4.0 * nl * 512 * Bsz * m.nh * hd
        by = wbytes + 2.0 * nl * 2 * Bsz * 512 * nkv * hd + 4.0 * Bsz * 151936
        star("decode", f"Qwen3 decode B={Bsz}", flops, by, med, mn, B=Bsz)
        torch.cuda.empty_cache()
    after = C.smi("after_roofline")
    meta = dict(stack=C.stack(), smi_before=before, smi_after=after, gemm_peak_tflops=gemm_peak / 1e12, bw_peak_gbps=bw_peak / 1e9,
                ridge=gemm_peak / bw_peak, qwen=dict(params_body=nparam_body, params_head=nhead))
    json.dump(dict(meta=meta, stars=stars), open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print("peak GEMM %.1f TFLOP/s, peak BW %.1f GB/s, ridge %.0f F/B" % (gemm_peak / 1e12, bw_peak / 1e9, gemm_peak / bw_peak))


if __name__ == "__main__":
    main()
