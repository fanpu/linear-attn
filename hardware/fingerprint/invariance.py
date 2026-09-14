"""Piece 1 (+3): batch-invariance bitmaps.

One fixed target row goes through an op inside batches of size B = 1..Bmax, at batch positions
first / middle / last; filler rows come from a fixed random pool. We store the raw bits of the target
row's output (first <= 4096 elements) for every (position, B), plus over the full row: number of
elements that differ from B = 1 and max |difference|. Checks: run-to-run repeats and swapping the
filler rows at fixed B. Piece 3: torch.profiler CUDA kernel-name signature for every (op, B).

  python invariance.py --bmax 512 --out cache/inv.npz            (all ops)
  python invariance.py --bmax 32 --ops lin_up_bf16 --out cache/toy.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import common as C

sys.path.insert(0, C.DECODE_MAP)
from qwen import Qwen3  # noqa: E402

dev = "cuda"
bf, f32 = torch.bfloat16, torch.float32
NPOOL = 513                      # index 0 = target row, 1..512 = fillers
LAYER = 13
MAXCOL = 4096


def gen(shape, dtype, seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dev, dtype)


def build_ops(m):
    """name -> (inputs dict of [NPOOL, ...] tensors, fn(batch dict) -> [B, D] output)."""
    w = m.w
    p = f"model.layers.{LAYER}."
    W = {k[len(p):]: v for k, v in w.items() if k.startswith(p)}
    ops = {}

    # Thinking Machines' torch.mm example: rows of linspace(-1000, 1000, 2048*4096), b = linspace D x D
    D = 4096
    a = torch.linspace(-1000, 1000, 2048 * D, device=dev).reshape(2048, D)[:NPOOL]
    b = torch.linspace(-1000, 1000, D * D, device=dev).reshape(D, D)
    ops["mm_tm_fp32"] = ({"x": a}, lambda z: torch.mm(z["x"], b))
    ops["mm_tm_bf16"] = ({"x": a.to(bf)}, lambda z, bb=b.to(bf): torch.mm(z["x"], bb))

    x1024 = gen((NPOOL, 1024), bf, 1)
    x3072 = gen((NPOOL, 3072), bf, 2)
    ops["lin_q_bf16"] = ({"x": x1024}, lambda z: F.linear(z["x"], W["self_attn.q_proj.weight"]))
    ops["lin_up_bf16"] = ({"x": x1024}, lambda z: F.linear(z["x"], W["mlp.up_proj.weight"]))
    ops["lin_down_bf16"] = ({"x": x3072}, lambda z: F.linear(z["x"], W["mlp.down_proj.weight"]))
    Wup32 = W["mlp.up_proj.weight"].float()
    ops["lin_up_fp32"] = ({"x": x1024.float()}, lambda z: F.linear(z["x"], Wup32))
    ops["lin_head_fp32"] = ({"x": x1024.float()}, lambda z: F.linear(z["x"], w["lm_head.weight"]))
    xs = gen((NPOOL, 32, 1024), bf, 3)
    ops["lin_up_prefill_bf16"] = ({"x": xs}, lambda z: F.linear(z["x"], W["mlp.up_proj.weight"])[:, -1])

    nw = W["input_layernorm.weight"]
    ops["rms_hf_bf16"] = ({"x": x1024 * 8}, lambda z: m.rms(z["x"], nw))
    ops["rms_fused_bf16"] = ({"x": x1024 * 8}, lambda z: F.rms_norm(z["x"], (1024,), nw, m.eps))
    ops["rms_fused_fp32"] = ({"x": (x1024 * 8).float()},
                             lambda z: F.rms_norm(z["x"], (1024,), nw.float(), m.eps))
    ops["softmax_vocab_fp32"] = ({"x": gen((NPOOL, 151936), f32, 4, 4.0)}, lambda z: torch.softmax(z["x"], -1))

    backends = {"math": SDPBackend.MATH, "eff": SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "default": None}
    qd, kd, vd = gen((NPOOL, 16, 1, 128), bf, 5), gen((NPOOL, 8, 256, 128), bf, 6), gen((NPOOL, 8, 256, 128), bf, 7)
    qp, kp, vp = gen((NPOOL, 16, 64, 128), bf, 8), gen((NPOOL, 8, 64, 128), bf, 9), gen((NPOOL, 8, 64, 128), bf, 10)

    def sdpa(be, causal):
        def f(z):
            call = lambda: F.scaled_dot_product_attention(z["q"], z["k"], z["v"], is_causal=causal, enable_gqa=True)
            if be is None:
                o = call()
            else:
                with sdpa_kernel([be]):
                    o = call()
            return o[:, :, -1].reshape(o.shape[0], -1)
        return f
    for n, be in backends.items():
        ops[f"sdpa_decode_{n}_bf16"] = ({"q": qd, "k": kd, "v": vd}, sdpa(be, False))
        ops[f"sdpa_prefill_{n}_bf16"] = ({"q": qp, "k": kp, "v": vp}, sdpa(be, True))

    # one full Qwen3 decoder layer (layer 13 weights), decode step with a 127-token KV cache, and prefill S=32
    nh, nkv, hd = m.nh, m.nkv, m.hd

    def layer(x, kc, vc, start):
        B, S, _ = x.shape
        cos, sin = m.cos[start:start + S][None, None], m.sin[start:start + S][None, None]
        h = m.rms(x, W["input_layernorm.weight"])
        q = F.linear(h, W["self_attn.q_proj.weight"]).view(B, S, nh, hd)
        k = F.linear(h, W["self_attn.k_proj.weight"]).view(B, S, nkv, hd)
        v = F.linear(h, W["self_attn.v_proj.weight"]).view(B, S, nkv, hd)
        q = m.rms(q, W["self_attn.q_norm.weight"]).transpose(1, 2)
        k = m.rms(k, W["self_attn.k_norm.weight"]).transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * cos + m.rot_half(q) * sin
        k = k * cos + m.rot_half(k) * sin
        if kc is not None:
            k, v = torch.cat([kc, k], 2), torch.cat([vc, v], 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=kc is None, enable_gqa=True)
        a = a.transpose(1, 2).reshape(B, S, nh * hd)
        x = x + F.linear(a, W["self_attn.o_proj.weight"])
        h = m.rms(x, W["post_attention_layernorm.weight"])
        x = x + F.linear(F.silu(F.linear(h, W["mlp.gate_proj.weight"])) * F.linear(h, W["mlp.up_proj.weight"]),
                         W["mlp.down_proj.weight"])
        return x[:, -1]
    xl = gen((NPOOL, 1, 1024), bf, 11)
    kc, vc = gen((NPOOL, 8, 127, 128), bf, 12), gen((NPOOL, 8, 127, 128), bf, 13)
    ops["layer_decode_bf16"] = ({"x": xl, "k": kc, "v": vc}, lambda z: layer(z["x"], z["k"], z["v"], 127))
    ops["layer_prefill_bf16"] = ({"x": gen((NPOOL, 32, 1024), bf, 14)}, lambda z: layer(z["x"], None, None, 0))

    # the full model (28 layers, bf16 body, fp32 head): prefill of a 24-token prompt, fp32 logits of last token
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    ids = tok.apply_chat_template([{"role": "user", "content": "Tell me about Richard Feynman"}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor(tok(ids).input_ids, device=dev)
    S = ids.numel()
    g = torch.Generator(device="cpu").manual_seed(15)
    pool_ids = torch.randint(0, 150000, (NPOOL, S), generator=g).to(dev)
    pool_ids[0] = ids

    def model_prefill(z):
        B = z["ids"].shape[0]
        if m.cache_k is None or m.cache_k.shape[1] < NPOOL:
            m.alloc(NPOOL, S)
        return m.forward(z["ids"], 0)
    ops["model_prefill_fp32logits"] = ({"ids": pool_ids}, model_prefill)
    return ops


def batch_idx(B, pos, alt=False):
    fill = torch.arange(1, NPOOL) if not alt else torch.arange(NPOOL - 1, 0, -1)
    fill = fill[:B - 1].tolist()
    return torch.tensor(fill[:pos] + [0] + fill[pos:], device=dev)


def run_op(name, inputs, fn, Bs, do_prof):
    def call(B, pos, alt=False):
        ix = batch_idx(B, pos, alt)
        z = {k: v.index_select(0, ix) for k, v in inputs.items()}
        y = fn(z)
        return y[pos].reshape(-1)

    ref = call(1, 0).clone()
    D = ref.numel()
    ncol = min(D, MAXCOL)
    dtype = ref.dtype
    P = 3
    bits = np.zeros((P, len(Bs), ncol), dtype=C.int_view(ref[:1]).dtype)
    ndiff = np.zeros((P, len(Bs)), np.int64)
    maxabs = np.zeros((P, len(Bs)), np.float64)
    for j, B in enumerate(Bs):
        for pi, pos in enumerate([0, B // 2, B - 1]):
            y = call(B, pos)
            bits[pi, j] = C.int_view(y[:ncol])
            ne = (y.view(torch.int16 if y.element_size() == 2 else torch.int32)
                  != ref.view(torch.int16 if y.element_size() == 2 else torch.int32))
            ndiff[pi, j] = int(ne.sum())
            maxabs[pi, j] = float((y.double() - ref.double()).abs().max())
    # run-to-run: 3 repeats; filler swap: reversed pool, at a few B
    checks = {}
    for B in [b for b in (1, 7, 64, 333, 512) if b <= Bs[-1]]:
        y0 = call(B, 0)
        rep = all(torch.equal(y0, call(B, 0)) for _ in range(3))
        swap = torch.equal(y0, call(B, 0, alt=True)) if B > 1 else True
        checks[B] = (rep, swap)
    kern = []
    if do_prof:
        for B in Bs:
            ix = batch_idx(B, 0)
            z = {k: v.index_select(0, ix) for k, v in inputs.items()}
            kern.append(C.cuda_kernels(lambda: fn(z)))
    return dict(bits=bits, ndiff=ndiff, maxabs=maxabs, D=D, dtype=str(dtype), checks=checks, kernels=kern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmax", type=int, default=512)
    ap.add_argument("--ops", nargs="*", default=None)
    ap.add_argument("--noprof", action="store_true")
    ap.add_argument("--out", default="cache/inv.npz")
    a = ap.parse_args()
    os.chdir(C.ROOT)
    tag = os.path.splitext(os.path.basename(a.out))[0]
    before = C.smi(f"before_{tag}")
    torch.backends.cuda.matmul.allow_tf32 = False
    m = Qwen3()
    with torch.inference_mode():
        ops = build_ops(m)
        names = a.ops or list(ops)
        Bs = list(range(1, a.bmax + 1))
        res = {}
        t0 = time.time()
        for n in names:
            t1 = time.time()
            r = run_op(n, *ops[n], Bs, not a.noprof)
            res[n] = r
            nb = int((r["ndiff"] > 0).sum(1)[0])
            print(f"{n:28s} D={r['D']:6d} B-with-diff(pos0)={nb:4d}/{len(Bs)} max|d|={r['maxabs'].max():.3g} "
                  f"checks={r['checks']} kernsets={len(set(map(tuple, r['kernels'])))} {time.time()-t1:.0f}s",
                  flush=True)
        # Thinking Machines' snippet verbatim
        B, D = 2048, 4096
        aa = torch.linspace(-1000, 1000, B * D, device=dev).reshape(B, D)
        bb = torch.linspace(-1000, 1000, D * D, device=dev).reshape(D, D)
        tm = float((torch.mm(aa[:1], bb) - torch.mm(aa, bb)[:1]).abs().max())
        print("TM snippet max|out1-out2| =", tm)
    after = C.smi(f"after_{tag}")
    out = {"Bs": np.array(Bs), "names": np.array(names), "tm_snippet": tm}
    meta = {"stack": C.stack(), "smi_before": before, "smi_after": after, "wall_s": time.time() - t0, "ops": {}}
    for n, r in res.items():
        out[f"{n}__bits"], out[f"{n}__ndiff"], out[f"{n}__maxabs"] = r["bits"], r["ndiff"], r["maxabs"]
        meta["ops"][n] = dict(D=r["D"], dtype=r["dtype"], checks={str(k): v for k, v in r["checks"].items()},
                              kernels=r["kernels"])
    np.savez_compressed(a.out, **out)
    C.save_json(a.out.replace(".npz", ".json"), meta)
    print("saved", a.out, f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
