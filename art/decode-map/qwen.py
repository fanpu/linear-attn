"""Minimal Qwen3 (bf16 body, fp32 final norm + unembedding -> fp32 logits) forward pass with a preallocated (static) KV cache.

Written by hand (instead of HF generate()) so that:
  * every row of a batch has exactly the same length -> no padding at all;
  * the shared prompt is prefilled once and its KV is copied to every row;
  * the sampling loop (sample.py) controls all randomness.
Weights are loaded from the local HF cache (safetensors) and checked against
transformers' Qwen3ForCausalLM in test_model.py.
"""
import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HF_DIR = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/"))[0]


class Qwen3:
    def __init__(self, device="cuda", dtype=torch.bfloat16, head_dtype=torch.float32):
        cfg = json.load(open(os.path.join(HF_DIR, "config.json")))
        self.cfg = cfg
        self.nl = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.dtype = dtype
        self.device = device
        sd = load_file(os.path.join(HF_DIR, "model.safetensors"))
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        # final norm + unembedding in head_dtype (fp32) so that logits are fp32
        self.w["lm_head.weight"] = head.to(device=device, dtype=head_dtype)
        self.w["model.norm.weight"] = sd["model.norm.weight"].to(device=device, dtype=head_dtype)
        self.head_dtype = head_dtype
        theta = cfg["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        pos = torch.arange(4096, dtype=torch.float64)
        fr = torch.outer(pos, inv)
        emb = torch.cat([fr, fr], -1)
        self.cos = emb.cos().to(device, dtype)
        self.sin = emb.sin().to(device, dtype)
        self.cache_k = self.cache_v = None

    def rms(self, x, w):
        dt = x.dtype                                   # as in HF: normalise in fp32
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(var + self.eps)).to(dt)

    @staticmethod
    def rot_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def alloc(self, batch, max_len):
        shape = (self.nl, batch, self.nkv, max_len, self.hd)
        self.cache_k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.cache_v = torch.zeros(shape, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def forward(self, ids, start, r0=0, n_out=None):
        """ids: [B, S] token ids occupying positions start..start+S-1, for cache rows r0..r0+B-1.
        Writes KV into the cache and returns logits of the last position [B, V]."""
        B, S = ids.shape
        w = self.w
        x = w["model.embed_tokens.weight"][ids]
        cos = self.cos[start:start + S][None, None]
        sin = self.sin[start:start + S][None, None]
        end = start + S
        for i in range(self.nl):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(B, S, self.nh, self.hd)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(B, S, self.nkv, self.hd)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(B, S, self.nkv, self.hd)
            q = self.rms(q, w[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = self.rms(k, w[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + self.rot_half(q) * sin
            k = k * cos + self.rot_half(k) * sin
            r1 = r0 + B
            self.cache_k[i, r0:r1, :, start:end] = k
            self.cache_v[i, r0:r1, :, start:end] = v
            K = self.cache_k[i, r0:r1, :, :end]
            V = self.cache_v[i, r0:r1, :, :end]
            if S > 1:
                a = F.scaled_dot_product_attention(q, K, V, is_causal=True, enable_gqa=True)
            else:
                a = F.scaled_dot_product_attention(q, K, V, is_causal=False, enable_gqa=True)
            a = a.transpose(1, 2).reshape(B, S, self.nh * self.hd)
            x = x + F.linear(a, w[p + "self_attn.o_proj.weight"])
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g = F.linear(h, w[p + "mlp.gate_proj.weight"])
            u = F.linear(h, w[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"])
        x = x[:, -1] if n_out is None else x[:n_out, -1]      # unembed only the rows needed
        x = self.rms(x.to(self.head_dtype), w["model.norm.weight"])
        return F.linear(x, w["lm_head.weight"])


class StepGraphs:
    """CUDA-graph-captured single-token decode step for fixed windows of Fb cache rows.

    One graph per window offset r0 in {0, Fb, 2Fb, ...}. Every graph attends over the full
    static cache length with a boolean mask (instead of slicing K[:end]), so shapes never
    change. The attention kernel therefore differs from the eager path (math/mem-efficient
    instead of flash on a sliced cache); bf16 logits differ at the ~1e-3 level, which is the
    same class of batch/kernel noise the placement test measures. Always unembeds all Fb rows."""

    def __init__(self, m, Fb, n_windows, max_len):
        self.m, self.Fb, self.max_len = m, Fb, max_len
        dev = m.device
        self.ids = torch.zeros(Fb, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.mask = torch.zeros(1, 1, 1, max_len, dtype=torch.bool, device=dev)
        self.ar = torch.arange(max_len, device=dev)
        self.graphs, self.outs = [], []
        self.pool = torch.cuda.graph_pool_handle()
        self.pos.fill_(max_len - 1); self.mask.fill_(True)
        for j in range(n_windows):
            r0 = j * Fb
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step(r0)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._step(r0)
            self.graphs.append(g)
            self.outs.append(out)

    @torch.no_grad()
    def _step(self, r0):
        m, w, Fb = self.m, self.m.w, self.Fb
        x = w["model.embed_tokens.weight"][self.ids]
        cos = m.cos.index_select(0, self.pos)[None, None]
        sin = m.sin.index_select(0, self.pos)[None, None]
        for i in range(m.nl):
            p = f"model.layers.{i}."
            h = m.rms(x, w[p + "input_layernorm.weight"])
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(Fb, 1, m.nh, m.hd)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(Fb, 1, m.nkv, m.hd)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(Fb, 1, m.nkv, m.hd)
            q = m.rms(q, w[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = m.rms(k, w[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + m.rot_half(q) * sin
            k = k * cos + m.rot_half(k) * sin
            K = m.cache_k[i, r0:r0 + Fb]
            V = m.cache_v[i, r0:r0 + Fb]
            K.index_copy_(2, self.pos, k)
            V.index_copy_(2, self.pos, v)
            a = F.scaled_dot_product_attention(q, K, V, attn_mask=self.mask, enable_gqa=True)
            a = a.transpose(1, 2).reshape(Fb, 1, m.nh * m.hd)
            x = x + F.linear(a, w[p + "self_attn.o_proj.weight"])
            h = m.rms(x, w[p + "post_attention_layernorm.weight"])
            g = F.linear(h, w[p + "mlp.gate_proj.weight"])
            u = F.linear(h, w[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"])
        x = m.rms(x[:, -1].to(m.head_dtype), w["model.norm.weight"])
        return F.linear(x, w["lm_head.weight"])

    def run(self, ids, start, r0):
        """ids [Fb,1] at position start, cache rows r0..r0+Fb-1. Returns a view of the static
        output buffer [Fb,V] (clone or copy before the next call)."""
        self.ids.copy_(ids)
        self.pos.fill_(start)
        torch.lt(self.ar, start + 1, out=self.mask[0, 0, 0])
        self.graphs[r0 // self.Fb].replay()
        return self.outs[r0 // self.Fb]
