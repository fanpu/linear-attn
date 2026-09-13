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
    def forward(self, ids, start):
        """ids: [B, S] token ids occupying positions start..start+S-1.
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
            self.cache_k[i, :, :, start:end] = k
            self.cache_v[i, :, :, start:end] = v
            K = self.cache_k[i, :, :, :end]
            V = self.cache_v[i, :, :, :end]
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
        x = self.rms(x[:, -1].to(self.head_dtype), w["model.norm.weight"])
        return F.linear(x, w["lm_head.weight"])
