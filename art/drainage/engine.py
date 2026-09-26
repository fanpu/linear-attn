"""Minimal Qwen3 forward pass with a static KV cache, for greedy decoding from single tokens.

Adapted from ../decode-map/qwen.py (which was checked against transformers there; re-checked
here by verify.py). Every row of a batch starts from exactly one token at position 0, so all
rows always share the same length: no padding and no attention mask are needed at all.

Body dtype is configurable (bf16 or fp32); final norm + unembedding are fp32, so greedy argmax
is taken over fp32 logits.
"""
import glob
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def hf_dir(name):
    return glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--Qwen--{name}/snapshots/*/"))[0]


class Qwen3:
    def __init__(self, name="Qwen3-0.6B", device="cuda", dtype=torch.bfloat16):
        d = hf_dir(name)
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cfg = cfg
        self.nl = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.dtype = dtype
        self.device = device
        sd = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            sd.update(load_file(f))
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        self.w["lm_head.weight"] = head.to(device=device, dtype=torch.float32)
        self.w["model.norm.weight"] = sd["model.norm.weight"].to(device=device, dtype=torch.float32)
        del sd
        theta = cfg["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        pos = torch.arange(4096, dtype=torch.float64)
        fr = torch.outer(pos, inv)
        emb = torch.cat([fr, fr], -1)
        self.cos = emb.cos().to(device, dtype)
        self.sin = emb.sin().to(device, dtype)
        self.cache_k = self.cache_v = None

    def rms(self, x, w):
        dt = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(var + self.eps)).to(dt)

    @staticmethod
    def rot_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], -1)

    def alloc(self, batch, max_len):
        self.cache_k = self.cache_v = None
        torch.cuda.empty_cache()
        shape = (self.nl, batch, self.nkv, max_len, self.hd)
        self.cache_k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.cache_v = torch.zeros(shape, device=self.device, dtype=self.dtype)

    def compact(self, keep, length):
        """Move cache rows `keep` (LongTensor) to rows 0..len(keep)-1, positions [:length]."""
        n = keep.numel()
        for c in (self.cache_k, self.cache_v):
            for i in range(self.nl):           # per layer to bound the temporary copy
                c[i, :n, :, :length] = c[i, keep, :, :length]

    @torch.no_grad()
    def step(self, ids, pos):
        """ids: [B] tokens at position `pos` for cache rows 0..B-1. Returns fp32 logits [B, V]."""
        B = ids.shape[0]
        w = self.w
        x = w["model.embed_tokens.weight"][ids][:, None]          # [B,1,D]
        cos = self.cos[pos:pos + 1][None, None]
        sin = self.sin[pos:pos + 1][None, None]
        end = pos + 1
        for i in range(self.nl):
            p = f"model.layers.{i}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(B, 1, self.nh, self.hd)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(B, 1, self.nkv, self.hd)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(B, 1, self.nkv, self.hd)
            q = self.rms(q, w[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = self.rms(k, w[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + self.rot_half(q) * sin
            k = k * cos + self.rot_half(k) * sin
            self.cache_k[i, :B, :, pos:end] = k
            self.cache_v[i, :B, :, pos:end] = v
            K = self.cache_k[i, :B, :, :end]
            V = self.cache_v[i, :B, :, :end]
            a = F.scaled_dot_product_attention(q, K, V, is_causal=False, enable_gqa=True)
            a = a.transpose(1, 2).reshape(B, 1, self.nh * self.hd)
            x = x + F.linear(a, w[p + "self_attn.o_proj.weight"])
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g = F.linear(h, w[p + "mlp.gate_proj.weight"])
            u = F.linear(h, w[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(g) * u, w[p + "mlp.down_proj.weight"])
        x = self.rms(x[:, -1].float(), w["model.norm.weight"])
        return F.linear(x, w["lm_head.weight"])


EOS_IDS = (151643, 151645)          # <|endoftext|>, <|im_end|>
N_REGULAR = 151643                  # ids 0..151642 are ordinary BPE tokens


def trailing_period(seq, L, pmax, min_span):
    """seq: [B, >=L] int tensor on GPU. For each row, the smallest p <= pmax such that the
    last `span` tokens of seq[:, :L] are p-periodic with span >= max(3p, min_span).
    Returns LongTensor [B] (0 = none)."""
    B = seq.shape[0]
    found = torch.zeros(B, dtype=torch.long, device=seq.device)
    for p in range(1, pmax + 1):
        need = max(2 * p, min_span - p)       # number of equalities s[j]==s[j-p] at the tail
        if need + p > L:
            break
        ok = (seq[:, L - need:L] == seq[:, L - need - p:L - p]).all(1)
        found = torch.where((found == 0) & ok, torch.full_like(found, p), found)
    return found
