"""Shared helpers: real text from fineweb-edu, model loading, pasar_job import.

Text: the fineweb-edu validation shard in day3/data (llm.c format, GPT-2 token ids). We decode
each document back to a string with tiktoken and re-tokenize it with each model's own tokenizer,
so every model sees the same characters.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SHARD = "/home/fzeng/ml/research/day3/data/fineweb_edu/edu_fineweb_val_000000.bin"

# pasar_job lives in art/.venv; the root .venv (which has fla) does not have it.
try:
    import pasar_job  # noqa: F401
except ImportError:
    for p in glob.glob("/home/fzeng/ml/research/art/.venv/lib/python*/site-packages"):
        sys.path.append(p)
    import pasar_job  # noqa: F401

pasar_job = sys.modules["pasar_job"]


def fineweb_docs(start, n, min_chars=1500):
    """Return n documents (strings) from the val shard, skipping short ones.
    Documents [0, 200) are the 'calibration/measurement' pool; [1000, ...) are held out."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    a = np.fromfile(SHARD, dtype=np.uint16)[512:]  # 256 int32 header
    eot = 50256
    idx = np.where(a == eot)[0]
    out = []
    i = start
    while len(out) < n and i + 1 < len(idx):
        s = enc.decode(a[idx[i] + 1: idx[i + 1]].tolist())
        if len(s) >= min_chars:
            out.append(s)
        i += 1
    return out


SHORT = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "allenai/OLMo-2-0425-1B": "olmo2-1b",
    "fla-hub/delta_net-1.3B-100B": "deltanet-1.3b",
    "fla-hub/gla-1.3B-100B": "gla-1.3b",
    "fla-hub/rwkv7-1.47B-pile": "rwkv7-1.5b",
    "linear-moe-hub/Gated-Deltanet-1.3B": "gdn-1.3b",
    "fla-hub/gla-340M-15B": "gla-340m",
    "linear-moe-hub/Gated-Deltanet-340M": "gdn-340m",
}


def load(model_id, dtype, device, random_init=False, seed=0):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if model_id.startswith(("fla-hub", "linear-moe-hub")):
        import fla  # noqa: F401  registers the fla model types with transformers
        import fla.models as fm
        # fla 0.5.2 declares _tied_weights_keys as a list; transformers 5 expects {target: source}
        for n in dir(fm):
            c = getattr(fm, n)
            if isinstance(c, type) and isinstance(getattr(c, "_tied_weights_keys", None), list):
                c._tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}
    tok = AutoTokenizer.from_pretrained(model_id)
    is_fla = model_id.startswith(("fla-hub", "linear-moe-hub"))
    if is_fla and not random_init and not model_id.startswith("linear-moe-hub"):
        # unfused SwiGLU so that hooks on mlp.down_proj fire (numerically the same computation)
        cfg = AutoConfig.from_pretrained(model_id)
        if hasattr(cfg, "fuse_swiglu"):
            cfg.fuse_swiglu = False
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, config=cfg)
        model.to(device).eval()
        return tok, model
    if random_init:
        cfg = AutoConfig.from_pretrained(model_id)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    elif model_id.startswith("linear-moe-hub"):
        # older fla checkpoint format: fused gate_proj = [gate; up] (GatedMLP did gate, y = chunk(2)),
        # plus an unused attn.D. Split and drop; checked by perplexity in measure.py.
        import glob as _g
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(model_id)
        if hasattr(cfg, "fuse_swiglu"):
            cfg.fuse_swiglu = False
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        sd = {}
        for f in _g.glob(os.path.join(snapshot_download(model_id), "*.safetensors")):
            sd.update(load_file(f))
        new = {}
        for k, v in sd.items():
            if k.endswith("attn.D"):
                continue
            if k.endswith("mlp.gate_proj.weight") and v.shape[0] == 2 * model.get_parameter(k).shape[0]:
                g, u = v.chunk(2, 0)
                new[k] = g
                new[k.replace("gate_proj", "up_proj")] = u
            else:
                new[k] = v
        missing, unexpected = model.load_state_dict(new, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]
        assert not missing and not unexpected, (missing, unexpected)
        if getattr(cfg, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embeddings.weight
        model.to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    return tok, model


def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.blocks"):
        m = model
        ok = True
        for p in path.split("."):
            if not hasattr(m, p):
                ok = False
                break
            m = getattr(m, p)
        if ok:
            return m
    raise RuntimeError("no decoder layers found")
