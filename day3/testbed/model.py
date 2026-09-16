"""Model shapes and construction.  [AI-owned]
# after fla-org/flash-linear-attention fla/models/transformer/ (the model class
# and its config) and karpathy/nanoGPT model.py (named size presets).

Copied from the throughput benchmark with one change: the vocabulary is a
parameter (today 50,304; the benchmark used 32,000). The SDPA shim is what
makes fla's attention layer run on the GB10, where FlashAttention-3 has no
sm_121 build: fla's Attention calls `flash_attn_func`, and we replace that
module global with a wrapper around torch's scaled_dot_product_attention.
"""

import torch
import torch.nn.functional as F

# size -> (hidden_size d, num_layers L). Head dim is 64 at every size, so the
# number of heads is d / 64. Measured non-embedding parameter counts (throughput benchmark):
# 30M -> 34.1M, 60M -> 63.7M, 125M -> 127.4M.
SHAPES = {
    "30M": (512, 10),
    "60M": (768, 9),
    "125M": (768, 18),
}


def build_config(size: str, vocab: int = 50304):
    """fla TransformerConfig for the attention baseline at `size`.

    tie_word_embeddings=False (input embedding and LM head are separate
    matrices) and fuse_cross_entropy=True (fla's fused cross-entropy kernel)
    are the throughput benchmark's settings and are kept so that tok/s stays comparable.
    """
    from fla.models import TransformerConfig

    d, L = SHAPES[size]
    return TransformerConfig(
        hidden_size=d,
        num_hidden_layers=L,
        num_heads=d // 64,
        vocab_size=vocab,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        fuse_cross_entropy=True,
    )


def build_model(size: str, vocab: int = 50304, device: str = "cuda"):
    """Construct the model on `device`. The caller seeds torch before calling
    this, since fla's __init__ draws the initial weights."""
    from fla.models import TransformerForCausalLM

    cfg = build_config(size, vocab)
    model = TransformerForCausalLM(cfg).to(device)
    return model, cfg


def count_params(model) -> dict:
    """Parameter counts in millions: body (non-embedding), embedding, head, total."""
    emb = head = total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if "embed" in name:
            emb += n
        elif "lm_head" in name:
            head += n
    return {
        "body_M": (total - emb - head) / 1e6,
        "embed_M": emb / 1e6,
        "head_M": head / 1e6,
        "total_M": total / 1e6,
    }


# ---------------------------- SDPA shim for fla's Attention ----------------------------
def _sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    # fla calls flash_attn_func(q, k, v, causal=True, window_size=...) with q, k, v: [B, T, H, D]
    assert tuple(window_size) == (-1, -1), "shim supports full causal attention only"
    assert not kwargs, f"unsupported kwargs: {list(kwargs)}"
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # -> [B, H, T, D]
    o = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=(q.shape[1] != k.shape[1])
    )
    return o.transpose(1, 2)  # -> [B, T, H, D]


def install_sdpa_shim():
    """Route fla's attention through torch SDPA. Call once, before building a model."""
    import fla.layers.attn as fla_attn

    fla_attn.flash_attn_func = _sdpa_flash_attn_func  # looked up at call time, so patching the module global works


class _SDPAFlashCtx:
    """Reusable context pinning SDPA to Flash. torch's sdpa_kernel is a
    @contextlib.contextmanager and can be entered only once, so we build a
    fresh one on every __enter__."""

    def __enter__(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        self._cm = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def sdpa_ctx():
    """Context manager pinning SDPA to the Flash backend, the fastest on the GB10
    in the throughput benchmark (10-13% faster than cuDNN at every head count). Every forward pass in
    training and evaluation runs inside this context so both use the same kernel.
    The returned object may be entered any number of times."""
    return _SDPAFlashCtx()
