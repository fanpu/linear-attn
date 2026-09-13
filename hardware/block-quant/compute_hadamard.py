"""Hadamard before/after (QuaRot-style residual-stream rotation) for block quantization.

Q = H_1024 diag(s) / sqrt(1024), H Sylvester-Hadamard, s random signs (seed 0): an orthogonal matrix.
The residual stream is rotated x -> Q^T x, so
  - matrices that READ the residual (q/k/v/gate/up, rows of the embedding):  W' = W_fused Q
    where W_fused = W diag(g) absorbs the preceding RMSNorm gain g (QuaRot fuses norms first so that the
    rotation is function-preserving; the norm becomes a plain RMS normalisation, which commutes with Q)
  - matrices that WRITE the residual (o_proj, down_proj):                      W' = Q^T W
Not done here (declared simplification): the online Hadamard QuaRot also applies inside attention / on the
down_proj input, and any activation quantization. Qwen3 q_norm/k_norm act after q/k_proj so they do not
interfere with the input-side rotation.

Variants stored per matrix: raw (as shipped), fused (norm gain absorbed, if applicable), rotated.
-> cache/hadamard_<tag>.npz
"""
import os

import numpy as np
import torch

import formats as F
from weights import SafeTensors, layer_name

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {  # tag: (tensor, side, norm gain tensor or None, row slice)
    "L26_q_proj": (layer_name(26, "q_proj"), "in", "model.layers.26.input_layernorm.weight", None),
    "L16_k_proj": (layer_name(16, "k_proj"), "in", "model.layers.16.input_layernorm.weight", None),
    "L27_gate_proj": (layer_name(27, "gate_proj"), "in", "model.layers.27.post_attention_layernorm.weight", None),
    "L26_gate_proj": (layer_name(26, "gate_proj"), "in", "model.layers.26.post_attention_layernorm.weight", None),
    "L06_down_proj": (layer_name(6, "down_proj"), "out", None, None),
    "embed_rare": ("model.embed_tokens.weight", "in", None, (147456, 150528)),
    "embed_head": ("model.embed_tokens.weight", "in", None, (0, 3072)),
}


def hadamard(n, seed=0):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    s = torch.tensor(np.random.default_rng(seed).choice([-1.0, 1.0], n))
    return H * s[None, :] / np.sqrt(n)


def to_bf16_raw(W):
    return W.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)


def summarize(W):
    A = W.abs()
    c = W - W.mean()
    return dict(kurtosis=float(c.pow(4).mean() / c.pow(2).mean() ** 2),
                colmax_med=float(A.amax(0).max() / A.amax(0).median()),
                rowmax_med=float(A.amax(1).max() / A.amax(1).median()),
                colrms_cv=float(W.pow(2).mean(0).sqrt().std() / W.pow(2).mean(0).sqrt().mean()),
                rowrms_cv=float(W.pow(2).mean(1).sqrt().std() / W.pow(2).mean(1).sqrt().mean()))


def main():
    st = SafeTensors("Qwen3-0.6B")
    Q = hadamard(1024)
    assert torch.allclose(Q @ Q.T, torch.eye(1024, dtype=torch.float64), atol=1e-12)
    for tag, (name, side, gname, rows) in TARGETS.items():
        W = st.tensor(name)
        full_amax = None
        if rows:
            W = W[rows[0]:rows[1]]
        variants = {"raw": W}
        if gname:
            g = st.tensor(gname)
            variants["fused"] = W * g[None, :]
        base = variants.get("fused", W)
        variants["rotated"] = base @ Q if side == "in" else Q.T @ base
        out = dict(name=name, side=side, rows=np.array(rows if rows else (0, W.shape[0])))
        if gname:
            out["gain"] = g.numpy()
        for v, M in variants.items():
            out[f"{v}_W"] = M.numpy().astype(np.float32)
            out[f"{v}_raw"] = to_bf16_raw(M)
            s = summarize(M)
            for k, val in s.items():
                out[f"{v}_{k}"] = val
            ms = float(M.pow(2).mean())
            for fmt, q in F.QUANTIZERS.items():
                r = q(M, mode="nearest")
                e = r["deq"] - M
                out[f"{v}_{fmt}_err"] = e.numpy().astype(np.float32)
                out[f"{v}_{fmt}_scale"] = r["scale"].numpy()
                out[f"{v}_{fmt}_relmse"] = float(e.pow(2).mean() / ms)
                out[f"{v}_{fmt}_nib"] = F.fp4_nibbles(r).numpy()
            print(f"{tag:14s} {v:8s} kurt {s['kurtosis']:6.2f} colmax/med {s['colmax_med']:5.2f} rowmax/med {s['rowmax_med']:5.2f}"
                  f" colRMS cv {s['colrms_cv']:.3f} rowRMS cv {s['rowrms_cv']:.3f}"
                  f" | NVFP4 {out[f'{v}_NVFP4_relmse']:.5f} MXFP4 {out[f'{v}_MXFP4_relmse']:.5f}", flush=True)
        np.savez(os.path.join(HERE, "cache", f"hadamard_{tag}.npz"), **out)


if __name__ == "__main__":
    main()
