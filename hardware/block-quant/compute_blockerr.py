"""Quantize selected real Qwen3-0.6B matrices with NVFP4 and MXFP4 (RTN and stochastic) and cache everything
the Block Error / Scale map / Spectral / Bitplane renders need.

   python compute_blockerr.py            -> cache/mat_<tag>.npz  (one file per matrix)
"""
import os
import sys

import numpy as np
import torch

import formats as F
from weights import KINDS, SafeTensors, layer_name

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

LAYERS = [0, 13, 27]
EXTRA = [(16, "k_proj"), (26, "q_proj"), (26, "gate_proj"), (6, "down_proj")]
EMBED_CROPS = {"embed_head": (0, 3072), "embed_rare": (147456, 150528), "embed_mid": (60000, 63072), "embed_tail": (148864, 151936)}


def targets():
    for i in LAYERS:
        for k in KINDS:
            yield f"L{i:02d}_{k}", layer_name(i, k), None
    # chosen from the 0.6B atlas as the most outlier-structured matrices (row/col max-over-median, kurtosis,
    # spread of log2 block scale): see README "How the hero matrices were chosen"
    for i, k in EXTRA:
        yield f"L{i:02d}_{k}", layer_name(i, k), None
    for tag, (a, b) in EMBED_CROPS.items():
        yield tag, "model.embed_tokens.weight", (a, b)


def run(tag, name, rows, st):
    path = os.path.join(CACHE, f"mat_{tag}.npz")
    if os.path.exists(path):
        return
    W = st.tensor(name)
    raw = st.raw_u16(name)
    tensor_amax = float(W.abs().max())            # NVFP4 per-tensor scale uses the WHOLE tensor, even for crops
    if rows is not None:
        W = W[rows[0]:rows[1]]
        raw = raw[rows[0]:rows[1]]
    d = dict(W=W.numpy().astype(np.float32), raw=np.ascontiguousarray(raw), name=name,
             rows=np.array(rows if rows else (0, W.shape[0])))
    for fmt, q in F.QUANTIZERS.items():
        for mode in ("nearest", "stochastic"):
            kw = dict(tensor_amax=tensor_amax) if fmt == "NVFP4" else {}
            r = q(W, mode=mode, seed=1234, **kw)
            p = f"{fmt}_{mode[:3]}"
            d[p + "_err"] = (r["deq"] - W).numpy().astype(np.float32)
            d[p + "_nib"] = F.fp4_nibbles(r).numpy()
            if mode == "nearest":
                d[fmt + "_scale"] = r["scale"].numpy().astype(np.float64)
                d[fmt + "_scale_code"] = r["scale_code"].numpy()
                d[fmt + "_amax"] = r["amax"].numpy().astype(np.float32)
                d[fmt + "_clipped"] = np.packbits(r["clipped"].numpy(), axis=-1)
                if fmt == "NVFP4":
                    d["NVFP4_s2"] = np.array(r["s2"])
    f8 = F.fp8_e4m3_per_tensor(W)
    d["FP8_byte"] = f8["byte"].numpy()
    d["FP8_err"] = (f8["deq"] - W).numpy().astype(np.float32)
    np.savez(path, **d)
    e = d["NVFP4_nea_err"]
    print(f"{tag:18s} {tuple(W.shape)}  NVFP4 relMSE {np.mean(e**2)/np.mean(d['W'].astype(np.float64)**2):.4f}"
          f"  MXFP4 {np.mean(d['MXFP4_nea_err']**2)/np.mean(d['W'].astype(np.float64)**2):.4f}", flush=True)


if __name__ == "__main__":
    st = SafeTensors("Qwen3-0.6B")
    only = sys.argv[1:]
    for tag, name, rows in targets():
        if not only or tag in only:
            run(tag, name, rows, st)
