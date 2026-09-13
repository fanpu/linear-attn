"""Layer atlas: quantize every linear matrix of every layer (NVFP4, MXFP4; RTN and stochastic) and record
statistics + pooled thumbnails.   python compute_atlas.py [Qwen3-0.6B Qwen3-1.7B Qwen3-4B]
-> cache/atlas_<model>.npz
"""
import os
import sys

import numpy as np
import torch

import formats as F
from weights import KINDS, SafeTensors, layer_name

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
TH = 96   # thumbnail side


def pool(a, oh, ow):
    """area-mean pool a (H,W) to (oh,ow) with integer-ish bins (np.add.reduceat)."""
    H, W = a.shape
    ri = np.linspace(0, H, oh + 1).astype(int)[:-1]
    ci = np.linspace(0, W, ow + 1).astype(int)[:-1]
    s = np.add.reduceat(np.add.reduceat(a, ri, axis=0), ci, axis=1)
    cnt = np.add.reduceat(np.add.reduceat(np.ones_like(a), ri, axis=0), ci, axis=1)
    return s / cnt


def weave_stats(E, bs):
    """Correlation of |e| between horizontally adjacent elements, split by whether the pair straddles a
    block boundary (j % bs == bs-1) or lies inside one block."""
    A = np.abs(E)
    a, b = A[:, :-1], A[:, 1:]
    j = np.arange(A.shape[1] - 1)
    bound = (j % bs) == bs - 1

    def corr(x, y):
        x = x.ravel() - x.mean()
        y = y.ravel() - y.mean()
        return float((x * y).mean() / (x.std() * y.std() + 1e-30))
    return corr(a[:, ~bound], b[:, ~bound]), corr(a[:, bound], b[:, bound])


def stats_for(W, name):
    Wd = W.double()
    ms = float(Wd.pow(2).mean())
    out = dict(name=name, shape=tuple(W.shape))
    c = Wd - Wd.mean()
    out["kurtosis"] = float(c.pow(4).mean() / c.pow(2).mean() ** 2)
    A = Wd.abs()
    out["colmax_over_median"] = float(A.amax(0).max() / A.amax(0).median())
    out["rowmax_over_median"] = float(A.amax(1).max() / A.amax(1).median())
    out["rms"] = ms ** 0.5
    thumbs = {}
    for fmt, q in F.QUANTIZERS.items():
        r = q(Wd, mode="nearest")
        e = (r["deq"] - Wd)
        out[f"{fmt}_relmse"] = float(e.pow(2).mean() / ms)
        out[f"{fmt}_rmse"] = float(e.pow(2).mean() ** 0.5)
        rs = q(Wd, mode="stochastic", seed=7)
        out[f"{fmt}_relmse_sr"] = float((rs["deq"] - Wd).pow(2).mean() / ms)
        out[f"{fmt}_clip_frac"] = float(r["clipped"].double().mean())
        out[f"{fmt}_blocks_clipped"] = float(
            r["clipped"].reshape(W.shape[0], -1, r["block_size"]).any(-1).double().mean())
        ls = torch.log2(r["scale"]).numpy()
        out[f"{fmt}_log2scale_mean"] = float(ls.mean())
        out[f"{fmt}_log2scale_std"] = float(ls.std())
        codes = r["scale_code"].numpy().ravel()
        h = np.bincount(codes, minlength=256) / codes.size
        nz = h[h > 0]
        out[f"{fmt}_scale_code_entropy"] = float(-(nz * np.log2(nz)).sum())
        out[f"{fmt}_scale_code_hist"] = h
        out[f"{fmt}_level_hist"] = np.bincount(r["idx"].numpy().ravel(), minlength=8) / r["idx"].numel()
        # fraction of blocks whose max element sits on the top level; blocks where >= 12 of 16/32 elements
        # land on 0 or +-0.5 ("crushed" blocks)
        idx = r["idx"].numpy().reshape(W.shape[0], -1, r["block_size"])
        out[f"{fmt}_crushed_blocks"] = float(((idx <= 1).mean(-1) >= 0.75).mean())
        En = e.numpy()
        wi, wb = weave_stats(En, r["block_size"])
        out[f"{fmt}_weave_corr_within"], out[f"{fmt}_weave_corr_boundary"] = wi, wb
        thumbs[f"{fmt}_scale"] = pool(ls, TH, TH).astype(np.float32)
        thumbs[f"{fmt}_err"] = pool(np.abs(En), TH, TH).astype(np.float32)
    wi, wb = weave_stats(Wd.numpy(), 16)
    out["null_W_corr_within16"], out["null_W_corr_boundary16"] = wi, wb
    thumbs["absW"] = pool(A.numpy(), TH, TH).astype(np.float32)
    return out, thumbs


def main(model):
    st = SafeTensors(model)
    L = 1 + max(int(k.split(".")[2]) for k in st.names() if k.startswith("model.layers."))
    rows, thumbs = [], {}
    for i in range(L):
        for k in KINDS:
            nm = layer_name(i, k)
            s, th = stats_for(st.tensor(nm, torch.float64), nm)
            s["layer"], s["kind"] = i, k
            rows.append(s)
            for kk, v in th.items():
                thumbs.setdefault(kk, []).append(v)
            print(f"{model} L{i:02d} {k:9s} nv {s['NVFP4_relmse']:.4f} mx {s['MXFP4_relmse']:.4f} "
                  f"kurt {s['kurtosis']:.1f} weave nv {s['NVFP4_weave_corr_within']:.3f}/{s['NVFP4_weave_corr_boundary']:.3f}",
                  flush=True)
    keys = [k for k in rows[0] if k not in ("name", "shape")]
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    arr["names"] = np.array([r["name"] for r in rows])
    arr["shapes"] = np.array([r["shape"] for r in rows])
    for kk, v in thumbs.items():
        arr["thumb_" + kk] = np.stack(v).reshape(L, len(KINDS), TH, TH)
    np.savez(os.path.join(HERE, "cache", f"atlas_{model}.npz"), **arr, n_layers=L, kinds=np.array(KINDS))


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["Qwen3-0.6B"]):
        main(m)
