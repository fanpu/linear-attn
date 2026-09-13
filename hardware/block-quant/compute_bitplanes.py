"""Bitplane statistics for the stored bf16 bits and for the FP8 / FP4 codes of the cached matrices.

For every plane (a 0/1 image) we measure
  p1        fraction of ones
  H         binary entropy of p1                                   (bits/pixel; 1 = coin flip, 0 = constant)
  Hc        conditional entropy given the causal neighbours (left, up, up-left), from 8-context counts
  MI = H - Hc   bits/pixel predictable from neighbours; MI/H = "structure fraction"
  rh, rv    lag-1 autocorrelation along rows / along columns of the +-1 image
  null      the same statistics after a random pixel permutation (seeded), i.e. what pure noise with the
            same p1 gives.  z = r * sqrt(N) for the autocorrelation under the null.
-> cache/bitplanes.npz  (+ printed table)
"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def H2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))) if 0 < p < 1 else 0.0


def plane_stats(b):
    b = b.astype(np.uint8)
    p1 = b.mean()
    H = H2(p1) if 0 < p1 < 1 else 0.0
    x, l, u, ul = b[1:, 1:], b[1:, :-1], b[:-1, 1:], b[:-1, :-1]
    ctx = (l.astype(np.int32) << 2) | (u.astype(np.int32) << 1) | ul
    n = ctx.size
    Hc = 0.0
    for c in range(8):
        m = ctx == c
        nc = m.sum()
        if nc:
            Hc += nc / n * H2(x[m].mean())
    s = b.astype(np.float64) * 2 - 1
    s -= s.mean()
    sd = s.std()
    if sd == 0:
        rh = rv = 0.0
    else:
        rh = float((s[:, 1:] * s[:, :-1]).mean() / sd ** 2)
        rv = float((s[1:, :] * s[:-1, :]).mean() / sd ** 2)
    # stripe structure: variance of row (column) means relative to the binomial variance expected for iid bits
    var_bin = p1 * (1 - p1)
    od_row = float(b.mean(1).var() / (var_bin / b.shape[1])) if var_bin > 0 else 1.0
    od_col = float(b.mean(0).var() / (var_bin / b.shape[0])) if var_bin > 0 else 1.0
    return dict(p1=float(p1), H=H, Hc=float(Hc), MI=float(max(H - Hc, 0.0)), rh=rh, rv=rv, od_row=od_row, od_col=od_col)


def planes_bf16(raw):
    return [((raw >> (15 - k)) & 1).astype(np.uint8) for k in range(16)]      # k=0 -> bit 15 (sign)


BF16_LABELS = ["sign"] + [f"exp b{7 - i}" for i in range(8)] + [f"mant b{6 - i}" for i in range(7)]
FP8_LABELS = ["sign", "exp b3", "exp b2", "exp b1", "exp b0", "mant b2", "mant b1", "mant b0"]
FP4_LABELS = ["sign", "exp b1", "exp b0", "mant"]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "mat_*.npz"))):
        tag = os.path.basename(f)[4:-4]
        d = np.load(f)
        sets = {
            "bf16": (planes_bf16(d["raw"]), BF16_LABELS),
            "fp8": ([((d["FP8_byte"] >> (7 - k)) & 1) for k in range(8)], FP8_LABELS),
            "nvfp4": ([((d["NVFP4_nea_nib"] >> (3 - k)) & 1) for k in range(4)], FP4_LABELS),
            "mxfp4": ([((d["MXFP4_nea_nib"] >> (3 - k)) & 1) for k in range(4)], FP4_LABELS),
            "nvfp4_scale": ([((d["NVFP4_scale_code"] >> (7 - k)) & 1) for k in range(8)], [f"E4M3 b{7 - k}" for k in range(8)]),
            "mxfp4_scale": ([((d["MXFP4_scale_code"] >> (7 - k)) & 1) for k in range(8)], [f"E8M0 b{7 - k}" for k in range(8)]),
        }
        for sname, (planes, labels) in sets.items():
            rows = []
            for b in planes:
                s = plane_stats(b)
                perm = rng.permutation(b.ravel()).reshape(b.shape)
                sn = plane_stats(perm)
                s.update(null_MI=sn["MI"], null_rh=sn["rh"], null_rv=sn["rv"], null_od_row=sn["od_row"], null_od_col=sn["od_col"], N=b.size)
                rows.append(s)
            for key in rows[0]:
                out[f"{tag}/{sname}/{key}"] = np.array([r[key] for r in rows])
            out[f"{tag}/{sname}/labels"] = np.array(labels)
            if sname in ("bf16", "fp8", "nvfp4"):
                print(f"\n{tag} {sname}  {d['raw'].shape}")
                print("  plane        p1     H      MI    r_row   r_col  rowOD  colOD | null MI  null r_row null rowOD")
                for lab, r in zip(labels, rows):
                    print(f"  {lab:10s} {r['p1']:.3f} {r['H']:.3f} {r['MI']:.4f}"
                          f" {r['rh']:+.3f} {r['rv']:+.3f} {r['od_row']:6.1f} {r['od_col']:6.1f} | {r['null_MI']:.5f}  {r['null_rh']:+.4f} {r['null_od_row']:.2f}")
    np.savez(os.path.join(CACHE, "bitplanes.npz"), **out)


if __name__ == "__main__":
    main()
