"""Scan every Qwen3-0.6B weight matrix for bitplane stripe structure, to choose the Bitplanes hero honestly.
For each matrix and each bf16 plane: entropy H, row-stripe and column-stripe overdispersion (see compute_bitplanes).
Embeddings are scanned in 3072-row windows (the matrix is 151936 x 1024).
-> cache/bitplane_scan.npz, printed ranking
"""
import os

import numpy as np

from compute_bitplanes import plane_stats
from weights import KINDS, SafeTensors, layer_name

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    st = SafeTensors("Qwen3-0.6B")
    items = [(f"L{i:02d}_{k}", layer_name(i, k), None) for i in range(28) for k in KINDS]
    for a in range(0, 151936 - 3072 + 1, 3072 * 4):
        items.append((f"embed_{a}", "model.embed_tokens.weight", (a, a + 3072)))
    tags, H, ODr, ODc = [], [], [], []
    for tag, nm, rows in items:
        raw = st.raw_u16(nm)
        if rows:
            raw = raw[rows[0]:rows[1]]
        hs, rs, cs = [], [], []
        for k in range(16):
            s = plane_stats(((raw >> (15 - k)) & 1).astype(np.uint8))
            hs.append(s["H"]); rs.append(s["od_row"]); cs.append(s["od_col"])
        tags.append(tag); H.append(hs); ODr.append(rs); ODc.append(cs)
        print(f"{tag:16s} expb3 row x{rs[5]:7.1f} col x{cs[5]:6.1f} | sign row x{rs[0]:5.1f} col x{cs[0]:6.1f} | mant b0 row x{rs[15]:.1f} col x{cs[15]:.1f}", flush=True)
    H, ODr, ODc = map(np.array, (H, ODr, ODc))
    np.savez(os.path.join(HERE, "cache", "bitplane_scan.npz"), tags=np.array(tags), H=H, od_row=ODr, od_col=ODc)
    score = np.log(ODr[:, 5:8]).mean(1) + np.log(ODc[:, 5:8]).mean(1)
    print("\nranking by mean log stripe overdispersion of exp b3..b1 (rows+cols):")
    for i in np.argsort(-score)[:15]:
        print(f"  {tags[i]:16s} score {score[i]:.2f}  row x{ODr[i, 5:8].round(1)} col x{ODc[i, 5:8].round(1)}")


if __name__ == "__main__":
    main()
