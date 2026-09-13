"""Concatenate zoom-chain files computed in pieces (same N, D, seed, dtype, res, label) into one chain.

  python merge_chains.py --out cache/zoom_Bvideoall_N100_D1000_s0_f32_r1280.npz \
      cache/zoom_Bvideo_N100_D1000_s0_f32_r1280.npz:0-4 cache/zoom_Bvideo2_N100_D1000_s0_f32_r1280.npz
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("parts", nargs="+", help="file or file:a-b (inclusive level range)")
args = ap.parse_args()
acc, meta = {k: [] for k in ("windows", "L_D", "L_avg", "t_hit", "L_mf")}, None
for p in args.parts:
    f, _, rng = p.partition(":")
    Z = np.load(f)
    n = Z["L_avg"].shape[0]
    a, b = (map(int, rng.split("-")) if rng else (0, n - 1))
    for k in acc:
        acc[k].append(Z[k][a:b + 1])
    m = {k: str(Z[k]) for k in ("N", "D", "seed", "dtype", "res", "label", "tau", "step")}
    assert meta is None or meta == m, (meta, m)
    meta = m
out = {k: np.concatenate(v) for k, v in acc.items()}
np.savez_compressed(args.out, **out, N=int(meta["N"]), D=int(meta["D"]), seed=int(meta["seed"]), dtype=meta["dtype"],
                    res=int(meta["res"]), label=meta["label"], tau=float(meta["tau"]), step=float(meta["step"]))
print(args.out, out["L_avg"].shape)
