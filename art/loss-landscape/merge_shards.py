"""Merge sharded surfaces: python merge_shards.py resnet56_noshort_final_g101  (needs all shardKofN files)."""
import glob, json, sys, numpy as np
from common import CACHE
for stem in sys.argv[1:]:
    fs = sorted(glob.glob(f"{CACHE}/surf/{stem}.shard*of*.npz"))
    ds = [np.load(f) for f in fs]
    L = np.full(ds[0]["loss"].shape, np.nan); A = L.copy()
    for d in ds:
        m = np.isfinite(d["loss"]); L[m] = d["loss"][m]; A[m] = d["acc"][m]
    meta = json.loads(str(ds[0]["meta"])); meta["merged_from"] = fs
    meta["wall_s"] = sum(json.loads(str(d["meta"])).get("wall_s", 0) for d in ds)
    print(stem, len(fs), "shards, missing", int(np.isnan(L).sum()))
    if np.isnan(L).sum() == 0:
        np.savez(f"{CACHE}/surf/{stem}.npz", loss=L, acc=A, xs=ds[0]["xs"], ys=ds[0]["ys"], meta=json.dumps(meta))
