"""Re-evaluate trained deep-LSA / GD-k runs with many prompts and report the heavy tail.

A depth-L linear-attention network is a polynomial of degree ~3^L in the prompt, so its squared error is
heavy-tailed: the mean is dominated by rare prompts with large inputs.  For each run we record the mean
squared error with a standard error, the median, and the mean after dropping the worst 0.1% of prompts.

  .venv/bin/python 08-icl-linear-attention/eval_deep.py
"""
import json, pathlib, sys
import numpy as np
import torch
sys.argv = [sys.argv[0]]
HERE = pathlib.Path(__file__).parent
src = open(HERE / "train_lsa_deep.py").read().split("configs = [")[0]
exec(src)          # brings lsa_forward, prompts, make_cov, ... with dev=cpu, float64 evaluation
torch.set_num_threads(4)
out = {}
for f in sorted((HERE / "cache").glob("lsa_deep_*.json")):
    runs = json.load(open(f))["runs"]
    for r in runs:
        kind, L, d, n, sigma, cov = r["cfg"]
        if kind not in ("gd", "pgd", "dense"):
            continue
        params = {k: [torch.tensor(np.array(v), dtype=DT) for v in vs] for k, vs in r["params"].items()}
        Lc = torch.linalg.cholesky(make_cov(cov, d))
        se = []
        with torch.no_grad():
            for _ in range(32):
                X, y, xq, yq = prompts(8192, n, d, Lc, sigma)
                se.append(((lsa_forward(params, kind, X, y, xq) - yq) ** 2))
        se = torch.cat(se).numpy()
        srt = np.sort(se)
        key = f"{f.stem}|{kind}|{L}|{n}|{sigma}|{cov}"
        out[key] = {"kind": kind, "L": L, "d": d, "n": n, "sigma": sigma, "cov": cov, "init": r.get("init"),
                    "mean": float(se.mean()), "sem": float(se.std() / np.sqrt(len(se))), "median": float(np.median(se)),
                    "trim999": float(srt[: int(0.999 * len(srt))].mean()), "max": float(srt[-1]),
                    "train_tail": float(np.mean(r["curve"][-10:]))}
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[key].items() if k in ("mean", "sem", "median", "trim999", "train_tail")}, flush=True)
json.dump(out, open(HERE / "cache" / "deep_eval.json", "w"), indent=1)
