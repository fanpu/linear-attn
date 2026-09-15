#!/usr/bin/env python3
"""
capacity.py

Question: a linear-attention state S in R^{d x d} stores n key/value pairs. How does retrieval
degrade as n/d grows, and does the write rule matter?

Setup (one trial):
  keys     n random unit vectors in R^d (not orthogonal: realistic, and forced once n > d)
  values   n entries drawn from a codebook of V random unit vectors in R^d (think token embeddings)
  write    one of
             additive   S = sum_i v_i k_i^T                         (vanilla linear attention)
             delta      S <- S + beta (v_t - S k_t) k_t^T, in order   (DeltaNet, beta = 1)
             lstsq      S = V^T (K^T)^+                              (best any linear map can do)
  read     r_j = S k_j for every stored key
  score    relative error ||r_j - v_j||^2 / ||v_j||^2, and top-1 accuracy of decoding r_j
           against the whole codebook (chance = 1/V)
           and the split r_j = sig * v_j + noise: sig = <r_j, v_j>, noise = ||r_j - sig v_j||^2

Predictions for random unit keys:
  additive  error ~ (n-1)/d: crosstalk k_i.k_j ~ N(0, 1/d), summed over n-1 other items
  lstsq     error ~ max(0, 1 - d/n): exact while n <= d, then the rank-d projection loses the rest
  delta     recency-biased: each later write multiplies S by (I - k k^T), which scales the signal
            along an old key by (1 - (k.k_j)^2) ~ (1 - 1/d), so sig ~ exp(-age/d)

Written to results/:
  metrics.json   per method, d, n: mean error / accuracy (and std over trials)
  by_age.json    per method at n/d in {1, 2}: accuracy, error, sig, noise vs how many writes came after
  (capacity.png is drawn by plot.py)

Usage
  python capacity.py                     # d in {32, 64, 128}, 30 trials, ~1 min on CPU
  python capacity.py --dims 64 --trials 5
"""

import argparse
import json
from pathlib import Path

import numpy as np

RATIOS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
AGE_RATIOS = [1.0, 2.0]
METHODS = ["additive", "delta", "lstsq"]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def write_additive(K, Vals):
    return Vals.T @ K


def write_delta(K, Vals, beta=1.0):
    d = K.shape[1]
    S = np.zeros((Vals.shape[1], d))
    for k, v in zip(K, Vals):
        S += beta * np.outer(v - S @ k, k)
    return S


def write_lstsq(K, Vals):
    # minimise ||K S^T - Vals||_F: S^T = K^+ Vals
    return (np.linalg.pinv(K) @ Vals).T


WRITERS = {"additive": write_additive, "delta": write_delta, "lstsq": write_lstsq}


def trial(rng, d, n, codebook):
    K = unit(rng.standard_normal((n, d)))
    ids = rng.integers(len(codebook), size=n)
    Vals = codebook[ids]
    out = {}
    for name, write in WRITERS.items():
        R = K @ write(K, Vals).T  # row j = S k_j
        err = np.sum((R - Vals) ** 2, axis=1)  # values are unit norm
        acc = (np.argmax(R @ codebook.T, axis=1) == ids).astype(float)
        sig = np.sum(R * Vals, axis=1)
        noise = np.sum((R - sig[:, None] * Vals) ** 2, axis=1)
        out[name] = dict(err=err, acc=acc, sig=sig, noise=noise)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    metrics, by_age = [], []
    for d in args.dims:
        codebook = unit(rng.standard_normal((args.vocab, d)))
        for ratio in RATIOS:
            n = max(1, round(ratio * d))
            runs = [trial(rng, d, n, codebook) for _ in range(args.trials)]
            for m in METHODS:
                # (trials, n), column = write position
                X = {k: np.stack([r[m][k] for r in runs]) for k in ("err", "acc", "sig", "noise")}
                row = dict(method=m, d=d, n=n, ratio=n / d)
                for k, v in X.items():
                    row[k], row[k + "_std"] = v.mean(), v.mean(1).std()
                metrics.append(row)
                if ratio in AGE_RATIOS:
                    # age = number of writes after this one; column n-1 has age 0
                    by_age.append(dict(method=m, d=d, n=n, ratio=ratio, age=list(range(n)),
                                       **{k: v.mean(0)[::-1].tolist() for k, v in X.items()}))
            row = {m: f"{np.mean([r[m]['acc'] for r in runs]):.3f}" for m in METHODS}
            print(f"d={d:4d} n={n:4d} n/d={n / d:5.3f}  acc {row}", flush=True)

    (args.out / "metrics.json").write_text(json.dumps(dict(config=vars(args) | {"out": str(args.out)},
                                                           rows=metrics), indent=1))
    (args.out / "by_age.json").write_text(json.dumps(by_age))
    print(f"wrote {args.out}/metrics.json, by_age.json")


if __name__ == "__main__":
    main()
