"""The README's numbers, from summary.json.

For every group (arch_width_opt) with an A->B run and its nulls:
  psnr1           PSNR on A at the end of phase 1
  psnrB           PSNR on B at the end of phase 2
  hlA / hlB       per-band half-life (phase-2 steps) of A's remaining amplitude g_A, and of B's
                  unwritten fraction; hlA < hlB means A is scraped before B is written
  scrape          per band, min over t of max(A remaining, B written): < 0.5 means there is a
                  moment when that band of the page holds less than half of either text
  endA            A's letter-specific amplitude at the last step, and the same number in the
                  C->B and none->B runs (which never saw A), and the decoy spread
  off             the same, measured half a pixel off the training grid
  relearn         PSNR on A after 10 / 30 / 100 steps of retraining on A, from the final network
"""
import json
import os
import sys

import numpy as np


def fmt(a, p=2):
    return "[" + " ".join(("  inf" if np.isinf(x) else "  nan" if np.isnan(x) else f"{x:5.{p}f}" if abs(x) < 100 else f"{x:5.0f}") for x in a) + "]"


def main():
    root = sys.argv[1]
    S = {s["name"]: s for s in json.load(open(os.path.join(root, "summary.json")))}
    groups = sorted({k.split("_AB_")[0] + "|" + k.split("_AB_")[1] for k in S if "_AB_" in k})
    rows = []
    for g in groups:
        pre, suf = g.split("|")
        A = S[pre + "_AB_" + suf]
        C = S.get(pre + "_CB_" + suf)
        N = S.get(pre + "_noneB_" + suf)
        gA = np.array(A["gA"]); gB = np.array(A["gB"])
        scrape = np.maximum(gA, 1 + gB).min(0)
        print(f"\n== {pre}_{suf}   bands {A['bands']}")
        print(f"   psnr1 {A['psnr1']:.1f}  psnrB {A['psnrB_final']:.1f}" + (f"   (none->B psnrB {N['psnrB_final']:.1f})" if N else ""))
        print("   hlA      ", fmt(A["half_life_gA"], 1))
        print("   hlB      ", fmt(A["half_life_Bunwritten"], 1))
        print("   t10 lettr", fmt(A["tenth_life_letters"], 0))
        if N:
            print("   hlB none ", fmt(N["half_life_Bunwritten"], 1))
        print("   scrape   ", fmt(scrape))
        print(f"   palimpsest index: B coarse over A fine {A['pal_fwd']:.2f} (step {A['pal_fwd_step']});  A coarse under B fine {A['pal_rev']:.2f} (step {A['pal_rev_step']})")
        if "fitA" in A:
            print("   fitA     ", fmt(A["fitA"]))
        print("   endA     ", fmt(np.array(A["letters"])[-1], 4), " decoy sd", fmt(np.array(A["gD_std"])[-1], 4))
        for nm, X in (("C->B", C), ("none", N)):
            if X:
                print(f"   end {nm:5s}", fmt(np.array(X["letters"])[-1], 4))
        offA = np.array(A["off_gA"]) - np.array(A["off_gD_mean"])
        print("   offA-D   ", fmt(offA, 3), " sd", fmt(A["off_gD_std"], 3), f" off-grid psnrB {A['off_psnrB']:.1f}")
        for nm, X in (("C->B", C), ("none", N)):
            if X:
                print(f"   off {nm:5s}", fmt(np.array(X["off_gA"]) - np.array(X["off_gD_mean"]), 3))
        rl = lambda X: [X["relearn_psnr"].get(k, np.nan) for k in ("10", "30", "100")] if X and "relearn_psnr" in X else [np.nan] * 3
        print("   relearn A@10/30/100:  A->B", fmt(rl(A), 1), " C->B", fmt(rl(C), 1), " none", fmt(rl(N), 1))
        rows.append(dict(group=pre + "_" + suf, psnr1=A["psnr1"], psnrB=A["psnrB_final"], hlA=A["half_life_gA"],
                         hlB=A["half_life_Bunwritten"], scrape=scrape.tolist(), bands=A["bands"],
                         relearn=dict(AB=rl(A), CB=rl(C), none=rl(N))))
    json.dump(rows, open(os.path.join(root, "table.json"), "w"), default=float)


if __name__ == "__main__":
    main()
