"""
Day 3: table + plot from runs/*/. Ownership: Claude (loud) except seed_stats (Fan Pu, silent).
Usage: python summarize.py [--runs runs] [--sizes 30M 60M 125M]
"""
import argparse, glob, json, math, os

import numpy as np


# ======================= FAN PU OWNS THIS (silent-failure code) =======================
def seed_stats(final):
    """
    final: dict size -> list of final_val_loss over seeds, e.g. {"30M": [4.41, 4.43], "60M": [...]}.
    Return a dict with, per size: mean, s (sample std, ddof=1), n; and pooled over sizes:
      s_pooled = sqrt( sum_s sum_k (x_sk - xbar_s)^2 / nu ),  nu = sum_s (n_s - 1)
      ci80 = [s_pooled * sqrt(nu / chi2_{0.9,nu}),  s_pooled * sqrt(nu / chi2_{0.1,nu})]
        (chi2 quantiles: nu=1: 0.0158 / 2.7055; nu=2: 0.2107 / 4.6052; nu=3: 0.5844 / 6.2514)
      se_diff_n2 = s_pooled * sqrt(2/2)   # SE of (mean_A - mean_B) with 2 seeds per arm
      mdd_n2 = 2.5 * se_diff_n2           # the smallest cross-mixer difference you would call, at n=2
    Derivation is in day3_assignment.md §2.3. Do not import scipy for this; the constants are above.
    """
    raise NotImplementedError("Fan Pu writes seed_stats")


# =======================================================================================


def load_runs(runs_dir):
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*_s*"))):
        rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
        meta = next((r["meta"] for r in rows if "meta" in r), None)
        if meta is None:
            continue
        summ = os.path.join(d, "summary.json")
        out[os.path.basename(d)] = dict(meta=meta, rows=rows,
                                        summary=json.load(open(summ)) if os.path.exists(summ) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--sizes", nargs="+", default=["30M", "60M", "125M"])
    a = ap.parse_args()
    runs = load_runs(a.runs)

    print(f"{'run':10s} {'steps':>7s} {'tokens':>7s} {'final_val':>10s} {'tok/s':>8s} {'peakGB':>7s} {'status':>8s}")
    final = {}
    for name, r in runs.items():
        m, s = r["meta"], r["summary"]
        done = s is not None
        last = max((x["step"] for x in r["rows"] if "step" in x), default=0)
        print(f"{name:10s} {last:7d} {m['tokens_budget']/1e6:6.0f}M "
              f"{(s['final_val_loss'] if done else float('nan')):10.4f} "
              f"{(s['tok_s_median']/1e3 if done else float('nan')):7.1f}k "
              f"{(s['peak_mem_GB'] if done else float('nan')):7.1f} {'done' if done else 'running':>8s}")
        if done:
            final.setdefault(m["size"], []).append(s["final_val_loss"])

    if all(len(final.get(sz, [])) >= 2 for sz in a.sizes) or any(len(v) >= 2 for v in final.values()):
        try:
            st = seed_stats({k: v for k, v in final.items() if len(v) >= 2})
            print("\nseed_stats:", json.dumps(st, indent=2))
        except NotImplementedError as e:
            print(f"\n({e})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {sz: c for sz, c in zip(a.sizes, ["C0", "C1", "C2", "C3"])}
    for name, r in runs.items():
        m = r["meta"]
        if m["size"] not in colors:
            continue
        ev = [(x["tokens"], x["val_loss"]) for x in r["rows"] if "val_loss" in x and x["step"] > 0]
        tr = [(x["tokens"], x["train_loss"]) for x in r["rows"] if "train_loss" in x]
        c, ls = colors[m["size"]], ["-", "--", ":"][m["seed"] % 3]
        if ev:
            ax[0].plot([t / 1e6 for t, _ in ev], [v for _, v in ev], ls, color=c, marker="o", ms=3,
                       label=f"{m['size']} seed {m['seed']}")
        if tr:
            k = 25  # running mean over 25 log rows (= 500 steps) to make the train curve readable
            tl = np.convolve([v for _, v in tr], np.ones(k) / k, mode="valid")
            ax[1].plot([t / 1e6 for t, _ in tr][k - 1:], tl, ls, color=c, label=f"{m['size']} seed {m['seed']}")
    for x, ttl in zip(ax, ["validation loss (fixed 4.2M-token set)", "train loss (500-step running mean)"]):
        x.set_xscale("log"); x.set_xlabel("tokens seen (M)"); x.set_ylabel("nats/token"); x.set_title(ttl, fontsize=9)
        x.grid(True, which="both", alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("day3_baselines.png", dpi=150)
    print("wrote day3_baselines.png")


if __name__ == "__main__":
    main()
