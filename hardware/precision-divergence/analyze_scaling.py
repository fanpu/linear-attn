"""How long are the cycles?  Exact minifloat family + float32/float64 orbit census vs random-map theory.

Random-map / Grebogi-Ott-Yorke heuristic: a chaotic map on a lattice behaves like a random map on
N_eff states, with N_eff = 1 / sum_i mu_i^2 (mu = invariant measure of each rounding cell); the expected
number of cyclic states is ~ sqrt(pi N_eff / 2).

    python analyze_scaling.py    # -> gallery/cycle_scaling.png, cache/scaling.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
CACHE, GALLERY = HERE / "cache", HERE / "gallery"


def neff_integral(p, emin, emax=-1, npts=200001):
    """Continuous approximation of 1/sum mu_i^2 for a binary float with p-bit significand on [2^emin, 1]."""
    tot = 0.0
    for e in range(emin, emax + 1):
        a, b = 2.0 ** e, 2.0 ** (e + 1)
        u = np.linspace(0, 1, npts)[1:-1]
        x = a + (b - a) * u
        sp = 2.0 ** (e - p + 1)
        # sum over cells of (rho * sp)^2 = integral rho^2 * sp dx ; cap the x->1 singularity at one cell
        x = np.minimum(x, 1 - sp)
        tot += np.mean(1 / (np.pi ** 2 * x * (1 - x))) * (b - a) * sp
    return 1.0 / tot


def main():
    S = json.loads((CACHE / "graph_stats.json").read_text())
    fam = S["family"]
    out = {}
    fig, ax = plt.subplots(figsize=(13, 9), dpi=200, facecolor="#f3eee2")
    ax.set_facecolor("#f3eee2")
    cols = {3: "#2f5d8a", 4: "#3f8f5a", 5: "#d08a1f"}
    for E in (3, 4, 5):
        f = [x for x in fam if x["E"] == E and x["N"] > 50]
        ne = np.array([x["N_eff_invariant"] for x in f])
        ax.loglog(ne, [x["n_cyclic"] for x in f], "o-", color=cols[E], ms=5, lw=1, label=f"exact minifloat E={E}, M=1… (all cyclic states)")
        nm = [(x["N_eff_invariant"], x["null_mean"]["n_cyclic"]) for x in f if "null_mean" in x]
        if nm:
            ax.loglog(*zip(*nm), "x", color=cols[E], ms=7, alpha=0.6)
    for key, lab, c in (("float16|4x(1-x)", "float16 (exact graph)", "#b8322a"), ("bfloat16|4x(1-x)", "bfloat16 (exact graph)", "#7a4fb0")):
        s = S[key]
        ax.loglog(s["N_eff_invariant"], s["n_cyclic"], "s", ms=12, mfc="none", mew=2.5, color=c, label=lab)
        ax.loglog(s["N_eff_invariant"], s["null_mean"]["n_cyclic"], "x", ms=12, mew=2.5, color=c)
    census = {}
    for t, p, emin in (("f32", 24, -126), ("f64", 53, -300)):
        d = pd.read_csv(CACHE / f"cycles_{t}.tsv", sep="\t")
        cyc = d.groupby(["lambda", "cycle_min"]).size()
        total_cyclic = int(sum(l for l, _ in cyc.index))
        dom = int(cyc.idxmax()[0])
        ne = neff_integral(p, emin)
        census[t] = dict(seeds=len(d), cycles_found=[(int(l), float(m), int(n)) for (l, m), n in cyc.items()],
                         total_cyclic_states_found=total_cyclic, dominant_period=dom, N_eff_integral=ne,
                         random_map_expected_cyclic=float(np.sqrt(np.pi * ne / 2)), median_tail=float(d.mu.median()))
        ax.loglog(ne, total_cyclic, "D", ms=12, color="#1b1a17", mfc="none" if t == "f32" else "#1b1a17",
                  label=f"{'float32' if t == 'f32' else 'float64'}: cyclic states on cycles found from {len(d)} seeds (N_eff approx.)")
    xx = np.logspace(1, 17, 50)
    ax.loglog(xx, np.sqrt(np.pi * xx / 2), "k--", lw=1, label="random map: sqrt(pi N_eff / 2)")
    ax.set_xlabel("N_eff = 1 / sum(mu_cell^2)   (effective number of states under the invariant measure)", family="Nimbus Mono PS")
    ax.set_ylabel("number of states lying on cycles", family="Nimbus Mono PS")
    ax.legend(frameon=False, prop=dict(family="Nimbus Mono PS", size=9), loc="upper left")
    ax.set_title("Cycle census of x -> round(4x(1-x)); crosses = in-degree-preserving null", family="P052", fontsize=18, loc="left")
    fig.savefig(GALLERY / "cycle_scaling.png", facecolor="#f3eee2")
    # fits
    f = [x for x in fam if x["N"] > 1000]
    ne = np.array([x["N_eff_invariant"] for x in f])
    nc = np.array([x["n_cyclic"] for x in f])
    out["family_fit_slope_log_ncyclic_vs_log_Neff"] = float(np.polyfit(np.log(ne), np.log(nc), 1)[0])
    out["family_median_ratio_ncyclic_over_sqrt_piNeff_over_2"] = float(np.median(nc / np.sqrt(np.pi * ne / 2)))
    out["census"] = census
    (CACHE / "scaling.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
