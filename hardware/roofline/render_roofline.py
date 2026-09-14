"""render_roofline.py - the Roofline as a night sky, from cache/roofline.json (no GPU).

    python render_roofline.py

Measured: the two roofs (peak bf16 GEMM, peak device bandwidth) and every star's (operational intensity,
achieved FLOP/s) from timed kernels with analytic FLOP/byte counts. Declared: the sky, the star glyphs, sizes
(by wall time), constellation lines joining a kernel family in parameter order, and the horizon composition.
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"
INK = "#1b1b1b"; PAPER = "#f3eee3"; NIGHT = "#06060c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6})
GROUPS = {  # group -> (constellation name, colour night, colour paper, marker)
    "gemm": ("The Square (GEMM n²)", "#fff3b0", "#b8860b", "*"),
    "linear": ("The Ladder (decode linear, B = 1…2048)", "#ffb703", "#c1121f", "o"),
    "sdpa": ("The Fan (causal attention, T)", "#8ecae6", "#005caf", "D"),
    "prefill": ("The Prow (Qwen3-0.6B prefill, T)", "#95d5b2", "#1b813e", "^"),
    "decode": ("The Chain (Qwen3-0.6B decode, B)", "#f4a261", "#ca7a2c", "s"),
    "copy": ("The Riverbed (copy)", "#a8dadc", "#0089a7", "v"),
    "read": ("The Riverbed (read)", "#a8dadc", "#0089a7", "v"),
    "eltwise": ("Dust (elementwise)", "#cdb4db", "#592c63", "x"),
}


def load():
    d = json.load(open(f"{CACHE}/roofline.json"))
    for s in d["stars"]:
        s["oi"] = s["flops"] / s["bytes"] if s["flops"] > 0 else np.nan
        s["tflops"] = s["flops"] / s["sec"] / 1e12
        s["gbps"] = s["bytes"] / s["sec"] / 1e9
    return d


def plate(d, style="night"):
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    m = d["meta"]; pk, bw = m["gemm_peak_tflops"], m["bw_peak_gbps"]; ridge = pk * 1e12 / (bw * 1e9)
    fig = plt.figure(figsize=(18, 11), facecolor=bg)
    ax = fig.add_axes([0.07, 0.10, 0.90, 0.80]); ax.set_facecolor(bg)
    if dark:   # a faint star field and a Milky Way haze, declared decoration
        rng = np.random.default_rng(7)
        xs = 10 ** rng.uniform(-1.3, 4.3, 900); ys = 10 ** rng.uniform(-2.3, np.log10(pk) + 0.35, 900)
        ax.scatter(xs, ys, s=rng.uniform(.1, 1.4, 900), color="#ffffff", alpha=.35, lw=0, zorder=0)
    oi = np.logspace(-1.3, 4.3, 400)
    roof = np.minimum(pk, bw * oi / 1e3)
    ax.fill_between(oi, 1e-3, roof, color=("#10121e" if dark else "#e6e1d4"), zorder=1, lw=0)
    ax.plot(oi, roof, color=fg, lw=1.6, zorder=2)
    ax.plot(oi, np.full_like(oi, pk), ":", color=fg, lw=.6, alpha=.6); ax.plot(oi, bw * oi / 1e3, ":", color=fg, lw=.6, alpha=.6)
    ax.axvline(ridge, color=fg, lw=.5, ls=(0, (2, 4)), alpha=.7)
    ax.text(ridge, 2e-3, f"  ridge {ridge:.0f} FLOP/byte", color=fg, fontsize=8, rotation=90, va="bottom", ha="left")
    ax.text(oi[-1], pk * 1.12, f"compute roof {pk:.1f} TFLOP/s (bf16 GEMM, measured)  ", ha="right", color=fg, fontsize=8.5)
    ax.text(0.06, bw * 0.06 / 1e3 * 1.25, f"memory roof {bw:.0f} GB/s (measured)", rotation=39, color=fg, fontsize=8.5, rotation_mode="anchor")
    for g, (name, cn, cp, mk) in GROUPS.items():
        S = [s for s in d["stars"] if s["group"] == g and np.isfinite(s["oi"])]
        if not S:
            continue
        c = cn if dark else cp
        x = np.array([s["oi"] for s in S]); y = np.array([s["tflops"] for s in S])
        size = 28 + 60 * np.log10(np.array([s["sec"] for s in S]) * 1e3 + 1)
        if len(S) > 1:
            ax.plot(x, y, "-", color=c, lw=.7, alpha=.55, zorder=3)
        if dark:
            ax.scatter(x, y, s=size * 5, color=c, alpha=.12, lw=0, zorder=3)
        ax.scatter(x, y, s=size, color=c, marker=mk, edgecolor=fg if not dark else "none", lw=.4, zorder=4, label=name)
        for s, xx, yy in zip(S, x, y):
            lab = s["label"].split()[-1]
            ax.annotate(lab, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=6.2, color=c, alpha=.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 2e4); ax.set_ylim(1.5e-3, pk * 1.6)
    ax.set_xlabel("operational intensity (FLOP per byte of minimal traffic, analytic)", color=fg)
    ax.set_ylabel("achieved TFLOP/s (measured)", color=fg); ax.tick_params(colors=fg)
    for sp in ax.spines.values(): sp.set_edgecolor(fg)
    ax.grid(True, which="major", color=fg, alpha=.08, lw=.5)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=fg, title="constellations", title_fontsize=8)
    leg.get_title().set_color(fg)
    fig.text(0.07, 0.955, "Roofline: the machine's constellation", fontsize=17, color=fg, weight="bold")
    fig.text(0.07, 0.925, "Every star is one timed kernel at its (intensity, throughput). The horizon is the roofline: the memory roof on the left, the compute roof on the right, "
             "meeting at the ridge. Star size = wall time per call. Lines join a kernel family in parameter order.", fontsize=9, color=fg)
    st = m["stack"]
    fig.text(0.07, 0.03, f"GB10 sm_{st['capability'][0]}{st['capability'][1]} | driver {st['driver_name'].split(',')[0]} | CUDA {st['cuda']} | torch {st['torch']} | cuBLAS {st['cublas']} | "
             f"{st['date']} | idle GPU before: {m['smi_before']['util_temp_power']} (util %, °C, W)", fontsize=7.5, color=fg, alpha=.8)
    fig.savefig(f"{GAL}/roofline_{style}.png", dpi=150, facecolor=bg); plt.close(fig); print("wrote roofline", style)


def plate_table(d):
    rows = sorted(d["stars"], key=lambda s: (s["group"], s.get("n", s.get("B", s.get("T", 0)))))
    with open(f"{CACHE}/stars.md", "w") as f:
        f.write("| group | kernel | FLOPs | bytes | OI (F/B) | ms | TFLOP/s | GB/s | of roof |\n|---|---|---|---|---|---|---|---|---|\n")
        m = d["meta"]; pk, bw = m["gemm_peak_tflops"], m["bw_peak_gbps"]
        for s in rows:
            roof = min(pk, bw * s["oi"] / 1e3) if np.isfinite(s["oi"]) else np.nan
            frac = s["tflops"] / roof if np.isfinite(roof) and roof > 0 else s["gbps"] / bw
            f.write(f"| {s['group']} | {s['label']} | {s['flops']:.3g} | {s['bytes']:.3g} | {s['oi']:.3g} | {s['sec']*1e3:.3f} | {s['tflops']:.2f} | {s['gbps']:.0f} | {frac:.0%} |\n")
    print("wrote stars.md")


if __name__ == "__main__":
    os.makedirs(GAL, exist_ok=True)
    d = load(); plate(d, "night"); plate(d, "paper"); plate_table(d)
