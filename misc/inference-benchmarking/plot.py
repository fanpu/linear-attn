"""Figures and summary tables for the GB10 inference survey.

Palette follows the data-viz method and was validated with its checker rather than
chosen by eye:

* The five dense models are an ordered magnitude (a size ladder), so they get an
  *ordinal* single-hue ramp, not categorical hues. The five steps pass the light
  ordinal gates (monotone lightness, adjacent dL >= 0.06, light end >= 2:1 on the
  surface). Six steps do not fit between the light-end floor and the dark end,
  which is a further reason the MoE is encoded separately.
* Qwen3-30B-A3B is sparse: it differs in *kind*, not degree, so it takes its own
  categorical hue (orange), which clears all-pairs CVD separation against the ramp
  (worst dE 24.7 protan).
* Predictions reuse their model's hue as a dashed line. A prediction is the same
  entity as its measurement, so it must not consume a new hue.

Run with the venv python, which has matplotlib:
    /home/fzeng/ml/research/.venv/bin/python plot.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/results.jsonl")
ROOFLINE = Path("results/roofline.json")
FIGDIR = Path("results")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

# Ordinal ramp for the dense size ladder (validated, light mode).
DENSE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
MOE_HUE = "#eb6834"

DENSE_ORDER = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]
MOE = "Qwen/Qwen3-30B-A3B"

# Ordinal ramp for context lengths within a single model (also an ordered scale).
CTX_RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]


def short(model: str) -> str:
    return model.split("/")[-1].replace("Qwen3-", "")


def color_for(model: str) -> str:
    if model == MOE:
        return MOE_HUE
    return DENSE_RAMP[DENSE_ORDER.index(model)] if model in DENSE_ORDER else INK2


def style_axes(ax, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left",
                 pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9.5,
                va="bottom")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def load_rows() -> list:
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def ok_rows(rows):
    return [r for r in rows if str(r.get("status", "")).startswith("ok")
            and r.get("decode_tok_s")]


def fig_decode_vs_batch(rows, in_len=128, out="fig_decode_vs_batch.png"):
    """Decode throughput against batch, for every model, at short context."""
    data = defaultdict(list)
    for r in ok_rows(rows):
        if r["in_len"] == in_len:
            data[r["model"]].append((r["batch"], r["decode_tok_s"]))
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    order = [m for m in DENSE_ORDER + [MOE] if m in data]
    for model in order:
        pts = sorted(data[model])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                color=color_for(model), label=short(model), zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "batch size (concurrent sequences)", "decode throughput (tok/s)",
               "Decode throughput scales with batch until bandwidth runs out",
               f"NVIDIA GB10 · vLLM 0.23 · BF16 · {in_len}-token prompts")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left",
              ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return FIGDIR / out


def fig_measured_vs_predicted(rows, roof, in_len=128,
                              out="fig_measured_vs_predicted.png"):
    """Measured decode throughput against the roofline prediction."""
    import budget
    import perf_model as pm

    data = defaultdict(list)
    for r in ok_rows(rows):
        if r["in_len"] == in_len:
            data[r["model"]].append((r["batch"], r["decode_tok_s"]))
    if not data or roof is None:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    for model in [m for m in DENSE_ORDER + [MOE] if m in data]:
        try:
            spec = budget.ModelSpec.from_hf_cache(model)
        except Exception:
            continue
        pts = sorted(data[model])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = color_for(model)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=c,
                label=short(model), zorder=3, markeredgecolor=SURFACE,
                markeredgewidth=1.2)
        act = pm.active_weight_bytes(spec.config, spec.weight_bytes)
        preds = [pm.predict_decode(spec.config, spec.weight_bytes, b,
                                   in_len + 128, roof,
                                   active_weight_bytes=act).tok_s for b in xs]
        ax.plot(xs, preds, linestyle="--", linewidth=1.5, color=c, alpha=0.75,
                zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "batch size", "decode throughput (tok/s)",
               "Measured throughput against the bandwidth-roofline prediction",
               f"solid = measured · dashed = predicted from {roof.bw_gbps:.0f} GB/s "
               f"and {roof.tflops:.0f} TFLOP/s")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return FIGDIR / out


def fig_context_effect(rows, model="Qwen/Qwen3-8B", out="fig_context_effect.png"):
    """How context length erodes the benefit of batching, for one model."""
    data = defaultdict(list)
    for r in ok_rows(rows):
        if r["model"] == model:
            data[r["in_len"]].append((r["batch"], r["decode_tok_s"]))
    if len(data) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    for i, in_len in enumerate(sorted(data)):
        pts = sorted(data[in_len])
        c = CTX_RAMP[min(i, len(CTX_RAMP) - 1)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=5,
                linewidth=2, color=c, label=f"{in_len} tokens", zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "batch size", "decode throughput (tok/s)",
               "Long context cancels the batching win",
               f"{short(model)} · each sequence re-reads its own KV cache every step")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, title="prompt length",
              title_fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return FIGDIR / out


def fig_latency(rows, in_len=128, out="fig_latency.png"):
    """Per-token latency against batch: the throughput/latency trade."""
    data = defaultdict(list)
    for r in ok_rows(rows):
        if r["in_len"] == in_len and r.get("tpot_ms"):
            data[r["model"]].append((r["batch"], r["tpot_ms"]))
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    for model in [m for m in DENSE_ORDER + [MOE] if m in data]:
        pts = sorted(data[model])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=5,
                linewidth=2, color=color_for(model), label=short(model), zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xscale("log", base=2)
    style_axes(ax, "batch size", "time per output token (ms)",
               "What each user feels as batch grows",
               f"{in_len}-token prompts · lower is a faster-feeling stream")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return FIGDIR / out


def fig_moe_experts(rows, out="fig_moe_experts.png"):
    """How many experts the MoE effectively reads, as batch grows.

    Emphasis form: one measured series in the MoE hue against a reference curve in
    neutral gray. The reference is not a second entity competing for identity, so
    it must not take a categorical hue.
    """
    import budget
    import perf_model as pm

    pts = [(r["batch"], r["decode_tok_s"], r["in_len"] + r["out_len"])
           for r in ok_rows(rows) if r["model"] == MOE and r["in_len"] == 128]
    if len(pts) < 4:
        return None
    try:
        spec = budget.ModelSpec.from_hf_cache(MOE)
    except Exception:
        return None
    roof = pm.Roofline.from_json(str(ROOFLINE))
    n_exp = int(spec.config["num_experts"])

    pts.sort()
    xs = [p[0] for p in pts]
    measured = [pm.experts_touched(t, b, spec.config, spec.weight_bytes, roof, ctx=c)
                for b, t, c in pts]
    theory = [pm.experts_touched_if_random(b, spec.config) for b in xs]

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    ax.plot(xs, theory, linestyle="--", linewidth=1.8, color="#8d8c86",
            label="if routing were random", zorder=2)
    ax.plot(xs, measured, marker="o", markersize=6, linewidth=2.2, color=MOE_HUE,
            label="measured", zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(n_exp, color=GRID, linewidth=1.2, zorder=1)
    ax.annotate(f"all {n_exp} experts", (xs[0], n_exp), textcoords="offset points",
                xytext=(2, 5), color=INK2, fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_ylim(0, n_exp * 1.15)
    style_axes(ax, "batch size", "experts read per layer, per step",
               "Batching erodes an MoE's sparsity",
               "Qwen3-30B-A3B · derived from measured throughput, not assumed")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return FIGDIR / out


def summary_table(rows) -> str:
    """Markdown table: best decode throughput and single-stream latency per model."""
    best, single = {}, {}
    for r in ok_rows(rows):
        m = r["model"]
        if r["decode_tok_s"] > best.get(m, (0,))[0]:
            best[m] = (r["decode_tok_s"], r["batch"], r["in_len"])
        if r["batch"] == 1 and r["in_len"] == 128:
            single[m] = r
    if not best:
        return "_no results yet_"

    lines = [
        "| model | single-stream tok/s | best tok/s | at batch | TTFT @1 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in [x for x in DENSE_ORDER + [MOE] if x in best]:
        b = best[m]
        s = single.get(m)
        lines.append(
            f"| {short(m)} | {s['decode_tok_s']:.1f} | {b[0]:.0f} | {b[1]} | "
            f"{s['ttft_ms']:.0f} |" if s else
            f"| {short(m)} | - | {b[0]:.0f} | {b[1]} | - |"
        )
    return "\n".join(lines)


def skipped_table(rows) -> str:
    sk = [r for r in rows if r.get("status") == "skipped:budget"]
    if not sk:
        return "_none — every planned cell fit under the ceiling_"
    lines = ["| model | batch | prompt | predicted total | reason |", "|---|---:|---:|---:|---|"]
    for r in sk:
        lines.append(f"| {short(r['model'])} | {r['batch']} | {r['in_len']} | "
                     f"{r['predicted_total_gib']:.0f} GiB | over ceiling |")
    return "\n".join(lines)


def main():
    rows = load_rows()
    roof = None
    if ROOFLINE.exists():
        import perf_model as pm
        roof = pm.Roofline.from_json(str(ROOFLINE))

    made = []
    for fn, args in [
        (fig_decode_vs_batch, (rows,)),
        (fig_measured_vs_predicted, (rows, roof)),
        (fig_context_effect, (rows,)),
        (fig_latency, (rows,)),
        (fig_moe_experts, (rows,)),
    ]:
        try:
            p = fn(*args)
            if p:
                made.append(p)
        except Exception as e:
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")

    print(f"{len(ok_rows(rows))} usable rows; wrote {len(made)} figures")
    for p in made:
        print(f"  {p}")
    print("\n## Summary\n")
    print(summary_table(rows))
    print("\n## Skipped cells\n")
    print(skipped_table(rows))


if __name__ == "__main__":
    main()
