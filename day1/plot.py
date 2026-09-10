"""Plot kernel-correctness logs across precision modes.

Usage:
  python plot.py --log tf32=log_tf32.txt --log ieee=log_ieee.txt --out kernel_check
  python plot.py --csv results.csv --out kernel_check     # replot without the raw logs
Each --log is TAG=PATH. float32 rows get mode "fp32-TAG"; bfloat16 rows get mode "bf16".
Header lines: `T= 256 Dk= 64 Dv= 64 dtype=float32 [alpha=0.999 beta=0.05 ...]`
Any extra key=value in a header becomes a column (for later alpha/beta sweeps).
Writes OUT_summary.png, OUT_detail.png, OUT.csv, OUT_tables.md.
"""
import argparse, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HDR = re.compile(r"^T=\s*(\d+)\s+Dk=\s*(\d+)\s+Dv=\s*(\d+)\s+dtype=(\w+)(.*)$")
ROW = re.compile(r"^\s*(ok|FAIL|fail)\s+(.+?)\s+max_rel=([\d.eE+-]+)\s+cos=([\d.eE+-]+)")
KV = re.compile(r"(\w+)=([\w.+-]+)")

U = {"fp32": 2.0**-24, "tf32": 2.0**-11, "bf16": 2.0**-8}      # unit roundoff
MODES = ["fp32-ieee", "fp32-tf32", "bf16"]                      # display order
U_EFF = {"fp32-ieee": U["fp32"], "fp32-tf32": U["tf32"], "bf16": U["bf16"]}
U_NAME = {"fp32-ieee": "fp32", "fp32-tf32": "tf32", "bf16": "bf16"}
COLOR = {"fp32-ieee": "#4C72B0", "fp32-tf32": "#DD8452", "bf16": "#55A868"}
TENSORS = ["fwd out", "fwd state", "dq", "dk", "dv", "dg", "dbeta"]


def parse(path, tag):
    rows, cur = [], None
    for line in open(path):
        if m := HDR.match(line.rstrip()):
            cur = dict(T=int(m[1]), Dk=int(m[2]), Dv=int(m[3]), dtype=m[4], **dict(KV.findall(m[5])))
        elif (m := ROW.match(line)) and cur:
            mode = "bf16" if cur["dtype"] == "bfloat16" else f"fp32-{tag}"
            rows.append({**cur, "mode": mode, "status": m[1], "tensor": m[2].strip(),
                         "max_rel": float(m[3]), "one_minus_cos": float(m[4])})
    return rows


def ref_lines(ax, x_text):
    for name, ls in [("bf16", "--"), ("tf32", ":"), ("fp32", "-.")]:
        ax.axhline(U[name], color="grey", ls=ls, lw=1)
        ax.text(x_text, U[name], f" $u_{{\\mathrm{{{name}}}}}$", va="center", fontsize=9, color="grey")


def summary_fig(df, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rng = np.random.default_rng(0)
    modes = [m for m in MODES if m in df["mode"].unique()]
    for k, mode in enumerate(modes):
        for i, t in enumerate(TENSORS):
            y = df[(df["mode"] == mode) & (df.tensor == t)].max_rel.values
            x = i + (k - 1) * 0.25 + rng.uniform(-0.07, 0.07, len(y))
            ax.scatter(x, y, s=18, color=COLOR[mode], alpha=0.8, label=mode if i == 0 else None)
    ref_lines(ax, len(TENSORS) - 0.45)
    ax.set_yscale("log"); ax.set_ylim(3e-8, 3e-2)
    ax.set_xticks(range(len(TENSORS))); ax.set_xticklabels(TENSORS)
    ax.set_ylabel("max_rel vs naive recurrent reference")
    ax.set_title("fla chunked GDN on GB10: TF32 inflates fp32 error ~15×;\n"
                 "true-fp32 error still sits ~2000 $u_{\\mathrm{fp32}}$ above the fp32 floor",
                 fontsize=12, fontweight="bold")
    ax.text(0.01, 0.02, "each dot = one (T, $d_k$, $d_v$) config; T ∈ {256, 1024, 4096}, "
            "$(d_k,d_v)$ ∈ {64×64, 64×128, 128×128}", transform=ax.transAxes, fontsize=8, color="grey")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.3), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out}_summary.png", dpi=150, bbox_inches="tight"); plt.close()


def detail_fig(df, out):
    modes = [m for m in MODES if m in df["mode"].unique()]
    dims = sorted(df.dims.unique(), key=lambda s: tuple(map(int, s.split("×"))))
    dcol = dict(zip(dims, ["#4C72B0", "#DD8452", "#55A868"]))
    dmark = dict(zip(dims, ["o", "s", "^"]))
    Ts = sorted(df["T"].unique())
    fig, axes = plt.subplots(2, len(modes), figsize=(6.2 * len(modes), 9.5),
                             gridspec_kw=dict(height_ratios=[1, 1.15]))
    for j, mode in enumerate(modes):
        ax = axes[0, j]; sub = df[df["mode"] == mode]
        for d in dims:
            g = sub[sub.dims == d].groupby("T").max_rel.agg(["min", "max"]).sort_index()
            ax.fill_between(g.index, g["min"], g["max"], color=dcol[d], alpha=0.12)
            ax.plot(g.index, g["max"], marker=dmark[d], color=dcol[d], lw=2, label=f"$d_k\\times d_v$ = {d}")
        ref_lines(ax, Ts[-1] * 1.12)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(3e-8, 3e-2)
        ax.set_xticks(Ts); ax.set_xticklabels(Ts)
        ax.set_xlabel("sequence length $T$")
        if j == 0:
            ax.set_ylabel("max_rel (line = worst of 7 tensors,\nband = range across tensors)")
            ax.legend(loc="lower left", fontsize=8)
        ax.set_title(mode, fontsize=12, fontweight="bold", color=COLOR[mode])
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, j]
        sub = sub.assign(cfg="T=" + sub["T"].astype(str) + "\n" + sub.dims)
        cols = [f"T={t}\n{d}" for t in Ts for d in dims]
        piv = sub.pivot(index="tensor", columns="cfg", values="max_rel").loc[TENSORS, cols] / U_EFF[mode]
        im = ax.imshow(piv.values, cmap="YlOrRd", vmin=0, aspect="auto")
        fmt = "{:.0f}" if piv.values.max() > 100 else "{:.1f}"
        for (r, c), v in np.ndenumerate(piv.values):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6.5)
        ax.set_yticks(range(len(TENSORS))); ax.set_yticklabels(TENSORS)
        ax.set_title(f"max_rel in units of $u_{{\\mathrm{{{U_NAME[mode]}}}}}$", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle("Kernel-vs-reference error by precision mode, sequence length, and head shape",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{out}_detail.png", dpi=150, bbox_inches="tight"); plt.close()


def md_table(piv, fmt):
    idx = list(piv.index.names)
    lines = ["| " + " | ".join(idx + list(piv.columns)) + " |",
             "|" + "---:|" * (len(idx) + len(piv.columns))]
    for key, r in piv.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        lines.append("| " + " | ".join(map(str, key)) + " | " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def tables(df, out):
    def piv(mode, col):
        s = df[df["mode"] == mode]
        return s.pivot_table(index=["T", "Dk", "Dv"], columns="tensor", values=col, sort=False)[TENSORS]
    md = []
    if "fp32-ieee" in df["mode"].unique():
        md += ["### max_rel, fp32-ieee, units of 1e-4", md_table(piv("fp32-ieee", "max_rel") / 1e-4, "{:.3f}".format)]
        md += ["### 1-cos, fp32-ieee, in ulps of 2^-24",
               md_table((piv("fp32-ieee", "one_minus_cos") / U["fp32"]).round(), lambda v: f"{v:.0f}")]
    if {"fp32-ieee", "fp32-tf32"} <= set(df["mode"]):
        md += ["### ratio max_rel(TF32) / max_rel(ieee)",
               md_table(piv("fp32-tf32", "max_rel") / piv("fp32-ieee", "max_rel"), "{:.1f}".format)]
    open(f"{out}_tables.md", "w").write("\n\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", help="TAG=PATH, repeatable")
    ap.add_argument("--csv", help="replot from a CSV written by an earlier run")
    ap.add_argument("--out", default="kernel_check")
    a = ap.parse_args()
    if not (a.log or a.csv):
        ap.error("need --log TAG=PATH (raw kernel_check output) or --csv PATH (earlier run)")
    if a.csv:
        df = pd.read_csv(a.csv)
    else:
        rows = []
        for spec in a.log:
            tag, path = spec.split("=", 1); rows += parse(path, tag)
        df = pd.DataFrame(rows)
    df["dims"] = df.Dk.astype(str) + "×" + df.Dv.astype(str)
    if f"{a.out}.csv" != a.csv:
        df.to_csv(f"{a.out}.csv", index=False)
    summary_fig(df, a.out); detail_fig(df, a.out); tables(df, a.out)
    print(df.groupby("mode").max_rel.agg(["count", "median", "min", "max"]))
