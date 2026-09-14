"""render_stair.py - Staircase plates from cache/stair_*.tsv and cache/hb_*.bin (no GPU, no timing).

    python render_stair.py [--prefix stair] [--only staircase heartbeat]

Measured: ns per dependent load vs working-set size (min and median of reps), per core and page policy;
heartbeat = ns per chunk of 2000 L1-resident loads, sampled back to back. Declared: colours, the log axes,
the annotated cache sizes (from sysfs), and the folding of the heartbeat into a raster.
"""
import argparse, glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"
INK = "#1b1b1b"; PAPER = "#f3eee3"; NIGHT = "#0b0b0e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6})

CORES = {0: ("Cortex-A725", "cluster 0-9", "#005caf"), 5: ("Cortex-X925", "cluster 0-9", "#cb1b45"),
         10: ("Cortex-A725", "cluster 10-19", "#0089a7"), 15: ("Cortex-X925", "cluster 10-19", "#e16b8c")}
SYSFS = {"A725": dict(L1=64 << 10, L2=512 << 10), "X925": dict(L1=64 << 10, L2=2048 << 10)}
L3 = {"cluster 0-9": 8 << 20, "cluster 10-19": 16 << 20}
STACK = "GB10 host CPU: 10× Cortex-X925 + 10× Cortex-A725 (sysfs L1d 64K, L2 2M/512K, L3 8M/16M per cluster) | governor performance | 4K pages, THP madvise | 2026-09-14"


def read(path):
    hdr = open(path).readline().strip()
    a = np.loadtxt(path, skiprows=2)
    return dict(hdr=hdr, bytes=a[:, 0], lines=a[:, 1], mn=a[:, 2], med=a[:, 3], mx=a[:, 4],
                cpu=int(re.search(r"cpu=(-?\d+)", hdr).group(1)), huge=int(re.search(r"huge=(\d)", hdr).group(1)),
                seq=int(re.search(r"seq=(\d)", hdr).group(1)))


def runs(prefix):
    out = {}
    for f in sorted(glob.glob(f"{CACHE}/{prefix}_*.tsv")):
        d = read(f); tag = os.path.basename(f)[len(prefix) + 1:-4]
        out[tag] = d
    return out


def fmt_bytes(b):
    for u, s in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= s:
            v = b / s
            return f"{v:.0f} {u}" if v == int(v) else f"{v:.1f} {u}"
    return f"{b} B"


def plate_staircase(rs, style="paper", which="mn", name="staircase"):
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=bg)
    ax.set_facecolor(bg)
    drawn = []
    for tag, d in rs.items():
        if d["seq"]:
            continue
        core, cl, col = CORES.get(d["cpu"], (f"cpu{d['cpu']}", "", "#888888"))
        ls = "-" if d["huge"] == 0 else "--"
        lab = f"{core} ({cl}), {'THP 2 MB pages' if d['huge'] else '4 KB pages'}"
        y = d[which]
        ax.step(d["bytes"], y, ls, where="mid", color=col if not dark else col, lw=1.4 if d["huge"] == 0 else 1.0, label=lab, alpha=.95)
        if which == "mn":
            ax.fill_between(d["bytes"], d["mn"], d["med"], step="mid", color=col, alpha=.12, lw=0)
        drawn.append(d)
    for tag, d in rs.items():
        if d["seq"]:
            core, cl, col = CORES.get(d["cpu"], (f"cpu{d['cpu']}", "", "#888888"))
            ax.step(d["bytes"], d[which], ":", where="mid", color=col, lw=.9, alpha=.7, label=f"{core} sequential chain (prefetch null)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("working set (bytes, random cycle of 64-byte lines)", color=fg)
    ax.set_ylabel("ns per dependent load" + (" (min of reps; band to median)" if which == "mn" else " (median)"), color=fg)
    xt = [1 << k for k in range(10, 31, 2)]
    ax.set_xticks(xt); ax.set_xticklabels([fmt_bytes(x) for x in xt])
    ax.tick_params(colors=fg)
    for sp in ax.spines.values(): sp.set_edgecolor(fg)
    ax.grid(True, which="major", color=fg, alpha=.12, lw=.5)
    # sysfs cache sizes as vertical rules
    for lab, x, c in [("L1d 64K", 64 << 10, fg), ("L2 A725 512K", 512 << 10, CORES[0][2]), ("L2 X925 2M", 2 << 20, CORES[5][2]),
                      ("L3 8M (cl 0-9)", 8 << 20, CORES[0][2]), ("L3 16M (cl 10-19)", 16 << 20, CORES[10][2])]:
        ax.axvline(x, color=c, lw=.7, ls=(0, (2, 3)), alpha=.7)
        ax.text(x, ax.get_ylim()[1] * .92, " " + lab, rotation=90, va="top", ha="left", fontsize=7.5, color=c)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=fg)
    fig.text(0.06, 0.955, "Staircase: random-access latency versus working set, one core at a time", fontsize=15, color=fg, weight="bold")
    fig.text(0.06, 0.925, "Each step is a cache level that stopped fitting. Dashed rules are the sizes sysfs reports; where the steps actually land is the measurement.", fontsize=9.5, color=fg)
    fig.text(0.06, 0.02, STACK + " | chase.c: Sattolo cycle, unrolled dependent loads, 5 reps", fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    fig.savefig(f"{GAL}/{name}_{style}.png", dpi=150, facecolor=bg); plt.close(fig); print("wrote", name, style)


def plate_plotter(rs, name="staircase_plotter"):
    """Single-ink plotter sheet: min latency only, one line per run, labels at the right end."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=PAPER); ax.set_facecolor(PAPER)
    for tag, d in rs.items():
        core, cl, col = CORES.get(d["cpu"], (f"cpu{d['cpu']}", "", "#888"))
        ls = ":" if d["seq"] else ("--" if d["huge"] else "-")
        ax.plot(d["bytes"], d["mn"], ls, color=INK, lw=.9, drawstyle="steps-mid")
        short = f"{core.split('-')[-1]} {cl.split()[-1] if cl else ''}".strip()
        ax.text(d["bytes"][-1] * 1.08, d["mn"][-1], f"{short}{' THP' if d['huge'] else ''}{' seq' if d['seq'] else ''}", fontsize=6.5, va="center", color=INK)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    xt = [1 << k for k in range(10, 31, 2)]; ax.set_xticks(xt); ax.set_xticklabels([fmt_bytes(x) for x in xt])
    ax.set_xlabel("working set"); ax.set_ylabel("ns per dependent load (min)")
    for x in (64 << 10, 512 << 10, 2 << 20, 8 << 20, 16 << 20):
        ax.axvline(x, color=INK, lw=.5, ls=(0, (1, 4)))
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 2.2)
    for sp in ax.spines.values(): sp.set_edgecolor(INK)
    fig.text(0.06, 0.95, "Staircase (plotter sheet)", fontsize=14, color=INK, weight="bold")
    fig.text(0.06, 0.02, STACK, fontsize=7.5, color=INK, alpha=.8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(f"{GAL}/{name}.png", dpi=150, facecolor=PAPER); plt.close(fig); print("wrote", name)


def plate_heartbeat(path, name, style="night", row_ms=1.0):
    """Fold the back-to-back chunk timings into a raster: one row per `row_ms` of wall time, colour = ns/load."""
    a = np.fromfile(path, dtype=np.float64).reshape(-1, 2)
    t, dur = a[:, 0] / 1e6, a[:, 1]           # ms, ns
    chunk = 2000
    nsl = dur / chunk
    rows = int(t[-1] // row_ms) + 1
    cols = int(np.ceil(len(t) / rows * 1.05))
    img = np.full((rows, cols), np.nan)
    r = (t // row_ms).astype(int)
    # position within row by time fraction
    c = ((t % row_ms) / row_ms * cols).astype(int).clip(0, cols - 1)
    img[r, c] = nsl
    base = np.nanmedian(nsl)
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig = plt.figure(figsize=(16, 9), facecolor=bg)
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.60]); ax2 = fig.add_axes([0.06, 0.08, 0.90, 0.17])
    cm = plt.get_cmap("cet_fire" if dark else "cmc.lajolla").copy(); cm.set_bad(bg)
    im = ax.imshow(np.log2(img / base), aspect="auto", cmap=cm, vmin=0, vmax=4, interpolation="nearest",
                   extent=[0, row_ms, rows * row_ms, 0])
    ax.set_ylabel(f"wall time (ms), one row per {row_ms:g} ms", color=fg); ax.set_xlabel("position within the row (ms)", color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values(): sp.set_edgecolor(fg)
    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01); cb.set_label("log2(ns per load / median)", color=fg); cb.ax.tick_params(colors=fg)
    ax2.plot(t, nsl, color=fg, lw=.3); ax2.set_yscale("log"); ax2.set_xlim(0, t[-1])
    ax2.set_xlabel("wall time (ms)", color=fg); ax2.set_ylabel("ns/load", color=fg); ax2.tick_params(colors=fg); ax2.set_facecolor(bg)
    for sp in ax2.spines.values(): sp.set_edgecolor(fg)
    q = np.percentile(nsl, [50, 99, 99.9, 100])
    fig.text(0.06, 0.955, f"Heartbeat: {len(t):,} back-to-back chunks of {chunk} dependent loads, {os.path.basename(path)}", fontsize=14, color=fg, weight="bold")
    fig.text(0.06, 0.925, f"median {q[0]:.2f} ns/load · p99 {q[1]:.2f} · p99.9 {q[2]:.2f} · max {q[3]:.1f}. Bright pixels = chunks slowed by interrupts, timer ticks or migrations; "
             f"periodic texture = the OS's own rhythm.", fontsize=9, color=fg)
    fig.text(0.06, 0.01, STACK, fontsize=7.5, color=fg, alpha=.8)
    fig.savefig(f"{GAL}/{name}_{style}.png", dpi=150, facecolor=bg); plt.close(fig); print("wrote", name, style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="stair")
    ap.add_argument("--hb", default="hb")
    ap.add_argument("--only", nargs="*", default=["staircase", "heartbeat"])
    a = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    if "staircase" in a.only:
        rs = runs(a.prefix)
        if rs:
            plate_staircase(rs, "paper"); plate_staircase(rs, "night"); plate_staircase(rs, "paper", "med", "staircase_median"); plate_plotter(rs)
    if "heartbeat" in a.only:
        for f in sorted(glob.glob(f"{CACHE}/{a.hb}_*.bin")):
            nm = "heartbeat_" + os.path.basename(f)[len(a.hb) + 1:-4]
            plate_heartbeat(f, nm, "night"); plate_heartbeat(f, nm, "paper")


if __name__ == "__main__":
    main()
