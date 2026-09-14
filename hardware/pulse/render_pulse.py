"""render_pulse.py - Pulse plates from cache/<tag>_sensors.csv (nvidia-smi), <tag>_host.csv, <tag>_phases.json.

    python render_pulse.py --tag phases [--tag soak] [--burst /path/to/misc/burst/results/<run>]

Measured: power (instant and NVML average), temperature, SM clock, utilisation, throttle reasons, host
sensors, per phase. Declared: the stacked-strip composition, colours, the phase shading, the folded raster.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from datetime import datetime
sys.path.insert(0, "/home/fzeng/ml/research/art/color-research")
import palettes as P

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, GAL = f"{HERE}/cache", f"{HERE}/gallery"
INK = "#1b1b1b"; PAPER = "#f3eee3"; NIGHT = "#0b0b0e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": .6})
STACK = "NVIDIA GB10 | driver 580.173.02 | CUDA 13.0 | torch 2.14.0+cu130 | nvidia-smi --loop-ms=100 (NVML) | 2026-09-14"
PHASE_COL = {"idle": None, "matmul_burst": "#cb1b45", "matmul_soak": "#cb1b45", "copy_2GiB": "#005caf", "prefill_B64x256": "#1b813e",
             "decode_B64": "#ffb11b", "decode_B1": "#ca7a2c", "square_2s": "#592c63", "square_0.5s": "#86a697"}


def load_sensors(tag):
    rows = list(csv.reader(open(f"{CACHE}/{tag}_sensors.csv")))
    hdr = [h.strip() for h in rows[0]]
    t, data = [], {h: [] for h in hdr[1:]}
    for r in rows[1:]:
        if len(r) != len(hdr):
            continue
        try:
            ts = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        t.append(ts)
        for h, v in zip(hdr[1:], r[1:]):
            v = v.strip()
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(v)
    d = {"t": np.array(t)}
    for h, v in data.items():
        try:
            d[h] = np.array(v, float)
        except (ValueError, TypeError):
            d[h] = np.array(v)
    return d


def load_host(tag):
    rows = list(csv.reader(open(f"{CACHE}/{tag}_host.csv")))
    hdr = rows[0]; a = {h: [] for h in hdr}
    for r in rows[1:]:
        if len(r) != len(hdr):
            continue
        for h, v in zip(hdr, r):
            try:
                a[h].append(float(v))
            except ValueError:
                a[h].append(np.nan)
    return {h: np.array(v) for h, v in a.items()}


def col(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def plate_phases(tag, style="night"):
    d = load_sensors(tag); h = load_host(tag)
    ph = json.load(open(f"{CACHE}/{tag}_phases.json"))
    t0 = ph[0]["t0"]; t = d["t"] - t0
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    rows = [("power (W)", [("power.draw.instant", "instant", "#ff8c42"), ("power.draw.average", "NVML average", "#ffd166")]),
            ("temperature (°C)", [("temperature.gpu", "GPU", "#ef476f")]),
            ("SM clock (MHz)", [("clocks.sm", "SM", "#06d6a0"), ("clocks.mem", "mem", "#4cc9f0")]),
            ("utilisation (%)", [("utilization.gpu", "GPU", "#8ecae6"), ("utilization.memory", "memory", "#219ebc")])]
    fig, axs = plt.subplots(len(rows) + 1, 1, figsize=(20, 12), facecolor=bg, sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1, 1, 1, 0.8]))
    for ax, (ylabel, series) in zip(axs, rows):
        for key, lab, c in series:
            y = col(d, key)
            if y is None or y.dtype.kind not in "fi":
                continue
            ax.plot(t, y, lw=.8, color=c if dark else P.rgb2hex(P.hex2rgb(c) * .75), label=lab)
        ax.set_ylabel(ylabel, color=fg); ax.tick_params(colors=fg); ax.set_facecolor(bg)
        ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=fg)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
    # host: ACPI temps and CPU MHz
    ax = axs[-1]
    for k in [k for k in h if k.startswith("acpitz")][:3]:
        ax.plot(h["t"] - t0, h[k] / 1000, lw=.8, label=k)
    ax.set_ylabel("host °C", color=fg); ax.tick_params(colors=fg); ax.set_facecolor(bg); ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=fg)
    for sp in ax.spines.values(): sp.set_edgecolor(fg)
    for ax in axs:
        for p in ph:
            c = PHASE_COL.get(p["name"], "#888888")
            if c:
                ax.axvspan(p["t0"] - t0, p["t1"] - t0, color=c, alpha=.10, lw=0)
    for p in ph:
        if p["name"] != "idle":
            axs[0].text((p["t0"] + p["t1"]) / 2 - t0, axs[0].get_ylim()[1], p["name"], ha="center", va="top", fontsize=7.5, color=fg, rotation=90)
    axs[-1].set_xlabel("seconds", color=fg); axs[-1].set_xlim(0, t[-1])
    fig.text(0.02, 0.975, "Pulse: the GB10's sensors through a scripted day of work", fontsize=15, color=fg, weight="bold", va="top")
    fig.text(0.02, 0.952, "Shaded: phases (matmul burst, 2 GiB copies, Qwen3-0.6B prefill B=64, decode B=64 and B=1, square waves of matmul). "
             "The power sensor updates about twice a second, so the square waves test what the sensor can follow.", fontsize=9, color=fg, va="top")
    fig.text(0.02, 0.006, STACK, fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.015, 1, 0.94])
    fig.savefig(f"{GAL}/pulse_{tag}_{style}.png", dpi=140, facecolor=bg); plt.close(fig); print("wrote pulse", tag, style)
    return d, ph


def plate_square(tag, style="night"):
    """Zoom on the square-wave phases: does the sensor follow 1 s on / 1 s off? 0.25 s?"""
    d = load_sensors(tag); ph = json.load(open(f"{CACHE}/{tag}_phases.json")); t0 = ph[0]["t0"]
    sq = [p for p in ph if p["name"].startswith("square")]
    if not sq:
        return
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    fig, axs = plt.subplots(len(sq), 1, figsize=(18, 3.6 * len(sq) + 1), facecolor=bg)
    for ax, p in zip(np.atleast_1d(axs), sq):
        m = (d["t"] >= p["t0"] - 3) & (d["t"] <= p["t0"] + 25)
        tt = d["t"][m] - p["t0"]
        ax.step(tt, d["power.draw.instant"][m], where="post", color="#ff8c42", lw=1.2, label="power.draw.instant")
        ax.step(tt, d["power.draw.average"][m], where="post", color="#ffd166", lw=.9, label="power.draw.average")
        ax2 = ax.twinx(); ax2.step(tt, d["utilization.gpu"][m], where="post", color="#8ecae6", lw=.7, label="util %"); ax2.set_ylim(0, 105); ax2.tick_params(colors=fg)
        per = p["period_s"]
        for k in range(int(25 / per) + 1):
            ax.axvspan(k * per, k * per + per / 2, color=fg, alpha=.06, lw=0)
        ax.set_ylabel("W", color=fg); ax.tick_params(colors=fg); ax.set_facecolor(bg); ax.set_xlim(-3, 25)
        ax.text(0.01, 0.9, f"{p['name']}: matmul {per/2:g} s on / {per/2:g} s off (grey = on)", transform=ax.transAxes, color=fg, fontsize=10, weight="bold")
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=fg)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
    np.atleast_1d(axs)[-1].set_xlabel("seconds from phase start", color=fg)
    fig.text(0.02, 0.975, "Pulse: what the power sensor can follow", fontsize=14, color=fg, weight="bold", va="top")
    fig.text(0.02, 0.006, STACK, fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.015, 1, 0.95])
    fig.savefig(f"{GAL}/pulse_square_{tag}_{style}.png", dpi=140, facecolor=bg); plt.close(fig); print("wrote square", tag, style)


def plate_soak(tag, style="night"):
    """Sustained load: temperature rise and cool-down with exponential fits (thermal time constants)."""
    d = load_sensors(tag); ph = json.load(open(f"{CACHE}/{tag}_phases.json")); t0 = ph[0]["t0"]
    soak = [p for p in ph if p["name"] == "matmul_soak"][0]
    dark = style == "night"; fg = "#e8e4d8" if dark else INK; bg = NIGHT if dark else PAPER
    t = d["t"] - t0; T = d["temperature.gpu"]; Pw = d["power.draw.instant"]; clk = d["clocks.sm"]
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), facecolor=bg, sharex=True)
    axs[0].plot(t, T, color="#ef476f", lw=1); axs[0].set_ylabel("GPU °C", color=fg)
    axs[1].plot(t, Pw, color="#ff8c42", lw=.8); axs[1].set_ylabel("W", color=fg)
    axs[2].plot(t, clk, color="#06d6a0", lw=.8); axs[2].set_ylabel("SM MHz", color=fg)
    fits = {}
    from scipy.optimize import curve_fit
    for name, a, b in (("heating", soak["t0"] - t0, soak["t1"] - t0), ("cooling", soak["t1"] - t0, t[-1])):
        m = (t >= a) & (t <= b); x = t[m] - a; y = T[m]
        try:
            f = lambda x, y0, dy, tau: y0 + dy * (1 - np.exp(-x / tau))
            p, _ = curve_fit(f, x, y, p0=[y[0], y[-1] - y[0], 60], maxfev=20000)
            fits[name] = dict(T0=float(p[0]), dT=float(p[1]), tau_s=float(p[2]))
            axs[0].plot(x + a, f(x, *p), "--", color=fg, lw=.8)
            axs[0].text(a + 5, y.max() if name == "heating" else y.min() + 2, f"{name}: τ = {p[2]:.0f} s, ΔT = {p[1]:+.1f} °C", color=fg, fontsize=8)
        except Exception as e:
            print("fit failed", name, e)
    for ax in axs:
        ax.axvspan(soak["t0"] - t0, soak["t1"] - t0, color="#cb1b45", alpha=.08, lw=0); ax.tick_params(colors=fg); ax.set_facecolor(bg)
        for sp in ax.spines.values(): sp.set_edgecolor(fg)
    axs[-1].set_xlabel("seconds", color=fg)
    fig.text(0.02, 0.975, f"Pulse: thermal soak, {soak['t1'] - soak['t0']:.0f} s of 8192² bf16 matmul then idle", fontsize=14, color=fg, weight="bold", va="top")
    fig.text(0.02, 0.006, STACK, fontsize=7.5, color=fg, alpha=.8)
    fig.tight_layout(rect=[0, 0.015, 1, 0.95])
    fig.savefig(f"{GAL}/pulse_soak_{tag}_{style}.png", dpi=140, facecolor=bg); plt.close(fig); print("wrote soak", tag, style, fits)
    json.dump(fits, open(f"{CACHE}/{tag}_fits.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="*", default=["phases"])
    a = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    for tag in a.tag:
        if not os.path.exists(f"{CACHE}/{tag}_sensors.csv"):
            continue
        ph = json.load(open(f"{CACHE}/{tag}_phases.json"))
        if any(p["name"] == "matmul_soak" for p in ph):
            plate_soak(tag, "night"); plate_soak(tag, "paper")
        else:
            plate_phases(tag, "night"); plate_phases(tag, "paper"); plate_square(tag, "night"); plate_square(tag, "paper")


if __name__ == "__main__":
    main()
