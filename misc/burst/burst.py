#!/usr/bin/env python3
"""
burst.py

Question: is the GB10 matmul ceiling (93.9 TFLOP/s, day2 roofline, 2026-09-10, sub-second and cold)
a sustained number, or a burst that decays once the chassis heat-soaks?

One run = cold start -> long matmul soak -> idle cooldown, with throughput and device telemetry on
one clock:

  cool_wait   idle until GPU temperature is flat, so the first window really is a cold chip
  copy_cold   2 GiB device copy, beta before the soak
  load        8192^2 bf16 matmul, synchronized every ~0.1 s (the power cap acts in ms; 5 s windows hide it)
  copy_hot    the same copy, right after the soak: does beta hold while pi drops?
  cooldown    idle; the cooling curve gives the thermal time constant with no throttling in the way

Written to results/<YYYYmmdd-HHMMSS>/:

  samples.csv     one row per synchronized chunk of matmuls: end time, count, wall time, TFLOP/s
  telemetry.csv   nvidia-smi every 250 ms (temp, T.Limit headroom, power, SM clock, clock-event
                  reasons + their microsecond counters) and host ACPI temps, same perf_counter clock
  meta.json       config, preconditions, phase boundaries, beta cold/hot
  summary.json    burst, plateau, fits, thermal R/C, shape verdict, ledger line
  timeseries.png  throughput / temperature / power / SM clock / clock-event reasons vs time
  mechanism.png   throughput vs temperature and vs power: which one does the drop track?

Every row is flushed (and fsync'd every 2 s) as it is written, so a crash or reboot mid-soak still
leaves a plottable run. Ctrl-C ends the current phase early and still analyses what was recorded.

Usage
  python burst.py run                                   # 600 s soak, ~15 min end to end
  python burst.py run --predict 88                      # record your prediction in summary.json
  python burst.py run --load-s 20 --cooldown-s 10 --cool-wait-max-s 0 --copy-s 1   # smoke test
  python burst.py plot results/<run>                    # re-analyse + re-plot a saved run
  python burst.py compare results/<a> results/<b> --labels upright flat --out compare.png
"""
import argparse, csv, glob, json, os, subprocess, sys, threading, time
from collections import deque
from contextlib import contextmanager
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_BURST_TFLOPS = 93.9  # day2/roofline.json as of 14d3ee6 (2026-09-10): cold, sub-second
REF_COPY_GBPS = 246.8  # same file

# nvidia-smi field -> column name. Fields the driver rejects are dropped at startup.
SMI_FIELDS = {
    "temperature.gpu": "gpu_temp_C",
    "temperature.gpu.tlimit": "tlimit_headroom_C",  # degrees below the slowdown threshold
    "power.draw.instant": "power_W",
    "power.draw.average": "power_avg_W",  # ~1 s average
    "clocks.sm": "sm_clock_MHz",
    "utilization.gpu": "util_pct",
    "clocks_event_reasons.active": "reasons_active",
    "clocks_event_reasons_counters.sw_power_cap": "ctr_sw_power_cap_us",
    "clocks_event_reasons_counters.sw_thermal_slowdown": "ctr_sw_thermal_us",
    "clocks_event_reasons_counters.hw_thermal_slowdown": "ctr_hw_thermal_us",
    "clocks_event_reasons_counters.hw_power_brake_slowdown": "ctr_hw_power_brake_us",
}
# nvml.h clock-event reason bits worth plotting: (bit, label, counter column, status color)
REASONS = [
    (0x04, "SW power cap", "ctr_sw_power_cap_us", "serious"),
    (0x20, "SW thermal slowdown", "ctr_sw_thermal_us", "critical"),
    (0x40, "HW thermal slowdown", "ctr_hw_thermal_us", "critical"),
    (0x08, "HW slowdown", None, "critical"),
    (0x80, "HW power brake", "ctr_hw_power_brake_us", "critical"),
]

# Plot tokens (light surface). Categorical order is fixed; validated for CVD separation.
C = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", wash="#f0efec")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
S1_LIGHT = "#b7d3f6"
STATUS = dict(serious="#ec835a", critical="#d03b3b")
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


# ---------------------------------------------------------------------------------------- logging
class Rows:
    """CSV writer that survives crashes: flush every row, fsync every 2 s. Thread-safe."""

    def __init__(self, path, header):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.lock = threading.Lock()
        self.synced = time.monotonic()
        self.write(header)

    def write(self, row):
        with self.lock:
            self.w.writerow(row)
            self.f.flush()
            if time.monotonic() - self.synced > 2:
                os.fsync(self.f.fileno())
                self.synced = time.monotonic()

    def close(self):
        with self.lock:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    body = [r for r in rows[1:] if len(r) == len(head)]  # a crash can leave a torn last row
    out = {}
    for j, name in enumerate(head):
        vals = [r[j] for r in body]
        try:
            out[name] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[name] = np.array(vals)
    return out


# -------------------------------------------------------------------------------------- telemetry
def smi(*args):
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout.strip()


def probe_fields():
    return [f for f in SMI_FIELDS if smi(f"--query-gpu={f}", "--format=csv,noheader,nounits")[0] == 0]


def smi_value(s):
    s = s.strip()
    if not s or "N/A" in s or "Not Supported" in s:
        return ""
    return str(int(s, 16)) if s.startswith("0x") else s


def other_gpu_processes():
    _, out = smi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
    me = str(os.getpid())
    return [l.strip() for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != me]


def host_temp_paths():
    paths = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() == "acpitz":
                paths += sorted(glob.glob(f"{h}/temp*_input"))
        except OSError:
            pass
    return paths


def read_host(paths):
    temps = []
    for p in paths:
        try:
            temps.append(int(open(p).read()) / 1000)
        except (OSError, ValueError):
            pass
    tmax = f"{max(temps):.1f}" if temps else ""
    tmean = f"{sum(temps) / len(temps):.1f}" if temps else ""
    return [tmax, tmean, f"{os.getloadavg()[0]:.2f}"]


class Telemetry:
    """nvidia-smi in loop mode on a reader thread; each line is stamped on arrival with our clock."""

    def __init__(self, path, t0, fields, period_ms):
        self.t0, self.phase = t0, "setup"
        self.cols = [SMI_FIELDS[f] for f in fields]
        self.host_paths = host_temp_paths()
        self.rows = Rows(path, ["t", "phase", *self.cols, "host_temp_max_C", "host_temp_mean_C", "loadavg_1m"])
        self.latest, self.temps = {}, deque(maxlen=20000)
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits", "-lms", str(period_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            t = time.perf_counter() - self.t0
            parts = line.split(",")
            if len(parts) != len(self.cols):
                continue
            vals = [smi_value(p) for p in parts]
            self.rows.write([f"{t:.3f}", self.phase, *vals, *read_host(self.host_paths)])
            self.latest = dict(zip(self.cols, vals))
            if self.latest.get("gpu_temp_C"):
                self.temps.append((t, float(self.latest["gpu_temp_C"])))

    def status(self):
        L = self.latest
        return (f"{L.get('gpu_temp_C') or '?':>3} °C  {L.get('power_W') or '?':>6} W  "
                f"{L.get('sm_clock_MHz') or '?':>5} MHz  {reasons_str(L.get('reasons_active'))}")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.thread.join(5)
        self.rows.close()


def reasons_str(v):
    if not v:
        return ""
    names = [name for bit, name, _, _ in REASONS if int(float(v)) & bit]
    return "[" + ", ".join(names) + "]" if names else ""


# ------------------------------------------------------------------------------------------ phases
def wait_for_cool(tel, max_s, window_s, tol_C):
    """Idle until GPU temp spans <= tol_C over the last window_s (integer °C readings, so tol 1 = flat)."""
    if max_s <= 0:
        return "skipped"
    start = last_print = time.perf_counter() - tel.t0
    while True:
        time.sleep(1)
        now = time.perf_counter() - tel.t0
        recent = [T for t, T in tel.temps if t >= now - window_s]
        span = max(recent) - min(recent) if recent else float("inf")
        if now - start >= window_s and len(recent) >= 4 and span <= tol_C:
            print(f"  idle temperature stable at {recent[-1]:.0f} °C after {now - start:.0f} s", flush=True)
            return "stable"
        if now - start >= max_s:
            print(f"  WARNING: temperature still moving after {max_s:.0f} s (span {span:.1f} °C); starting anyway", flush=True)
            return "timeout"
        if now - last_print >= 15:
            print(f"  waiting for idle: {now - start:4.0f} s, {tel.status()}, {window_s:.0f} s span {span:.1f} °C", flush=True)
            last_print = now


def copy_gbps(x, y, seconds):
    import torch

    y.copy_(x)
    torch.cuda.synchronize()
    n, t = 0, time.perf_counter()
    while n < 3 or time.perf_counter() - t < seconds:
        y.copy_(x)
        torch.cuda.synchronize()
        n += 1
    return 2 * x.numel() * n / (time.perf_counter() - t) / 1e9  # read + write, day2 convention


def soak(a, b, c, k, seconds, rows, tel, flop):
    import torch

    for _ in range(2):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    start = last = print_t = time.perf_counter()
    work = 0.0
    while last - start < seconds:
        for _ in range(k):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        now = time.perf_counter()
        rows.write([f"{now - tel.t0:.4f}", k, f"{now - last:.6f}", f"{k * flop / (now - last) / 1e12:.3f}"])
        work += k * flop
        last = now
        if now - print_t >= 5:
            print(f"  {now - start:5.0f} s  {work / (now - print_t) / 1e12:6.1f} TFLOP/s  {tel.status()}", flush=True)
            work, print_t = 0.0, now


def cmd_run(a):
    others = other_gpu_processes()
    if others and not a.allow_shared:
        print("Other processes are on the GPU; their load would heat the chip and steal cycles:", file=sys.stderr)
        for o in others:
            print("   ", o, file=sys.stderr)
        print("Stop them, or pass --allow-shared (the run is then flagged as contended).", file=sys.stderr)
        sys.exit(2)

    import torch

    run_dir = os.path.join(a.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    fields = probe_fields()
    _, clk_max = smi("--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits")
    _, driver = smi("--query-gpu=driver_version", "--format=csv,noheader")
    flop = 2 * a.n**3
    meta = dict(
        started=datetime.now().isoformat(timespec="seconds"), argv=sys.argv, device=torch.cuda.get_device_name(),
        driver=driver, torch=torch.__version__, n=a.n, dtype=a.dtype, flop_per_matmul=flop,
        load_s_target=a.load_s, sm_clock_max_MHz=float(smi_value(clk_max) or "nan"), smi_fields=fields,
        other_gpu_processes_start=others, loadavg_start=os.getloadavg()[0], ref_burst_tflops=a.ref_burst,
        ref_copy_gbps=REF_COPY_GBPS, predict_tflops=a.predict, note=a.note, phases={}, interrupted=False,
    )

    def save_meta():
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    tel = Telemetry(os.path.join(run_dir, "telemetry.csv"), t0, fields, a.smi_ms)
    samples = Rows(os.path.join(run_dir, "samples.csv"), ["t", "n", "dt", "tflops"])

    @contextmanager
    def phase(name):
        print(f"[{now():6.0f} s] {name}", flush=True)
        tel.phase = name
        meta["phases"][name] = [now(), None]
        save_meta()
        try:
            yield
        finally:
            meta["phases"][name][1] = now()
            tel.phase = "between"
            save_meta()

    print(f"run dir: {run_dir}\ntelemetry fields: {', '.join(SMI_FIELDS[f] for f in fields)}", flush=True)
    try:
        dtype = getattr(torch, a.dtype)
        A, B = (torch.randn(a.n, a.n, device="cuda", dtype=dtype) for _ in range(2))
        Cm = torch.empty_like(A)
        for _ in range(10):
            torch.matmul(A, B, out=Cm)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            torch.matmul(A, B, out=Cm)
        torch.cuda.synchronize()
        per_call = (time.perf_counter() - t) / 5
        meta["chunk_matmuls"] = k = max(1, round(a.chunk_s / per_call))
        print(f"  {per_call * 1e3:.1f} ms/matmul -> sync every {k} matmuls", flush=True)
        x = torch.empty(int(a.copy_gib * 2**30), device="cuda", dtype=torch.uint8) if a.copy_s > 0 else None
        y = torch.empty_like(x) if x is not None else None

        if x is not None:
            with phase("copy_cold"):
                meta["copy_cold_GBps"] = copy_gbps(x, y, a.copy_s)
                print(f"  beta cold {meta['copy_cold_GBps']:.1f} GB/s", flush=True)
        with phase("cool_wait"):
            meta["cool_wait"] = wait_for_cool(tel, a.cool_wait_max_s, a.cool_window_s, a.cool_tol_C)
        with phase("load"):
            soak(A, B, Cm, k, a.load_s, samples, tel, flop)
        meta["other_gpu_processes_end"] = other_gpu_processes()
        if x is not None:
            with phase("copy_hot"):
                meta["copy_hot_GBps"] = copy_gbps(x, y, a.copy_s)
                print(f"  beta hot {meta['copy_hot_GBps']:.1f} GB/s", flush=True)
        del A, B, Cm, x, y
        torch.cuda.empty_cache()
        with phase("cooldown"):
            end = time.perf_counter() + a.cooldown_s
            while time.perf_counter() < end:
                time.sleep(min(15, max(0, end - time.perf_counter())))
                print(f"  cooling: {tel.status()}", flush=True)
    except KeyboardInterrupt:
        meta["interrupted"] = True
        print("\ninterrupted; analysing what was recorded", flush=True)
    finally:
        tel.stop()
        samples.close()
        save_meta()

    summary = analyze(run_dir)
    plot_run(run_dir, summary)


# ---------------------------------------------------------------------------------------- analysis
def load_run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    return meta, read_csv(os.path.join(run_dir, "samples.csv")), read_csv(os.path.join(run_dir, "telemetry.csv"))


def load_window(meta, S):
    ph = meta["phases"]
    if "load" not in ph or len(S["t"]) == 0:
        raise SystemExit("no load phase recorded in this run")
    L0, L1 = ph["load"]
    return L0, (L1 if L1 is not None else float(S["t"].max()))  # None: the run died mid-soak


def chunk_series(meta, S):
    """Per chunk: midpoint time since soak start, FLOPs done, wall seconds."""
    L0, _ = load_window(meta, S)
    return S["t"] - L0 - S["dt"] / 2, S["n"] * meta["flop_per_matmul"], S["dt"]


def tput(W, dt, mask):
    return float(W[mask].sum() / dt[mask].sum() / 1e12) if mask.any() else float("nan")


def binned(s, W, dt, width):
    """Work-weighted throughput in fixed bins (sum FLOPs / sum seconds, not a mean of rates)."""
    idx = np.maximum(0, np.floor(s / width)).astype(int)
    nb = int(idx.max()) + 1
    work, secs = np.bincount(idx, W, nb), np.bincount(idx, dt, nb)
    keep = secs >= 0.5 * width
    return ((np.arange(nb) + 0.5) * width)[keep], (work[keep] / secs[keep] / 1e12)


def rolling(s, W, dt, width):
    cw, ct = np.concatenate([[0], np.cumsum(W)]), np.concatenate([[0], np.cumsum(dt)])
    lo = np.searchsorted(s, s - width / 2, "left")
    hi = np.searchsorted(s, s + width / 2, "right")
    return (cw[hi] - cw[lo]) / (ct[hi] - ct[lo]) / 1e12


def fit_exp(s, y, duration):
    """y = y_inf + amp * exp(-s / tau): grid over tau, linear least squares for (y_inf, amp)."""
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    if len(s) < 8 or np.ptp(y) == 0:
        return None
    tau_max = 20 * max(duration, 1)
    best = None
    for tau in np.geomspace(0.5, tau_max, 400):
        X = np.column_stack([np.ones_like(s), np.exp(-s / tau)])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((X @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, tau, coef)
    sse, tau, (y_inf, amp) = best
    return dict(y_inf=float(y_inf), amp=float(amp), tau_s=float(tau),
                r2=float(1 - sse / np.sum((y - y.mean()) ** 2)),
                converged=bool(3 * tau <= duration and tau < 0.95 * tau_max))  # 95% settled inside the window


def col(T, name):
    return T.get(name, np.full(len(T["t"]), np.nan))


def nanmean(v):
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def counter_frac(T, name, mask, seconds):
    v = col(T, name)[mask]
    v = v[np.isfinite(v)]
    return float((v[-1] - v[0]) / 1e6 / seconds) if len(v) >= 2 and seconds > 0 else None


def analyze(run_dir, quiet=False):
    meta, S, T = load_run(run_dir)
    L0, L1 = load_window(meta, S)
    s, W, dt = chunk_series(meta, S)
    dur = float((S["t"] - L0).max())
    ref = meta.get("ref_burst_tflops", REF_BURST_TFLOPS)

    first = np.zeros(len(s), bool)
    first[0] = True
    burst = tput(W, dt, (s + dt / 2 <= 0.5) | first)
    first5 = tput(W, dt, s <= 5)
    t5_15 = tput(W, dt, (s > 5) & (s <= 15))
    pw = min(60.0, 0.25 * dur)
    plateau = tput(W, dt, s >= dur - pw)
    bx, by = binned(s, W, dt, 1.0)
    in_plat = bx >= dur - pw
    plateau_cv = float(by[in_plat].std() / by[in_plat].mean()) if in_plat.sum() > 2 else None

    late = bx >= 5
    fit = fit_exp(bx[late], by[late], dur)
    if fit:
        fit["significant"] = bool(fit["amp"] > 0.01 * fit["y_inf"] and fit["r2"] >= 0.3)

    # sawtooth: periodic swings left after removing the trend (5 s smoothing kills chunk-level noise)
    saw_cv, saw_cross = None, None
    tail = bx >= 15
    if tail.sum() >= 30:
        trend = fit["y_inf"] + fit["amp"] * np.exp(-bx[tail] / fit["tau_s"]) if fit else by[tail].mean()
        r = np.convolve(by[tail] - trend, np.ones(5) / 5, mode="valid")
        saw_cv = float(r.std() / by[tail].mean())
        saw_cross = int(np.sum(np.diff(np.sign(r)) != 0))

    Tt = T["t"]
    tl = (Tt >= L0) & (Tt <= L1)
    ts_plat = tl & (Tt - L0 >= dur - pw)
    idle = (Tt >= L0 - 20) & (Tt < L0)
    temp, power = col(T, "gpu_temp_C"), col(T, "power_avg_W")
    if not np.isfinite(power).any():
        power = col(T, "power_W")
    temp_fit = fit_exp(Tt[tl] - L0, temp[tl], dur)
    T_idle, P_idle, P_load = nanmean(temp[idle]), nanmean(power[idle]), nanmean(power[ts_plat])
    thermal = None
    if temp_fit and np.isfinite(T_idle) and P_load - P_idle > 5:
        R = (temp_fit["y_inf"] - T_idle) / (P_load - P_idle)
        thermal = dict(R_th_K_per_W=R, C_th_J_per_K=temp_fit["tau_s"] / R if R > 0 else None,
                       T_idle_C=T_idle, P_idle_W=P_idle, P_load_W=P_load, note="GPU-reported power; effective values")
    cool_fit = None
    if meta["phases"].get("cooldown", [None, None])[1] is not None:
        c0, c1 = meta["phases"]["cooldown"]
        cm = (Tt >= c0) & (Tt <= c1)
        cool_fit = fit_exp(Tt[cm] - c0, temp[cm], c1 - c0)

    reasons = {}
    rs = col(T, "reasons_active")[tl]
    rs = rs[np.isfinite(rs)].astype(np.int64)
    for bit, name, ctr, _ in REASONS:
        reasons[name] = dict(
            sampled_frac=float(np.mean((rs & bit) != 0)) if len(rs) else None,
            counter_frac_load=counter_frac(T, ctr, tl, L1 - L0) if ctr else None,
            counter_frac_idle_before=counter_frac(T, ctr, idle, 20.0) if ctr else None,
        )

    sm_clk = col(T, "sm_clock_MHz")
    beta_hot = meta.get("copy_hot_GBps")
    beta = beta_hot or meta.get("copy_cold_GBps") or REF_COPY_GBPS
    shape, verdict = classify(dur, meta["load_s_target"], burst, t5_15, plateau, fit, saw_cv, saw_cross, ref)
    contended = bool(meta.get("other_gpu_processes_start") or meta.get("other_gpu_processes_end"))

    sm = dict(
        run=os.path.basename(os.path.normpath(run_dir)), started=meta["started"], soak_s=dur,
        burst_tflops=burst, first5s_tflops=first5, t5_15s_tflops=t5_15,
        plateau_tflops=plateau, plateau_window_s=pw, plateau_cv_1s=plateau_cv,
        drop_vs_burst_pct=100 * (1 - plateau / burst), plateau_vs_ref_pct=100 * (plateau / ref - 1), ref_burst_tflops=ref,
        fit_tput=fit, sawtooth=dict(resid_cv=saw_cv, crossings=saw_cross),
        temp_fit=temp_fit, cooldown_fit=cool_fit, thermal=thermal,
        tlimit_C=nanmean(temp + col(T, "tlimit_headroom_C")),
        sm_clock_MHz=dict(first5s=nanmean(sm_clk[tl & (Tt - L0 <= 5)]), plateau=nanmean(sm_clk[ts_plat]),
                          max=meta.get("sm_clock_max_MHz")),
        reasons=reasons,
        copy_cold_GBps=meta.get("copy_cold_GBps"), copy_hot_GBps=beta_hot,
        ridge_flop_per_byte=dict(sustained=plateau * 1e12 / (beta * 1e9), burst=burst * 1e12 / (beta * 1e9),
                                 beta_used_GBps=beta, ref=ref * 1e12 / (REF_COPY_GBPS * 1e9)),
        shape=shape, verdict=verdict, contended=contended, interrupted=meta.get("interrupted", False),
    )
    if meta.get("predict_tflops"):
        p = meta["predict_tflops"]
        sm["prediction"] = dict(predicted=p, error_pct=100 * (plateau - p) / p,
                                within_5pct=abs(plateau - p) <= 0.05 * p,
                                plateau_within_5pct_of_ref=abs(plateau - ref) <= 0.05 * ref)
    sm["ledger"] = ledger_line(sm)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(sm, f, indent=2)
    if not quiet:
        print_summary(sm)
    return sm


def classify(dur, target, burst, t5_15, plateau, fit, saw_cv, saw_cross, ref):
    """Map the curve onto the task's shape table. Heuristic thresholds, stated in the verdict text."""
    if dur < min(60, 0.5 * target) or not np.isfinite(t5_15):
        return "incomplete", f"soak stopped at {dur:.0f} s; too short to call"
    if saw_cv is not None and saw_cv > 0.03 and saw_cross >= 6:
        return "sawtooth", f"sawtooth: ±{100 * saw_cv:.0f}% swings around the trend; report the plateau mean"
    drop = 1 - plateau / burst
    if drop < 0.03:
        if plateau < 0.95 * ref:
            return "flat_below_ref", (f"flat, but {100 * (1 - plateau / ref):.0f}% under the {ref:.1f} reference: "
                                      "capped from the first ms, or contended")
        return "flat", "flat: no throttling, the burst number stands as sustained"
    if t5_15 - plateau < 0.25 * (burst - plateau):
        return "power_cap_step", f"{100 * drop:.0f}% drop inside the first window, then flat: power cap binding"
    if fit and fit["significant"] and not fit["converged"]:
        return "thermal_unconverged", f"still decaying at the end (τ ≈ {fit['tau_s']:.0f} s > soak/3): run longer"
    tau = f", τ ≈ {fit['tau_s']:.0f} s" if fit and fit["significant"] else ""
    return "thermal_decay", f"slow {100 * drop:.0f}% decay{tau}: thermal throttling, the plateau is the ceiling"


def ledger_line(sm):
    day = sm["started"][:10]
    line = (f"{day} burst-vs-sustained GB10: burst {sm['burst_tflops']:.1f} -> sustained {sm['plateau_tflops']:.1f} "
            f"TFLOP/s ({-sm['drop_vs_burst_pct']:+.1f}%, {sm['soak_s']:.0f} s heat-soaked), {sm['shape']}; "
            f"I* = {sm['ridge_flop_per_byte']['sustained']:.0f} FLOP/byte")
    if sm.get("copy_hot_GBps"):
        line += f"; beta hot {sm['copy_hot_GBps']:.0f} GB/s"
    if sm.get("prediction"):
        p = sm["prediction"]
        line += f"; predicted {p['predicted']:.1f} -> {'correct' if p['within_5pct'] else 'wrong'} ({p['error_pct']:+.1f}%)"
    if sm["contended"]:
        line += "; CONTENDED (other GPU processes)"
    return line


def print_summary(sm):
    f = lambda v, fmt: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    fit, tf, cf, th = sm["fit_tput"], sm["temp_fit"], sm["cooldown_fit"], sm["thermal"]
    print("\n" + "=" * 78)
    print(f"burst (first 0.5 s)       {f(sm['burst_tflops'], '7.1f')} TFLOP/s")
    print(f"first 5 s                 {f(sm['first5s_tflops'], '7.1f')}")
    print(f"sustained (last {sm['plateau_window_s']:.0f} s)     {f(sm['plateau_tflops'], '7.1f')}   "
          f"{-sm['drop_vs_burst_pct']:+.1f}% vs burst, {sm['plateau_vs_ref_pct']:+.1f}% vs ref {sm['ref_burst_tflops']:.1f}")
    if fit:
        print(f"throughput fit            pi_inf {fit['y_inf']:.1f}, tau {fit['tau_s']:.0f} s, R² {fit['r2']:.2f}, "
              f"{'significant' if fit['significant'] else 'no significant decay'}, "
              f"{'converged' if fit['converged'] else 'NOT converged'}")
    conv = lambda fit: "" if fit["converged"] else " (NOT converged: extrapolated)"
    if tf:
        print(f"temperature (soak)        T_inf {tf['y_inf']:.1f} °C, tau {tf['tau_s']:.0f} s{conv(tf)}   "
              f"T.Limit ≈ {f(sm['tlimit_C'], '.0f')} °C")
    if cf:
        print(f"temperature (cooldown)    T_end {cf['y_inf']:.1f} °C, tau {cf['tau_s']:.0f} s{conv(cf)}")
    if th:
        print(f"thermal RC (effective)    R_th {th['R_th_K_per_W']:.3f} K/W, C_th {f(th['C_th_J_per_K'], '.0f')} J/K "
              f"({th['P_idle_W']:.0f} -> {th['P_load_W']:.0f} W)")
    clk = sm["sm_clock_MHz"]
    print(f"SM clock                  {f(clk['first5s'], '.0f')} -> {f(clk['plateau'], '.0f')} MHz (max {f(clk['max'], '.0f')})")
    for name, r in sm["reasons"].items():
        if r["sampled_frac"] or r["counter_frac_load"]:
            print(f"{name:<25} active in {100 * (r['sampled_frac'] or 0):.0f}% of samples, "
                  f"counter {f(r['counter_frac_load'] and 100 * r['counter_frac_load'], '.0f')}% of soak")
    if sm["copy_cold_GBps"] or sm["copy_hot_GBps"]:
        print(f"beta cold -> hot          {f(sm['copy_cold_GBps'], '.1f')} -> {f(sm['copy_hot_GBps'], '.1f')} GB/s")
    rid = sm["ridge_flop_per_byte"]
    print(f"ridge I*                  {rid['sustained']:.0f} FLOP/byte sustained, {rid['burst']:.0f} burst, "
          f"{rid['ref']:.0f} ref (beta {rid['beta_used_GBps']:.0f} GB/s)")
    print(f"shape                     {sm['verdict']}")
    if sm["contended"]:
        print("WARNING                   other processes shared the GPU; these numbers are contaminated")
    print(f"\nledger: {sm['ledger']}")
    print("=" * 78)


# -------------------------------------------------------------------------------------------- plots
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "text.color": C["primary"], "axes.labelcolor": C["secondary"],
        "axes.edgecolor": C["axis"], "axes.facecolor": C["surface"], "figure.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"],
        "grid.linewidth": 0.8, "xtick.color": C["muted"], "ytick.color": C["muted"], "ytick.major.size": 0,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    return plt


def _end_label(ax, x, y, text):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        ax.annotate(text, (x[ok][-1], y[ok][-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=C["secondary"])


def _phase_spans(ax, meta, L0, x_lo, label=False):
    names = dict(cool_wait="idle", load="matmul soak", copy_hot="copy", cooldown="idle cooldown")
    for name, (p0, p1) in meta["phases"].items():
        if name not in names or p1 is None or p1 - L0 < x_lo:
            continue
        a, b = max(p0 - L0, x_lo), p1 - L0
        if name != "load":
            ax.axvspan(a, b, color=C["wash"], lw=0, zorder=0)
        if label and b - a > 0:
            ax.text((a + b) / 2, 1.0, names[name], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=C["muted"])


def plot_run(run_dir, sm):
    plt = _mpl()
    meta, S, T = load_run(run_dir)
    L0, _ = load_window(meta, S)
    s, W, dt = chunk_series(meta, S)
    dur, pw = sm["soak_s"], sm["plateau_window_s"]
    Ts = T["t"] - L0
    x_lo = max(-30.0, float(Ts.min()))
    x_hi = max(float(Ts.max()), 1.15 * dur)  # room for the plateau label even without a cooldown
    view = Ts >= x_lo
    ref = sm["ref_burst_tflops"]
    legend_above = dict(loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.2, 1.8, 1.4, 1.4, 1.1], hspace=0.5))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.855, bottom=0.05)

    # 1. throughput
    ax = axes[0]
    ax.plot(s, S["tflops"], color=S1_LIGHT, lw=0.6, label="per chunk (~0.1 s)")
    roll = rolling(s, W, dt, 5.0)
    ax.plot(s, roll, color=SERIES[0], lw=2, label="5 s rolling mean")
    fit = sm["fit_tput"]
    if fit and fit["significant"]:
        xs = np.linspace(5, dur, 300)
        ax.plot(xs, fit["y_inf"] + fit["amp"] * np.exp(-xs / fit["tau_s"]), color=C["secondary"], lw=1.2,
                ls=(0, (4, 3)), label=f"fit π∞ + A·exp(−t/τ): π∞ {fit['y_inf']:.1f}, τ {fit['tau_s']:.0f} s")
    ax.axhline(ref, color=C["muted"], lw=0.8, zorder=1)
    ax.annotate(f"09-10 roofline\nburst {ref:.1f}", (1, ref), xycoords=("axes fraction", "data"), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=7.5, color=C["muted"])
    bbox = dict(fc=C["surface"], ec="none", pad=1.5, alpha=0.85)
    ax.plot([dur - pw, dur], [sm["plateau_tflops"]] * 2, color=C["primary"], lw=3, solid_capstyle="butt", zorder=4)
    ax.annotate(f"sustained {sm['plateau_tflops']:.1f}\n(last {pw:.0f} s)", (dur, sm["plateau_tflops"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8.5, fontweight="bold", bbox=bbox)
    lo, hi = np.nanpercentile(S["tflops"], [1, 99.5])
    top = max(hi, ref, sm["burst_tflops"])
    ax.set_ylim(min(lo * 0.97, top * 0.8), top * 1.06)  # never zoom so far that noise reads as a cliff
    y0, y1 = ax.get_ylim()
    above = sm["burst_tflops"] < (y0 + y1) / 2
    ax.plot([0.25], [sm["burst_tflops"]], "o", ms=8, color=SERIES[0], mec=C["surface"], mew=2, zorder=5)
    ax.annotate(f"burst {sm['burst_tflops']:.1f} (first 0.5 s)", (0.25, sm["burst_tflops"]),
                xytext=(10, 12 if above else -14), textcoords="offset points", va="bottom" if above else "top",
                fontsize=8.5, fontweight="bold", bbox=bbox, zorder=6)
    ax.set_ylabel("throughput (TFLOP/s)")
    ax.legend(ncols=3, **{**legend_above, "bbox_to_anchor": (0, 1.07)})  # phase labels sit just under it
    _phase_spans(ax, meta, L0, x_lo, label=True)

    # 2. temperature
    ax = axes[1]
    temp, host = col(T, "gpu_temp_C"), col(T, "host_temp_max_C")
    ax.plot(Ts[view], temp[view], color=SERIES[0], lw=2, label="GPU (nvidia-smi)")
    _end_label(ax, Ts[view], temp[view], "GPU")
    if np.isfinite(host).any():
        ax.plot(Ts[view], host[view], color=SERIES[1], lw=2, label="host, hottest ACPI zone")
        _end_label(ax, Ts[view], host[view], "host")
    tf = sm["temp_fit"]
    if tf:
        xs = np.linspace(0, dur, 300)
        ax.plot(xs, tf["y_inf"] + tf["amp"] * np.exp(-xs / tf["tau_s"]), color=C["secondary"], lw=1.2,
                ls=(0, (4, 3)), label=f"fit: T∞ {tf['y_inf']:.0f} °C, τ {tf['tau_s']:.0f} s")
    if np.isfinite(sm["tlimit_C"]):
        ax.axhline(sm["tlimit_C"], color=C["muted"], lw=0.8)
        ax.text(x_lo, sm["tlimit_C"], f" T.Limit ≈ {sm['tlimit_C']:.0f} °C", va="bottom", fontsize=7.5, color=C["muted"])
    ax.set_ylabel("temperature (°C)")
    ax.legend(ncols=3, **legend_above)
    _phase_spans(ax, meta, L0, x_lo)

    # 3. power
    ax = axes[2]
    ax.plot(Ts[view], col(T, "power_W")[view], color=S1_LIGHT, lw=0.8, label="instant")
    ax.plot(Ts[view], col(T, "power_avg_W")[view], color=SERIES[0], lw=2, label="1 s average")
    ax.set_ylabel("GPU power (W)")
    ax.legend(ncols=2, **legend_above)
    _phase_spans(ax, meta, L0, x_lo)

    # 4. SM clock
    ax = axes[3]
    ax.plot(Ts[view], col(T, "sm_clock_MHz")[view], color=SERIES[0], lw=2)
    cmax = sm["sm_clock_MHz"]["max"]
    if cmax and np.isfinite(cmax):
        ax.axhline(cmax, color=C["muted"], lw=0.8)
        ax.text(x_lo, cmax, f" clocks.max.sm {cmax:.0f} MHz", va="top", fontsize=7.5, color=C["muted"])
    ax.set_ylabel("SM clock (MHz)")
    _phase_spans(ax, meta, L0, x_lo)

    # 5. clock-event reasons, straight from the driver
    ax = axes[4]
    ax.grid(False)
    rs = col(T, "reasons_active")
    labels = []
    any_on = False
    for i, (bit, name, _, status) in enumerate(REASONS):
        on = view & np.isfinite(rs) & ((np.nan_to_num(rs).astype(np.int64) & bit) != 0)
        idx = np.flatnonzero(on)
        segs = [(Ts[j], (Ts[j + 1] - Ts[j]) if j + 1 < len(Ts) else 0.25) for j in idx]
        if segs:
            any_on = True
            ax.broken_barh(segs, (i - 0.32, 0.64), facecolor=STATUS[status], lw=0)
        frac = sm["reasons"][name]["counter_frac_load"]
        labels.append(f"{name} ({100 * frac:.0f}%)" if frac is not None else name)
    ax.set_yticks(range(len(REASONS)), labels, fontsize=7.5)
    ax.tick_params(axis="y", colors=C["secondary"])
    ax.text(0, 1.0, "clock-event reasons: bars = sampled bit, % = driver counter over the soak",
            transform=ax.transAxes, va="bottom", fontsize=8, color=C["secondary"])
    ax.set_ylim(len(REASONS) - 0.5, -0.5)
    if not any_on:
        ax.text(0.5, 0.5, "no clock-event reasons active", transform=ax.transAxes, ha="center", va="center",
                color=C["muted"], fontsize=8.5, bbox=bbox)
    _phase_spans(ax, meta, L0, x_lo)
    ax.set_xlabel("time since soak start (s)")
    ax.set_xlim(x_lo, x_hi + 0.02 * (x_hi - x_lo))

    head = (f"Burst {sm['burst_tflops']:.1f} → sustained {sm['plateau_tflops']:.1f} TFLOP/s "
            f"({-sm['drop_vs_burst_pct']:+.1f}%)")
    if sm["contended"]:
        fig.text(0.15, 0.99, "⚠ Other GPU processes were running during this run: numbers are contaminated",
                 va="top", fontsize=9, color=STATUS["critical"], fontweight="bold")
    fig.text(0.15, 0.955, head, fontsize=15, fontweight="bold", va="bottom")
    sub = (f"{sm['verdict']}\n{meta['n']}² {meta['dtype']} matmul on {meta['device']}, {dur:.0f} s soak, "
           f"started {meta['started'].replace('T', ' ')}" + (f", {meta['note']}" if meta.get("note") else ""))
    fig.text(0.15, 0.948, sub, fontsize=9.5, color=C["secondary"], va="top", linespacing=1.5)
    out = os.path.join(run_dir, "timeseries.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")
    plot_mechanism(run_dir, sm, meta, S, T, plt)


def plot_mechanism(run_dir, sm, meta, S, T, plt):
    from matplotlib.colors import LinearSegmentedColormap

    L0, L1 = load_window(meta, S)
    s, W, dt = chunk_series(meta, S)
    bx, by = binned(s, W, dt, 2.0)
    Ts = T["t"] - L0
    in_load = (T["t"] >= L0) & (T["t"] <= L1)
    xs = []
    for name, label in [("gpu_temp_C", "GPU temperature (°C)"), ("power_avg_W", "GPU power, 1 s average (W)")]:
        v = col(T, name)
        ok = in_load & np.isfinite(v)
        if ok.sum() >= 2:
            xs.append((np.interp(bx, Ts[ok], v[ok]), label))
    if len(bx) < 5 or not xs:
        return
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, len(xs), figsize=(5.5 * len(xs), 4.6), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.8, bottom=0.13, wspace=0.08)
    for ax, (xv, label) in zip(axes[0], xs):
        ax.grid(True, axis="both")
        sc = ax.scatter(xv, by, c=bx, cmap=cmap, s=34, edgecolors=C["surface"], linewidths=1, zorder=3)
        r = np.corrcoef(xv, by)[0, 1] if np.ptp(xv) > 0 and np.ptp(by) > 0 else float("nan")
        ax.set_title(f"r = {r:+.2f}", loc="left", fontsize=9, color=C["secondary"])
        ax.set_xlabel(label)
    axes[0][0].set_ylabel("throughput, 2 s bins (TFLOP/s)")
    cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("time since soak start (s)", color=C["secondary"])
    cb.outline.set_visible(False)
    fig.text(0.08, 0.93, "What does the throughput drop track?", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.875, "Thermal throttling: throughput falls as temperature climbs. "
             "A power cap: a vertical step at low temperature, then flat.", fontsize=9, color=C["secondary"])
    out = os.path.join(run_dir, "mechanism.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def cmd_plot(a):
    plot_run(a.run_dir, analyze(a.run_dir))


def cmd_compare(a):
    if len(a.runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per comparison")
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit("--labels needs one label per run")
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1.3], hspace=0.12))
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.08)
    ref = None
    for i, (rd, lab) in enumerate(zip(a.runs, labels)):
        meta, S, T = load_run(rd)
        sm = analyze(rd, quiet=True)
        L0, L1 = load_window(meta, S)
        s, W, dt = chunk_series(meta, S)
        roll = rolling(s, W, dt, 5.0)
        ax1.plot(s, roll, color=SERIES[i], lw=2,
                 label=f"{lab}: {sm['burst_tflops']:.1f} → {sm['plateau_tflops']:.1f}{' (contended)' if sm['contended'] else ''}")
        _end_label(ax1, s, roll, lab)
        m = (T["t"] >= L0) & (T["t"] <= L1)
        ax2.plot(T["t"][m] - L0, col(T, "gpu_temp_C")[m], color=SERIES[i], lw=2, label=lab)
        _end_label(ax2, T["t"][m] - L0, col(T, "gpu_temp_C")[m], lab)
        ref = sm["ref_burst_tflops"]
    ax1.axhline(ref, color=C["muted"], lw=0.8, zorder=1, label=f"2026-09-10 roofline burst {ref:.1f}")
    ax1.set_ylabel("throughput, 5 s rolling (TFLOP/s)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(4, len(a.runs) + 1), borderaxespad=0.2)
    ax2.set_ylabel("GPU temperature (°C)")
    ax2.set_xlabel("time since soak start (s)")
    fig.text(0.08, 0.95, "Burst vs sustained, across runs", fontsize=14, fontweight="bold")
    fig.savefig(a.out, dpi=170)
    plt.close(fig)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure: cool-wait, soak, cooldown")
    r.add_argument("--load-s", type=float, default=600, help="matmul soak length (s)")
    r.add_argument("--cooldown-s", type=float, default=120, help="idle telemetry after the soak (s)")
    r.add_argument("--cool-wait-max-s", type=float, default=600, help="max idle wait for a cold start; 0 skips")
    r.add_argument("--cool-window-s", type=float, default=30, help="temperature must be flat over this window")
    r.add_argument("--cool-tol-C", type=float, default=1.0, help="allowed temperature span over the window")
    r.add_argument("--n", type=int, default=8192)
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--chunk-s", type=float, default=0.1, help="target time between synchronizations")
    r.add_argument("--copy-s", type=float, default=5, help="beta measurement length, cold and hot; 0 skips")
    r.add_argument("--copy-gib", type=float, default=2)
    r.add_argument("--smi-ms", type=int, default=250, help="nvidia-smi sampling period")
    r.add_argument("--predict", type=float, help="your plateau prediction (TFLOP/s), scored in summary.json")
    r.add_argument("--ref-burst", type=float, default=REF_BURST_TFLOPS)
    r.add_argument("--note", default="", help="free text stored with the run and shown on the plot, e.g. orientation")
    r.add_argument("--allow-shared", action="store_true", help="run even if other processes use the GPU")
    r.add_argument("--out", default=os.path.join(HERE, "results"))
    p = sub.add_parser("plot", help="re-analyse and re-plot a saved run")
    p.add_argument("run_dir")
    c = sub.add_parser("compare", help="overlay several runs")
    c.add_argument("runs", nargs="+")
    c.add_argument("--labels", nargs="+")
    c.add_argument("--out", default=os.path.join(HERE, "compare.png"))
    a = ap.parse_args()
    dict(run=cmd_run, plot=cmd_plot, compare=cmd_compare)[a.cmd](a)


if __name__ == "__main__":
    main()
