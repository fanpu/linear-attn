"""record.py - sample the GB10's sensors while a scripted workload runs; write cache/<tag>_sensors.csv and
cache/<tag>_phases.json. Sampling is nvidia-smi --loop-ms (NVML) plus host hwmon/cpufreq from Python.

    python record.py --tag phases            # the phase script below (~12 min)
    python record.py --tag soak --soak 900   # 15 min sustained bf16 matmul, then 5 min cool-down

Sensor bandwidth (measured 2026-09-14): power.draw.instant changes about every 0.5 s (16 distinct values in
387 samples at 20 ms), so anything faster than ~1 Hz is invisible. Sampling at 100 ms only sharpens the edges.
"""
import argparse, csv, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/cache"
FIELDS = ("timestamp,power.draw.instant,power.draw.average,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,"
          "utilization.memory,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,"
          "clocks_event_reasons.hw_thermal_slowdown")


def hwmon_sensors():
    out = {}
    base = "/sys/class/hwmon"
    for h in sorted(os.listdir(base)):
        try:
            name = open(f"{base}/{h}/name").read().strip()
        except Exception:
            continue
        for f in sorted(os.listdir(f"{base}/{h}")):
            if f.endswith("_input") and (f.startswith("temp") or f.startswith("power") or f.startswith("fan")):
                out[f"{name}.{f[:-6]}"] = f"{base}/{h}/{f}"
    return out


def host_sampler(path, stop, period=0.25):
    sens = hwmon_sensors()
    freqs = {f"cpu{c}.MHz": f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq" for c in (0, 5, 10, 15)}
    cols = ["t"] + list(sens) + list(freqs) + ["loadavg1"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        while not stop.is_set():
            row = [time.time()]
            for p in list(sens.values()):
                try:
                    row.append(int(open(p).read()))
                except Exception:
                    row.append("")
            for p in freqs.values():
                try:
                    row.append(int(open(p).read()) / 1000)
                except Exception:
                    row.append("")
            row.append(open("/proc/loadavg").read().split()[0])
            w.writerow(row); fh.flush()
            time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phases")
    ap.add_argument("--soak", type=int, default=0, help="seconds of sustained matmul instead of the phase script")
    ap.add_argument("--loop-ms", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,nounits", f"--loop-ms={a.loop_ms}"],
                           stdout=open(f"{CACHE}/{a.tag}_sensors.csv", "w"), stderr=subprocess.DEVNULL)
    stop = threading.Event()
    th = threading.Thread(target=host_sampler, args=(f"{CACHE}/{a.tag}_host.csv", stop), daemon=True); th.start()
    t0 = time.time()
    try:
        args = [sys.executable, f"{HERE}/workload.py", "--phases-out", f"{CACHE}/{a.tag}_phases.json"]
        if a.soak:
            args += ["--soak", str(a.soak)]
        subprocess.run(args, check=True)
    finally:
        time.sleep(2)
        stop.set(); smi.terminate(); smi.wait()
    json.dump(dict(t0=t0, t1=time.time(), loop_ms=a.loop_ms, host_sensors=list(hwmon_sensors())),
              open(f"{CACHE}/{a.tag}_meta.json", "w"), indent=1)
    print("recorded", a.tag, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
