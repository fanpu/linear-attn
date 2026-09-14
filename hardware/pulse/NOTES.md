# Pulse NOTES (handoff)

- `record.py --tag phases` (13.5 min) / `--tag soak --soak 900` (21 min): nvidia-smi --loop-ms=100 + host hwmon/cpufreq sampler
  around `workload.py`. Outputs `cache/<tag>_{sensors.csv,host.csv,phases.json,meta.json}`. Needs decode-map's model on sys.path.
- `render_pulse.py --tag phases soak`: phases -> `pulse_phases_*`, `pulse_square_phases_*`; soak -> `pulse_soak_soak_*` + `cache/soak_fits.json`.
  Fix this session: the CSV header has units ("power.draw.instant [W]") and `clocks.current.sm`; `load_sensors` now strips units and
  aliases to `clocks.sm` / `clocks.mem` (the square/soak plates indexed the raw names and crashed).
- Sensor bandwidth: instant power changes ~1.9x per second (819 distinct values / 8059 samples at 100 ms). A 2 Hz matmul square
  wave is invisible (contrast -0.6 W; reading frozen at 118 W for 13 s then 14 W) while T climbs 59 -> 78 C. The 0.5 Hz wave is
  followed (contrast 40 W). Token-rate rhythm is impossible with NVML here.
- Phase numbers (2026-09-14 11:45): idle 11.4 W / 37 C; matmul 94 W sustained (123 W first samples), SM 2400 -> 2150 MHz, no
  throttle reason flagged; copy 27 W; prefill B64x256 55 W, 20.8k tok/s, 2.6 mJ/token; decode B64 47 W, 2044 tok/s, 23 mJ/token;
  decode B1 28 W, 84 tok/s, 0.34 J/token.
- Soak (11:59-12:20): heating tau 30 s (+20.4 C from 59), plateau 80-82 C at 89 W / 2150 MHz, 94.7 TFLOP/s sustained over 900 s,
  cooling tau 44 s; unexplained 4 C drop at ~560 s at constant power (fan step? no fan sensor). `cache/soak_fits.json`.
- Status: COMPLETE (README, gallery, index row). Extras: a longer soak to reach steady state, fan/pump sensors if any appear
  in hwmon, correlating the clock sag with the power reading at 10 ms via NVML directly (pynvml) instead of nvidia-smi.
