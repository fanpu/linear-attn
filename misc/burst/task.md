# Side project: burst vs sustained compute on GB10

**Status:** filler task, about 20 minutes (a 15-minute run plus reading the output). Run it on a blocked day, or immediately if any tok/s number looks inexplicably low. It must be done before any multi-hour training run whose throughput you plan to report.

**Tool:** `burst.py` in this directory (`python burst.py --help`). It replaces the two-terminal snippet below.

## Question

Is the 93.9 TFLOP/s measured on 2026-09-10 a sustained ceiling, or a burst number that decays once the box heats up?

## Pre-flight findings (2026-09-11, before any clean run)

Resolve or keep in mind before reading the result:

1. **Which baseline is real is in doubt.** The committed `day2/roofline.json` (14d3ee6) says 93.9 TFLOP/s, 246.8 GB/s, and $I^*$ = 381. The uncommitted working-tree version says 35.6 TFLOP/s (both the 0.5 s burst and 30 s sustained), 158.4 GB/s, and $I^*$ = 225.
   - A 62% drop that is identical at 0.5 s and 30 s is not thermal. It looks like contention.
   - While I was checking the box, another job (`02_build_cache.py`) held the GPU at 90% utilization, and a second process joined it.
   - A 20 s smoke run under that contention measured about 40 TFLOP/s. It jumped to 94 for the few seconds the other job paused, even with the chip at 75 °C. That is contaminated data and only a weak hint, but it points at the other job, not at the box.
   - **Do not overwrite roofline.json with the 35.6 figures.** Re-run the roofline on an idle GPU.
2. **The MFU denominator is inconsistent.** This file said "keep 104 as the conventional denominator". `day2/bench_throughput.py` uses `PEAK_BF16_TFLOPS = 118.8`. Pick one before writing MFU tables.
3. **`RESEARCH_PROGRAM.md` does not exist in this repo**, so deliverable 2 has no target file yet.
4. **The power cap has a history.** `clocks_event_reasons_counters.sw_power_cap` read 711,320 s of accumulated SW power-cap time against 1,566,985 s of uptime (about 45%). It was not advancing at ~50 W. The power cap is therefore the first suspect for any drop in the first window.
5. **Telemetry available on driver 580.173.02:**
   - `clocks.sm` works: 2470 MHz under load, `clocks.max.sm` = 3003.
   - `power.draw.instant` and `.average` work. Module power and the enforced limit are N/A.
   - `temperature.gpu.tlimit` works. It reports headroom, which puts T.Limit at about 90–92 °C.
   - The throttle-reason field is `clocks_event_reasons.active`, alias `clocks_throttle_reasons.active`. Its per-reason microsecond counters also work.
   - Host ACPI zones are in `/sys/class/hwmon/*` (`acpitz`).
6. **The day2 sweep already pre-heats**, for 60 s by default (`--preheat-s`). Whether 60 s is enough depends on the τ measured here.

## Why it matters

1. **Your ceiling may be wrong.** The roofline matmul ran for well under a second. That is far shorter than the time the chassis takes to heat up, so it measured a cold chip.
2. **Throttling moves the ridge point.** Throttling lowers compute throughput $\pi$ but barely touches bandwidth $\beta$. If sustained $\pi$ is 70 TFLOP/s, the ridge $I^* = \pi/\beta$ drops from 381 to about 284 FLOP/byte. Your $d=768$ projections, which sat right at 381, would then be compute-bound, so the regime labels in the Day 2 table depend on this number.
3. **This box has a known history.** Early DGX Spark users reported power capping near 100 W, spontaneous reboots, and thermal problems under sustained load. Whether your unit does this is an empirical question.
4. **It belongs in the D1 note.** A burst-vs-sustained curve is exactly the kind of device characterization the planned "linear attention training on GB10" note is for.

## Background (the derivation, compressed)

Compute throughput is proportional to clock speed: $\pi = (\text{FLOPs per cycle}) \times f$. Firmware sets the clock $f$ dynamically.

Power grows roughly as the cube of the clock, $P \propto f^3$, because higher clocks also need higher voltage. So throttling the clock is the cheapest way for firmware to shed power.

Two limits act on the clock:

- **The power cap** acts within milliseconds, so it would already show up in a sub-second run.
- **The temperature limit** acts slowly. Temperature follows an RC circuit, $C_{th}\,\dot T = P - (T - T_{\text{amb}})/R_{th}$, where $C_{th}$ is how much heat the heatsink can absorb and $R_{th}$ is how hard it is to shed heat to room air. Temperature settles with time constant $\tau = R_{th}C_{th}$, which is tens of seconds to minutes. The sustained clock is the $f$ at which $T_{\text{amb}} + P(f)R_{th} = T_{\max}$.

A small chassis has a large $R_{th}$ (little fin area) and a small $C_{th}$ (little metal). That gives a lower sustained ceiling, reached sooner.

The same model makes the measurement self-checking. The heating curve gives $T_\infty$ and $\tau$. With the measured idle and load power, $R_{th} = (T_\infty - T_{\text{idle}})/(P_{\text{load}} - P_{\text{idle}})$ and $C_{th} = \tau / R_{th}$. The idle cooling curve gives a second $\tau$ with no throttling in the way. The two $\tau$ values should roughly agree.

## Prediction (write before running)

State the plateau TFLOP/s you expect, and your confidence that it lands within 5% of 93.9. Pass it as `--predict`. `summary.json` scores it, and the ledger line says whether you were right.

## Changes from the original protocol, and why

| Original | Now | Why |
|---|---|---|
| Nothing checked what else was on the GPU | Refuses to start if other compute processes exist (`--allow-shared` overrides and flags the run) | Contention, not heat, is the likeliest explanation for the 35.6 figure |
| Started whenever you pressed enter | Waits until GPU temperature is flat for 30 s (max 10 min) | If the box is already warm, the first window is not a burst |
| 5 s throughput windows | Synchronizes every ~0.1 s; burst = first 0.5 s, matching the roofline | The power cap acts in ms. A 5 s first window averages it away, which makes "drop in the first window" hard to see |
| `nvidia-smi -l 5` in a second terminal, on its own clock | Telemetry in-process at 250 ms, on the same `perf_counter` clock as throughput | Lets throughput be lined up against temperature, power and clock sample by sample |
| Mechanism read off the curve's shape | Also records the driver's clock-event reason bits, their μs counters, and the SM clock | Direct evidence of power cap vs thermal slowdown, not inference |
| 240 s soak, plateau by eye | 600 s soak; an exponential fit gives $\pi_\infty$ and τ; flagged "not converged" if 3τ > soak | 240 s only settles τ ≲ 80 s. The fit says when to run longer |
| No cooldown | 120 s idle tail, fit for cooling τ; $R_{th}$ and $C_{th}$ derived | Independent τ; device characterization for the D1 note |
| β re-check optional | 5 s copy benchmark before and right after the soak | The claim "β holds while π drops" is tested every run |
| Printed to terminal | CSV rows flushed per line, fsync'd every 2 s | "Crash or reboot" is a listed outcome; the data must survive it |

## Task

1. Stop other GPU jobs. `nvidia-smi --query-compute-apps=pid,process_name --format=csv` should list nothing.
2. Note the box's orientation and airflow.
3. Run:

   ```bash
   cd misc/burst
   ../../.venv/bin/python burst.py run --predict <TFLOP/s> --note "upright, vents clear"
   ```

   The run goes: copy (cold) → wait for idle temperature → 600 s matmul soak → copy (hot) → 120 s cooldown. Progress prints every 5 s. Ctrl-C ends early and still analyses what was recorded.
4. Read `results/<timestamp>/timeseries.png`, `mechanism.png`, and the printed summary (also in `summary.json`).
5. If a change to airflow or orientation is tried, run again with a new `--note`. Then overlay the runs:

   ```bash
   python burst.py compare results/<a> results/<b> --labels before after
   ```

6. If the verdict is `thermal_unconverged`, rerun with `--load-s` of at least 3× the reported τ.

Re-plot any run, including a partial one after a crash, with `python burst.py plot results/<timestamp>`.

## How to read the result

The script prints one verdict. The thresholds are heuristics and are stated so you can overrule them:

| Verdict | Rule | Meaning |
|---|---|---|
| `flat` | Plateau within 3% of burst, and ≥ 95% of 93.9 | No throttling. 93.9 stands as the sustained number |
| `flat_below_ref` | Within 3% of burst, but < 95% of 93.9 | Capped from the first millisecond, or contended. Check the reason bits and the contamination warning |
| `power_cap_step` | ≥ 3% drop, with < 25% of it still happening after 5–15 s | The power cap is binding |
| `thermal_decay` | The drop continues past 15 s, and the fit converged (3τ ≤ soak) | Thermal throttling. The plateau is your ceiling |
| `thermal_unconverged` | Decaying, but τ > soak/3 | Not at steady state yet. Run longer |
| `sawtooth` | Detrended residual (5 s smoothed) std > 3% of mean, with ≥ 6 zero crossings | The firmware keeps throttling and recovering. Report the plateau mean |
| Crash or reboot | Partial CSVs | The known issue. Fix airflow and orientation before trusting any long run |

Cross-check the verdict against the driver:
- `power_cap_step` should come with the SW power cap bit or its counter advancing, and throughput should be flat in temperature in `mechanism.png`.
- `thermal_decay` should come with temperature approaching T.Limit or thermal-slowdown bits, and throughput should fall as temperature rises.

If the shape and the driver disagree, say so in the note rather than picking one.

## Deliverable

1. **Plot:** `timeseries.png`, showing TFLOP/s, temperature, power, SM clock and clock-event reasons against one time axis. Add `mechanism.png` if the curve decays.
2. **`RESEARCH_PROGRAM.md` section 2:** create the file if it still doesn't exist. Take from `summary.json`:
   - `plateau_tflops`, labeled "sustained, `soak_s` s, heat-soaked, [date]"
   - `burst_tflops`
   - the recomputed ridge `ridge_flop_per_byte.sustained`, which uses the hot β from the same run
3. **MFU tables:** first settle 104 vs 118.8 as the conventional denominator (pre-flight item 2). Use the sustained plateau as the honest ceiling.
4. **Results ledger:** paste the `ledger` line from `summary.json`. It already includes whether the prediction was correct.
5. **Roofline:** if the run is uncontended, restore or re-measure `day2/roofline.json` (pre-flight item 1).

## Knock-on fix for the throughput sweep

Each benchmark configuration runs for only tens of seconds, inside the thermal transient. Configurations that run later in the sweep therefore run on a hotter box, which can look like a model-size effect.

The sweep already pre-heats for 60 s. If this measurement shows any decay, set `--preheat-s` to at least 3τ using the τ measured here, and log the run order. Pre-heating is better than cooling down between configs, because your real 3-hour training runs live at the hot steady state.

## Out of scope

Clock pinning, fan and firmware tuning, Nsight power traces. File them as KIV unless the plateau comes out below about 80% of burst.
