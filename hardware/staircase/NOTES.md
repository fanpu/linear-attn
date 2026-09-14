# Staircase NOTES (handoff)

- `chase.c` -> `./chase` (gcc -O2). `run_all.sh` = the full campaign (cpus 0,5,10,15 x {4K,THP} random 1K..1G; seq nulls; node-8 on
  cpu 0,5 to 256M; heartbeats L1 32K x 1e6 chunks and 64M x 3e5 chunks on cpu 0,5). Took 1 h 38 min on 2026-09-14 (10:07-11:45),
  GPU idle. Random runs ~10 min each (the DRAM octaves dominate: 20M loads x 5 reps x ~130 ns).
- `render_stair.py [--only staircase heartbeat] [--prefix stair --hb hb]` -> gallery. Main plate skips node != 64 runs and has a
  compact legend (per core + line-style proxies); `plate_nodesize` is the 8-byte diagnostic; heartbeats fold at 1 ms (L1) or 64 ms
  (DRAM chunks ~240 us each, auto-selected when median chunk > 10 us).
- Render pinned away from the measured core (`taskset -c 16-19` while cpus 0/5 run, `-c 1-4` while 10/15 run).
- Numbers: L1 64K on all cores (1.43 ns A725 / 1.03 ns X925 = 4 cycles); L2 steps at 512K / 2M; plateau ~5.6 ns; A725 leaves it at
  12 MB, X925 at ~40 MB (follows core type, not sysfs cluster L3 8M/16M); THP -20..25 ns at 1 GB and -1.6 ns on the plateau;
  DRAM 115-160 ns; seq null flat 1.6-2.0 ns. X925 64K-512K alternation 1.2 vs 2-2.8 ns by size, reproducible, THP-independent,
  unexplained. Heartbeat: one slow chunk every 1.000 ms (+1.6 us A725, +1.06 us X925) = CONFIG_HZ=1000 tick.
- Status: COMPLETE (README, gallery, index row). Extras if wanted: perf-counter confirmation of the tick (not traced), TLB-size
  sweep with 1 line per page, a plate of the X925 alternation vs size modulo set count.
