# Crystal NOTES (living handoff)

§3 of `art/ml-art-3d.md`: cuBLAS kernel choices of `torch.mm` over (m, n, k) ∈ [1, 256]³, drawn as a crystal.
Plan: `docs/superpowers/plans/2026-09-15-3d-pieces.md` §3. Sources: `hardware/lattice/` (read-only), `hardware/fingerprint/`.

## Stack
GB10 (sm_121a), driver 580.173.02, CUDA 13.0, torch 2.14.0+cu130, cuBLAS 13.1.1 (cublasLt 130101), bf16, TF32 off,
CUBLAS_WORKSPACE_CONFIG unset. gpu1.sh sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (lattice ran without it;
the k = 4096 slice check covers this).

## Files
- `common.py`: lattice's fingerprint method (masters, mats, 6-value fingerprint), lattice's profiler kernel-name pass
  generalised to (m, k, n), smi/stack; imports `hardware/lattice/common.py` (kernel_label) via importlib.
- `m1_calib.py` (GPU): phases load0, rate1, rate2, slice, kern, slab{3,16,64,127,128,256}, plane{127,128}, loadS, load1.
  Each phase -> `cache/m1_<phase>.npz`, skipped if present (resume = rerun).
- `m1_analyze.py` (CPU): `cache/m1_stats.json`, `cache/m1_mapping.md`, previews in `cache/preview/`.

## Decisions
- Decision: masters() verbatim from lattice for k >= 64; for k < 64 use nbig = 2*(k//4) cancelling big terms (0 for k <= 3) — lattice's construction indexes 64 distinct positions and fails for k < 64; half-or-fewer big terms keeps small terms to be absorbed so accumulation order still shows in the low bits.
- Decision: the fingerprint is lattice's 6-tuple (C[0,0], C[m//2,n//2], C[-1,-1] for two probes), without lattice's `nuniq` count — lattice's classes (`common.fp_labels`) use only the 6-tuple, and torch.unique would add a sync per shape.
- Decision: fingerprint classes are defined per k, as (k, 6-tuple) — the masters depend on k, so equal bits at different k mean nothing; cross-k identity of a region can only come from kernel names.
- Decision: k = 4096 slice sub-grid is m, n ∈ {1 + 4j + (j mod 4) : j = 0..63} — covers every residue mod 4 (parity and the 8-lattice), spans 1..256.
- Decision: besides the required 2000 uniform profiler samples (too sparse to test class purity per k: ~8 shapes per k), profile six dense (m, n) slabs k ∈ {3, 16, 64, 127, 128, 256} and two dense (m, k) planes n ∈ {127, 128} — gives exact class↔kernel purity and a first look along k, for ~10 min of GPU.
- Decision: load check = the same 2000 shapes fingerprinted at job start (load0), end (load1), in the kern phase, and under a self-made load (loadS: a thread on a second CUDA stream running 2048² bf16 matmuls); busy/idle state from nvidia-smi compute apps at each.

- Decision: M2 cube is dense [1, 256]³ — projected fingerprint + profiler-name wall time is 0.91 h (fp 64–86 µs/shape, names 115–147 µs/shape on dense slabs), far under the 6 h threshold; stride 2 would be 0.11 h.
- Decision: the crystal is coloured by profiler kernel label, not by fingerprint class — at k ≤ 256 the probe merges kernels: fp→kernel purity is 0.36 (k=256), 0.64 (k=128), 0.47 (k=64), and class 0 at k=256 holds both the align1 and align2 cutlass kernels, i.e. it erases the parity lamellae. Fingerprints stay as a secondary layer and for the k = 4096 slice check.
- Decision (for M2): use nbig = min(64, 2*(k//4)) for all k — identical to lattice for k >= 128 (so the slice check is unaffected), but fixes k in [64, 127], where lattice's 64 big terms leave 0–63 small ones (k = 64: every term big, sum exactly 0 in any order, one fingerprint class for 18 kernels).
- Decision: analysis labels use lattice's kernel_label after stripping 'std::enable_if<true, void>::type ' — lattice's short_kernel only strips the '!(false)' form, so one gemvx variant came out as 'enable_if<T>'.

## Status (M1 done 2026-09-15)
- m1_calib ran 04:38:34–04:40:40 (queued 43 min). No other compute process at any phase (GPU idle apart from this job).
- Rate (10⁴ random shapes, bf16): 63.6 / 62.1 µs per shape; rerun in an independent order: 100 % identical fingerprints.
- Slice: k = 4096 sub-grid 4096/4096 bitwise equal to lattice g256 (52 classes both).
- Kernel names: 2000 random shapes -> 27 labels; 104 of 563 (k, fp) classes hold > 1 label (584 shapes, 196 minority).
  Dense slabs: k=3 1 class/5 labels, 16: 4/15, 64: 1/18, 127: 6/11 (purity 0.96), 128: 10/25, 256: 12/28; 37 labels over all dense runs.
  66 sample shapes that also lie in a dense run: labels and fingerprints agree 66/66.
- Structure: odd k (3, 127) has no n-parity stripes (align1 everywhere); even k has them. A hyperbola-like boundary in (m, k)
  and (m, n) between align1 '+splitK' and plain align1 (at n = 127: k = 86 at m = 255, 101 at 200, 166 at 100; m·k not constant),
  label changes along m at m = 17, 33, 82, 95/97, 115/117, 129, 161 (k = 128/256, n = 200/201), small-m (m <= 16) gemmSN/gemv
  rows, and a boundary near k = 96 for m <= 16 (plane previews).
- Load check: load0, kern, loadS (11 427 concurrent 2048² matmuls, util 93 %), load1 all 100 % identical. No moment with
  another process on the GPU occurred, so the busy-by-another-process comparison is still open (M2 idle/busy window).
- Outputs: cache/m1_*.npz, cache/m1_stats.json, cache/m1_mapping.md, cache/preview/{slab*,plane*,slice_k4096_check}.png.
- Next (M2): cube = fingerprint + kernel names per k-slab (checkpoint per slab), bf16 then fp32; then timing 64³ needs idle window.

## Resume
```bash
cd /home/fzeng/ml/research/hardware/crystal
setsid nohup /home/fzeng/ml/research/art/_shared/gpu1.sh env OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python -u m1_calib.py > logs/m1_calib.log 2>&1 < /dev/null &
OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python m1_analyze.py
```

## M2 (started 2026-09-15 04:53)
Rulings from the controller: colour by profiler kernel names; dense bf16 cube with names + fingerprint per shape; fp32 cube
only if projected <= 1 h; load check at every slab; timing 64³ with GPU1_EXCLUSIVE=1 queued right after the cube.
- Files: `cube.py` (per k-slab checkpoints in `cache/cube_<dtype>/k<kkk>.npz`; slabs 32, 96, 160, 256 first, then 1..256;
  every slab re-profiles the first 200 M1 sample shapes and compares labels with `cache/m1_kern.npz`, and records nvidia-smi
  compute apps before/after), `timing.py` (lattice hygiene, checkpoint every 16 384 shapes), `run_m2.sh` (CPU driver: bf16 cube in <= 18-min
  gpu1.sh segments `cube.py --budget-min 18`, then timing as one GPU1_EXCLUSIVE=1 job, then fp32 segments with `--max-hours 1`),
  `m2_analyze.py` (label cube `cache/cube_bf16_labels.npz`, stats `cache/m2_stats_bf16.json`, `cache/m2_timing_stats.json`,
  previews `cache/preview/m2_*.png`).
- Decision: probe nbig = min(64, 2*(k//4)) from M2 on — identical to lattice for k >= 128, fixes k in [64, 127] (M1 note above).
- Decision: timing grid on each axis is v_j = 1 + 4j + (j mod 4) (64 values, spacing ~4) rather than {4, 8, ..., 256} — a pure multiple-of-4 grid has only even sizes and would sample none of the odd-n / odd-k lamellae; this grid holds every residue mod 4 and is the M1 slice sub-grid.
- Decision: timing reference shape (m, k, n) = (200, 200, 200) and overhead probe (1, 1, 1), re-timed every 256 shapes with GPU temperature and util; nvidia-smi compute apps every 4096 shapes.
- Decision: fp32 cube runs after the timing job, not before — the timing window is the scarce resource; the fp32 job measures slabs 32/96/160/256 first and writes `cache/cube_fp32/SKIPPED.json` if the projection exceeds 1 h.
- Decision: region counts use 6-connectivity over the full cube and, separately, inside each of the four (k mod 2, n mod 2) parity sub-lattices — in the full cube an even-k/even-n kernel is interleaved with odd-k/odd-n kernels, so its voxels are isolated lines and full-cube 6-connected counts mostly measure the interleave.

Resume M2:
```bash
cd /home/fzeng/ml/research/hardware/crystal
setsid nohup ./run_m2.sh > logs/run_m2.out 2>&1 < /dev/null &    # resumes per slab / per 16 384 timing shapes; log logs/run_m2.log
OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python m2_analyze.py
```
- 06:32 controller rule: no gpu1.sh job > ~20 min. The single queued cube job (never started) and chain_m2.sh were stopped and replaced by run_m2.sh (segments).

## PAUSED (2026-09-15 08:37, controller/user request)
- bf16 cube: **114 / 256 k-slabs done** (k = 1..112, 160, 256), all checkpoints load and validate (shapes, vocab indices, info).
  Segments 1–2 ran 07:01–08:04; segment 3 was queued, never started, and was killed along with the driver `run_m2.sh`.
  Load check so far: every slab 200/200 kernel labels identical to M1; no slab had another compute process present.
- Timing volume: **not run** (never queued). fp32 cube: not started.
- Nothing of crystal's is running or queued.
- Resume (continues from the per-slab checkpoints, then timing as one GPU1_EXCLUSIVE=1 job, then the optional fp32 cube):
```bash
cd /home/fzeng/ml/research/hardware/crystal
setsid nohup ./run_m2.sh > logs/run_m2.out 2>&1 < /dev/null &
# after the cube and timing finish:
OMP_NUM_THREADS=4 /home/fzeng/ml/research/art/.venv/bin/python m2_analyze.py
```
  Note: run_m2.sh names segment logs from seg01 again, so move `logs/cube_bf16_seg0*.log` aside before resuming if the old logs matter.
