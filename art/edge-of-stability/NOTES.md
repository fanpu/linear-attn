# edge-of-stability NOTES (handoff)

## State (update at every commit)
- Compute: `eos_batch.py` trains M networks (one per 2/eta) in one process (block-diagonal HVPs).
  `eos_train.py` = older single-run prototype (kept, patched to float32 + faster HVPs; not used for gallery).
- float64 ~17x slower on GB10 (HVP 1.38 s vs 0.066 s) -> float32. GPU eigh of 3x3 took 0.19 s -> CPU.
- Init sharpness (seed 0, n=5000) = 88.0; 2/eta=20 diverges at step 31.
- scout.npz (killed at t~3200; 2/eta 50,80,120,200,300; eig every 10): EoS at 50 (lam1~53 from
  step ~50, catapult), 80 (lam1 ~83-85 from ~550), 120 (reaches edge ~1300, hovers ~125),
  200 (171 at 3200, rising), 300 (121 at 3200). lam2, lam3 also join the edge at 50 and 80.
  eig-every 10 creates visible kinks in x at refresh -> main runs use eig-every 1.
- Queued via gpu_run (logs/main4.log, logs/sweep16.log):
  main4.npz : 2/eta 50,80,120,200; 6000 steps; eig every 1; sketch 1024
  sweep16.npz: 2/eta 30..250 (16); 5000 steps; eig every 5
- Renderers written & tested on scout: render_print.py (hero three-layer; night/paper/riso/spectral),
  render_phase.py (orbit plane + return map; needs dense data to judge), render_sweep.py (Spectral
  edge heatmap, phase heatmap, score staves), render_plotter.py (single-line SVG A2 + proofs),
  render_film.py (MP4+GIF with magnifier; magnifier is the best bit).
- Declared choices so far: mask x for t<40 (moving-average bias during the initial fast loss drop).

## Paper setup (checked in arXiv:2103.00065 text)
first 5000 CIFAR-10 train, per-channel standardise with full-CIFAR stats, 3072-200-200-10 tanh,
PyTorch default init, MSE 0.5*sum_classes averaged over examples; Fig 1 FC legend lists
2/eta in {20,50,80,110} (panel attribution from text extraction uncertain); Fig 3 uses 2/600.

## Resume
cd art/edge-of-stability; P=../.venv/bin/python
$P render_print.py main4.npz 1 eta80      # etc; film: $P render_film.py main4.npz 1 eta80 5
## Next
when main4/sweep16 finish: render all, verify (cold checks vs warm, residuals, crossing vs
eigenvector-overlap audit), iterate on strongest, README.
