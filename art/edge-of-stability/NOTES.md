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
- RUNNING via gpu_run (logs/main4.log, logs/sweep16.log); partial saves are loadable anytime:
  main4.npz : 2/eta 50,80,120,200; 6000 steps; eig every 1; sketch 1024 (~1.2-1.7 s/step, started
              ~13:05, expect done ~15:30). GPU contention is heavy (8 slots full).
  sweep16.npz: 2/eta 30,40,50,...,250 (16); 5000 steps; eig every 5; sketch 512 (queued ~13:45).
  Check: grep -v -i warn logs/main4.log | tail -1
- Renderers written & tested on scout: render_print.py (hero three-layer; night/paper/riso/spectral),
  render_phase.py (orbit plane + return map; needs dense data to judge), render_sweep.py (Spectral
  edge heatmap, phase heatmap, score staves), render_plotter.py (single-line SVG A2 + proofs),
  render_film.py (MP4+GIF with magnifier; magnifier is the best bit).
- Declared choices so far: mask x for t<40 (moving-average bias during the initial fast loss drop).

## KEY FINDING (main4 partial, t<=1000, eig every 1)
- Braid along the current top eigenvector u1 has abrupt sign flips: 69-76% of its 'crossings' sit
  within 2 steps of an eigenvector identity swap (|<u1(t),u1(t-1)>| < 0.95; 25% of steps at 2/eta=50,
  6% at 80; lam1~lam2 near-degenerate at the edge). They are coordinate artefacts.
- Windowed-PCA coordinate from the count-sketch (eos_common.pca_braid: detrend with 21-step centred MA,
  window 64 hop 32, sign-aligned, triangular blend) gives a smooth braid: 0-1 crossings after the
  edge (a genuine slow phase slip at ~940 for 2/eta=80), quasi-periodic bursts ~50-75 steps apart;
  median |cos(PC, u1)| = 0.88 (80), 0.81 (50). corr(x_u1, pca) ~ 0 because of the flips.
- DECISION: all renderers now use braid(d, m) = PCA coordinate by default (EOS_COORD=u1 env for the
  u1 version -> make a diptych 'rotating frame vs data frame' as an honesty plate).
- TODO labels: render_print/film/plotter captions still say <theta - thetabar, u1(t)>; change to
  'windowed-PCA oscillation coordinate' when EOS_COORD=pca. Docstrings too.
- verify.py (writes cache/<run>_verify.json): warm-vs-cold sharpness max rel err 3.7% (50), 3.9% (80);
  RR residual median 14% at 2/eta=50 (near-degenerate top-3, 1 iteration/step) vs 0.2% at 80.
  t_edge 405 for 2/eta=80 (init sharpness 88 starts above 50 and 80; catapult first).
  loss rises on 26-35% of EoS steps yet falls overall. Add PCA-based crossing/burst stats to verify.py.

## Paper setup (checked in arXiv:2103.00065 text)
first 5000 CIFAR-10 train, per-channel standardise with full-CIFAR stats, 3072-200-200-10 tanh,
PyTorch default init, MSE 0.5*sum_classes averaged over examples; Fig 1 FC legend lists
2/eta in {20,50,80,110} (panel attribution from text extraction uncertain); Fig 3 uses 2/600.

## Resume
cd art/edge-of-stability; P=../.venv/bin/python
$P render_print.py main4.npz 1 eta80      # etc; film: $P render_film.py main4.npz 1 eta80 5
## Next
0. fix captions (above). 1. render print (m=1 eta80 hero; also 120/200), 4-LR small multiples
   (new render_multiples.py, not written yet: 4 stacked prints, absolute lambda axis), film eta80 spf 5,
   plotter eta80 200/row, sweep plates, orbit plates, u1-vs-PCA diptych.
2. README.md (not written yet) per BRIEF. 3. Delete cache/frames after films. Commit.
when main4/sweep16 finish: render all, verify (cold checks vs warm, residuals, crossing vs
eigenvector-overlap audit), iterate on strongest, README.
