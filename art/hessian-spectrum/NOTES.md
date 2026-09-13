# hessian-spectrum NOTES (living handoff)

## State (session 3, 2026-09-13): deliverable COMPLETE
No background jobs. GPU used: 0 (all CPU). Python: /home/fzeng/ml/research/art/.venv/bin/python, run from project dir.

## Key result
Outliers = C-1, not C, in all 12 networks (exact small MLP + Lanczos 118k MLP; C=2,3,4,5,7,10 -> 1,2,3,4,6,9).
Count = leading H eigenvectors with ||proj onto span{d_c}||^2 > 0.5; k-th overlap >= 0.64, (k+1)-th <= 0.30,
so threshold-robust. Widest-log-gap count fails for C>=5-7 (C=10 gap says 8; Lanczos C=5 says 1).
Why C-1: class-mean directions anticorrelated (mean cos -0.62..-0.07 vs simplex -1/(C-1)); G1 C-th eig only
0.18-0.55 of (C-1)-th; never separates in H. Film (C=10): 1 line through step 17, 4/5/7 at 19-28, 9 from step 32
(except step 2135: 8); lambda_max 0.47 -> 16.3 at step 55 -> 3.3 final.
Papyan/Sagun abstracts don't state C vs C-1 explicitly (checked arXiv abs); README says so.

## Files
- compute: common.py, train.py, analyze.py (Lanczos+SLQ), exact_analyze.py, run_*.sh
- render: render_common.py, render_plates.py (emission/absorption/silver/negative, --src exact|lanczos),
  render_ladder.py (riso/night), render_barcode.py (barcode paper/night + gel), render_film.py
  (--stills styles hue,silver,magma,paper,split,aurora_ember,cyanotype_vandyke; --film --film_style hue|split),
  render_slq.py (verify_slq_kernel.png), verify.py (-> cache/verify.json, cache/verify_table.md)
- README.md full (verify table, SLQ m=80 x 4 Rademacher probes, kernels, symlog taus, caveats, refs).

## Decisions
symlog tau 2e-3 plates/ladder, 1e-4 barcode/gel/spectrograph. Spectrograph tone 0.6*expo + 0.4*log density
(pure exposure saturated bulk). Split styles: palettes.render_split on sign(lambda)*log1p(density).

## Optional next steps (not required)
FashionMNIST C=10 exact check (`train.py --ds FashionMNIST --arch mlps --down 10 --C 10 --epochs 10 --lr 0.05
--name f_mlps_C10` then `exact_analyze.py --run f_mlps_C10`); multi-seed; Papyan hierarchy plate (H vs G vs E).
