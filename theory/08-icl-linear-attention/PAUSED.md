# PAUSED (2026-09-13, wind-down): theory #8, in-context regression with linear attention

All my background jobs are killed. Nothing is running.

## Done
- **Reproduction (one layer, ZFB):** exact gradient flow reaches W* to 4e-11 (correlated inputs) and 3e-12 (isotropic). Adam gets B within 0.8% (isotropic) and 39% (correlated). Risk vs M, covariate scaling, and random-covariance plateau / slope (0.556 vs 0.552) all match the closed forms. Figures: `figures/hero.mp4`, `lsa1_plate.png`, `randcov.png`, `mechanism.png`, `think.mp4`.
- **Depth:** deep LSA vs GD / preconditioned GD (`depth.png`, `cache/deep_eval.json`). Finding: heavy-tailed mean error. Noiseless proportional-limit formula γ^k(1−γ)/(1−γ^{k+1}).
- **Build-on (σ=0, L=1/2/4, all 4 mixers):** `race.png`, `fingerprint.png`, `shift.png` (a, b). Findings: 1-layer DeltaNet ≈ normalized LMS (gap 0.002; learned β≈0.95); delta-rule models 30–50× better than attention at L=2; deep delta nets are closest to ridge.
- **Widgets:** `widgets/explorer.js` (live MP closed forms) and `widgets/delta_widget.js`. Both self-tests pass.
- **post.md:** sections 1–4 and 6–8 written. Citations checked.

## Interrupted (rerun commands)
1. Task-diversity sweep, checkpoint at step 4000/8000 (resumes automatically):
   `OMP_NUM_THREADS=4 nohup ../_shared/gpu_run.sh ../.venv/bin/python taskdiv.py --logM 0,2,4,6,7,8,9,10,11,12,14,16,inf --batch 256 --steps 8000 --eval_every 2000 --tag main > logs/taskdiv_main.log 2>&1 &`
   (Do NOT use run_gpu_batch.sh as-is: it deletes the checkpoint.)
2. σ=0.5 sequence models: gdelta and delta are done; softmax and linear are missing (skips finished ones):
   `OMP_NUM_THREADS=4 nohup ../_shared/gpu_run.sh ./run_seq_grid.sh "softmax linear" > logs/seq_grid_C.log 2>&1 &`
3. Widget deep-LSA grid (CPU). The widget_s0 json is complete; widget_s05 is partial and skips done configs:
   `OMP_NUM_THREADS=4 ../.venv/bin/python train_lsa_deep.py --covs iso --kinds gd,dense --Ls 1,2,4 --ns 20 --sigmas 0.5 --steps 6000 --out widget_s05`

## Next steps
1. `python analysis_seq.py` (cached baselines), then `python render_seq.py` (shift panel c needs σ=0.5 models).
2. `python eval_deep.py && python build_widget_data.py`.
3. `python render_taskdiv.py --tag main`, then fill `«STEPS» «FRAC» «CAPTION_TD_ANIM» «CAPTION_TD» «TEXT_TD» «BREAK_TD» «TEXT_SHIFT» «CALLOUT_FINDING»` in post.md.
4. Render with `render_post.py --shot`, run the console-error check, review screenshots, commit, delete PAUSED.md, and write the final report.
