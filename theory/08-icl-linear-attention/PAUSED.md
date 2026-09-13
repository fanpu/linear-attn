# PAUSED (2026-09-13 ~04:05) — theory #8, in-context regression with linear attention

No background jobs are running (all training/render processes killed).

## Done
- `icl_core.py` + `test_core.py` (all pass): ZFB Γ, W*, exact risk of preconditioned GD-1 (matches MC and ZFB Thm 4.2 to 1e-10), ridge/RLS (equal to 1e-14), GD-k, LMS, dMMSE (now batch-chunked; the unchunked version OOMed), MP quadrature.
- `seqmodels.py` + `test_seqmodels.py` (all pass): softmax / causal-mean linear attn / DeltaNet / Gated DeltaNet. Pure-torch recurrence == exact parallel triangular-solve form (1e-16) == fla chunk kernel (3e-4 fp32; needs `TRITON_F32_DEFAULT=ieee`, else 1.5e-3 from TF32). fla chunk kernel and the python loop are ~0.45 s/step at these tiny sizes under GPU contention; the triangular-solve form (`impl="par"`, default) is ~75 ms/step.
- `lsa_gradflow.py`: exact population gradient flow from ZFB init converges to W* (iso, ar1: |W−W*|/|W*| ≈ 1e-11); `randexp` (ZFB §4.3) also done → `cache/gradflow_*.npz`.
- `train_lsa1.py` (Adam, full matrices free): iso converges (B_eff rel err 0.02 at 2k steps); ar1 at 16k steps is 2.4% above L*, B_eff rel err 0.39 (slow low-variance directions) → `cache/lsa1_*.npz`. Plain SGD diverged (noted in docstring).
- `render_think.py` → `figures/think.mp4`, `figures/think.gif` (d=2 "watching it think", seed 163). Not yet reviewed after the last tweak.
- `widgets/delta_widget.js` (DeltaNet vs Hebbian state, live JS, `#selftest` passes). CSS for it is in `_wtest.md` (untracked test page).
- Build-on models finished: `cache/seq_softmax_L{1,2}…s0.0`, `cache/seq_delta_L1…s0.0`.

## Interrupted (rerun exactly)
- Task-diversity sweep crashed at its first eval (OOM, now fixed) — nothing saved. Rerun (now checkpoints every eval, resumes automatically):
  `cd theory/08-icl-linear-attention && OMP_NUM_THREADS=4 nohup ../_shared/gpu_run.sh ../.venv/bin/python taskdiv.py --logM 0,2,4,6,7,8,9,10,11,12,14,16,inf --batch 256 --steps 15000 --eval_every 2500 --tag main > logs/taskdiv_main.log 2>&1 &`  (~2 h under contention)
- Build-on grid (skips finished models; delta L2 was at step 4000/8000, no checkpoint):
  `OMP_NUM_THREADS=4 nohup ../_shared/gpu_run.sh ./run_seq_grid.sh "softmax linear" > logs/seq_grid_A.log 2>&1 &`
  `OMP_NUM_THREADS=4 nohup ../_shared/gpu_run.sh ./run_seq_grid.sh "delta gdelta" > logs/seq_grid_B.log 2>&1 &`

## Next steps (ordered)
1. Relaunch taskdiv main + seq grids (above).
2. `train_lsa_deep.py`: test dense-LSA trainability (300 steps was not enough), then `--grid main` and `--grid widget` (consider float32 / fewer steps; ~100 ms/step float64).
3. Renders: hero (dark, W_KQ/W_PV crystallizing + loss onto L*, from gradflow_ar1 or lsa1_iso), LSA-vs-W* plate, risk vs M + covariate-scale shift plate, randexp "1/3 slope" plate, depth plate, task-diversity plate + animated M sweep, algorithm-race small multiples, noise/cov-shift for seq models (eval script still to write).
4. Widget 1 (algorithm explorer, MP-limit ridge/GD-k live in JS + trained LSA dots), polish delta widget.
5. Write post.md, README.md; render with `_shared/render_post.py --shot`; console-error check; citation checks (WebSearch); commit at each checkpoint.
