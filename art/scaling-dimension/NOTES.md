# scaling-dimension NOTES (living handoff)

Python: `/home/fzeng/ml/research/art/.venv/bin/python` (call it PY). Run everything from this dir.

## State (2026-09-13 14:37, agent 1 handing off)
### Background jobs (left running on purpose; check with `ps aux | grep -E "ts_train|real_train"`)
1. **Main T/S sweep** -> `cache/main/ts_main.npz` (saved after each width; log `cache/logs/ts_main.log`).
   fams relu0 (paper teacher, zero bias) + relub (biases N(0,.1^2)); d=2,3,4,5,6,8,10,12; 3 seeds;
   widths in order 6,16,45,4,11,32,90,8,23,64 (N=199..~10.8k); 60k steps, batch 1024, Adam 3e-3 then cosine->3e-6 over last 60%.
   ~4 min/width, expected done ~15:10. Widths 6,16 done at handoff.
2. **Real-data CNN sweep** (queued behind gpu_run.sh slot at handoff; log `cache/logs/real.log`, empty until slot acquired)
   -> `cache/real/real.json` (resumable: skips done keys). cifar10,fmnist,mnist; widths c=2,4,8,16,3,6,12,24; 12 epochs, batch 256, 1 seed;
   per width: best test loss (early stopping), err, final-hidden TwoNN/MLE ID; plus pixel ID (10k imgs). Toy MNIST pixel ID: TwoNN 13.9, MLE10 12.7.
   If it never gets a slot: run it directly on CPU is too slow for cifar; consider `--data mnist,fmnist --widths 2,4,8,16` or wait.

### Scripts
- `idlib.py` TwoNN/MLE/N(r). **Fixed bug**: neighbour_count_curve subtracted self once per chunk instead of per centre.
- `ts_train.py` batched sweep (edited: shared pools per (fam,d), per-width incremental save, --mem_frac).
- `ts_analyze.py NPZ --id_widths 16,45,90` -> `NPZ_analysis.{json,npz}`: alpha fit per fam_d (median over seeds, bootstrap CI, local slopes),
  IDs of students' final hidden layer (12k pts; TwoNN, MLE k=5,10,20), N(r) curves (radii in units of median NN dist), input-ID sanity.
  Run on GPU briefly (<2 min, fine without slot) once sweep done: `PY ts_analyze.py cache/main/ts_main.npz --id_widths 16,45,90`.
  Toy (1000 steps) numbers: student ID d=2:1.98, d=6:5.6, d=12:9.8 (TwoNN; MLE lower at high d, known underestimate).
- `style.py` styles paper (log paper, cream + orange ruling + one ink), dark (magma), riso (Federal Blue + Fluo Pink, misregistration), spectral.
- `render_plates.py --analysis cache/main/ts_main_analysis --pieces diptych,fan --styles paper,dark,riso,spectral --idw 45`
  diptych works (checked on toy, looks clean). **fan() is a rough draft** (upper N(r) fan hack with a dead `if False` expression) - rewrite.
  **agreement plate not yet written**: x = median student TwoNN d (per fam,d; error bar = seed/width range), y = 4/alpha (CI from bootstrap),
  y=x line, relu0 filled / relub hollow; real-data points from real.json (fit alpha over c widths of test_loss vs N; x = final-hidden ID median,
  also pixel ID as secondary hollow marker); GPT-2 annotation off-scale: 4/alpha~53, d>90 (arrow, from paper, not measured here).
- `zoom_compute.py` + `render_zoom.py`: **the zoom film** (prototype works, looks strong). d=2 relub teacher graph + students (widths 6,16,45,90, seed 0)
  sampled on the same K random offsets in a window shrinking 10^-decades around z0; heights relative to teacher's tangent plane at z0, divided by rho
  (isotropic zoom). Teacher creases -> flattens to a plane; students peel away in order of N (bigger N stays glued longer = the scaling law, visible).
  Final commands (after sweep has w45,w90):
  `PY zoom_compute.py --npz cache/main/ts_main.npz --fam relub --widths 6,16,45,90 --frames 900 --K 80000 --decades 2.5 --rho0 0.2 --out cache/zoom/zoom.npz`
  `OMP_NUM_THREADS=1 PY render_zoom.py --data cache/zoom/zoom.npz --style dark --out gallery/zoom_dark` (also paper, riso). Check stills first with `--frames 0,300,600,899 --size 720`.
  Tune: kappa 0.5, slab 0.9 (new clip, untested), maybe add inset L(N) line with markers lighting as each student peels off.

## Doc-claim check (GPT-2 d>=90) - done, goes in README
Sharma&Kaplan JMLR 2022 sec 3.3 (refs/jmlr.txt ~l.1157-1200): GPT-2 small, alpha=0.076 -> 4/alpha~53; ID from last-token activations,
10k vectors, every layer (attn/FC/residual): ID roughly constant across layers **except first layer, significantly smaller (50-80, which matches 4/alpha)**;
"since d > 90, d >= 4/alpha ~ 53". IDs from 1024 tokens of one passage: ~7. So doc claim is essentially right but omits the first-layer exception and the
authors' note that ID estimators underestimate for d >~ 20 (appendix C). Paper setup: teacher [20,600,600,1] MSE, k features zero-padded (we embed via random
orthonormal Q into D=24, equal in distribution), students [20,n,n,1], 240k steps growing batch; ID from final hidden layer, 12k vectors; best 9 of 10 trials.

## Remaining pieces / todo
1. analyse main sweep; inspect local slopes (low-d curves may saturate at big N -> restrict fit range, report).
2. render diptych (4 styles), rewrite fan, write agreement plate (4 styles), zoom film (dark/paper/riso MP4+GIF).
3. real-data points; README (brief's 8 sections) with metrics, caveats (MLE underestimates high d; alpha depends on fit range; relub vs relu0).
4. commit via `/home/fzeng/ml/research/art/_shared/commit.sh scaling-dimension "msg"`.
