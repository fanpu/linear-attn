# Number Knot (§6) — M1 report: hidden states and fits

**Status: DONE.** Commit `1cc0062` (art/number-knot: code + NOTES). Date 2026-09-15.
Periodic number structure in OLMo-2-0425-1B and calendar circles in Qwen3-0.6B both beat their nulls
clearly. The caveats matter for M2 captions: the number circles are small (each explains 1.5–5% of the
100-PC variance), the T=10 structure is mostly *residue clusters* rather than a clean circle, and on the
K&T range 0–99 alone only T=100 clears the nulls, where it can't be told apart from smooth curvature.

## What was done
1. **Templates and token checks** (`common.py`). Every prompt is built at the id level as `prefix_ids + [target_id]`,
   and the state is read at the target (the last position). The prefix is asserted identical for all targets.
   - OLMo-2 (BOS `<|endoftext|>` prepended, since the tokenizer adds none): `{a}`, `The number {a}`, `x = {a}`,
     `Output ONLY a number. {a}`. The number is asserted to be one token for all 0–999 in every template.
     OLMo-2's pretokeniser emits a separate space token (220) before a number, and that is kept because it is the
     in-distribution form. The plan's `{a}+` was replaced: in a causal model the state at the number token is
     identical to `{a}`'s.
   - Random-token null: 1000 OLMo-2 vocabulary tokens that decode to ≥2 ASCII letters with no leading space and
     round-trip as one token (27,277 candidates, seed 0). They go in the number slot of the same id-level prompts,
     with labels 0..999 in random draw order.
   - Qwen3-0.6B: 7 days and 12 months as ` <Word>` single tokens, 24 declared templates each (e.g. `Today is{w}`,
     `She was born in{w}`). The full-string tokenisation is asserted equal to prefix + [target] for every pair.
2. **Forward passes** (`extract.py`). Forward hooks capture the embedding output plus every block output, so the
   last layer is the residual *before* the final norm (HF `hidden_states[-1]` is post-norm; checked, the
   pre-norm entries match to 1e-3). Compute was fp32, storage float16 in `cache/hs/`, max |h| = 44 (OLMo) and
   340 (Qwen). Shapes are OLMo (1000, 17, 2048) per template × {numbers, random} and Qwen (K, 29, 1024) per template.
   **Run on CPU** (4 threads, 526 s total) instead of the GPU queue. `gpu1.sh` was queued at 02:14 but
   autonomous/ jobs held the lock back to back. The queued job was cancelled before it started, and a declared
   CPU run is allowed under 45 min.
3. **Analysis** (`fits.py`, `analyze.py`, 146 s, 200 permutations). The K&T recipe was checked against
   arXiv:2502.00873: PCA to 100 dims, then linear regression PCA(h_a) = C·B(a), with B = [a, cos 2πa/T, sin 2πa/T]
   (raw a) and T ∈ {2, 5, 10, 100}. K&T use the output of layer 0 (our index 1) and activation patching; they
   report no R². Here, for each (template, layer, range 0–99 / 0–999, T):
   - R² is variance-weighted over the 100 PCs, and ΔR²_T = R²_T − R²([1, a]) is the circle's own share.
   - Nulls: (sh) shuffled a on the number cloud and (rt) the random-token cloud with random labels, both through
     the identical pipeline, with the same 200 permutations everywhere. Family-wise 99% = 99th percentile of the
     max over layers.
   - Fourier power over a (K&T/Zhou: centred, spectra summed over the PCs, with and without the linear trend), with
     a 50-shuffle 99% envelope.
   - Two comparators the plan did not list, added because shuffled labels can't separate a circle from other
     structure:
     - **mod-m one-hot** [1, a, one-hot(a mod m)] holds all residue-class structure. A set of residue clusters
       with no circle gives each circle a share of 2/(m−1).
     - **poly3** [1, a, a², a³] is smooth, non-periodic, and has the same column count as a circle.
   - Days and months: supervised mean-difference plane (top-2 PCs of the class means; not LDA, and not SAE
     clustering as in Engels et al.), then the circle regression P = c + A[cos 2πk/K, sin 2πk/K].
     - Scores: in-sample R², and held-out R² (plane and circle fitted on even templates, scored on odd, and swapped).
     - Nulls: 200 class-order shuffles, 200 point-label shuffles, and for days the exact enumeration of all 360
       cyclic orders up to rotation and reflection.
4. **Previews** (`preview.py`, matplotlib), all viewed.

## Commands
```
cd art/number-knot
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest -q test_fits.py              # 6 passed
OMP_NUM_THREADS=4 ../.venv/bin/python extract.py --device cpu --tiny          # smoke test, 20 s
setsid nohup env OMP_NUM_THREADS=4 ../.venv/bin/python extract.py --device cpu > logs/extract_cpu.log 2>&1 < /dev/null &   # 526 s
OMP_NUM_THREADS=4 ../.venv/bin/python analyze.py                              # 146 s
OMP_NUM_THREADS=4 ../.venv/bin/python preview.py --template 1 --layer 1
```

## Key numbers
**OLMo-2 numbers, 0–999** (best layer per period):
- Every period beats both family-wise nulls by 5–10× in every template, and by >4× at every layer ≤ 14.
- Chosen cell (`The number {a}`, L1): ΔR² is T=2 0.015, T=5 0.030, T=10 0.030, T=100 0.048. Family-wise null
  99% is 0.003–0.005 (ratios T=10 6.7×, T=100 9.4×).
- Raw R² at that cell: R²_lin 0.085, R²_T100 0.133.
- Fourier (0–999, L1) has sharp peaks, all ≥10× the shuffle envelope (0.003):
  - T=100 0.052 and its harmonics T=50 0.042, T=25 0.020;
  - T=10, 5, 2 at 0.033–0.035, T=2.5 0.025, T=3.3 0.013.
- Circle vs. clusters (chosen template, 0–999):
  - Mod-10: T=10 share 0.24–0.27 and T=5 0.27, against 0.22 for residue clusters with no circle; T=2 0.14–0.21
    against 0.11. So the mod-10 structure is **mostly clusters, only mildly circular**, and T=2/5/10 carry about 70%
    of it because T=3.3 is weak.
  - Mod-100: T=100 share 0.12–0.17 against 0.02 for clusters (≈0.24 after subtracting the one-hot's own chance
    level of 0.099). This is **genuinely circular**.
- Decoded helix angles at the chosen cell have a median angular error of 10° (T=100), 11° (T=10) and 13° (T=5).
  This is in-sample and optimistic. The shuffled-label decode is an isotropic blob (see preview).

**OLMo-2 numbers, 0–99** (K&T's range):
- T=2/5/10 reach only 0.9–1.8× the family-wise null in ΔR².
- The Fourier peaks at T=10 and T=5 exceed the shuffle 99% envelope (L1: 0.048/0.047 vs 0.033/0.031); T=2 barely does (0.047 vs 0.043).
- T=100 clears (3.2–3.5×, ΔR² 0.13–0.16) but equals poly3 at the same layer (0.126 vs 0.126; 0.160 vs 0.162).
  Over a single period, cos/sin is indistinguishable from curvature, so there is **no T=100 circle claim on 0–99**.

**Qwen3-0.6B days**:
- Held-out R² is 0.77–0.92 across layers 0–26 (L27 0.69, L28 0.50). The best is L12 at 0.920, against order-shuffle
  99% 0.685, point-shuffle 99% −0.054, and family-wise order 0.740.
- Exact order null: the true order ranks 1 of 360 at 27 of 29 layers and 2 at L10 and L22.
- Angular order of the class means is exactly Mon→Sun at 28 of 29 layers. The circle is already in the embeddings
  (L0 0.866).

**Qwen3-0.6B months**:
- Held-out R² 0.70–0.92. The best is L13 at 0.899, against order-shuffle 99% 0.438 (max of 200 is 0.512) and
  point-shuffle 99% −0.03.
- R² exceeds the maximum order-shuffle at all 29 layers.
- Exact cyclic order holds at 18 of 29 layers; the failures are L1–2, L4–10 and L27–28, where Jul/Aug/Sep bunch up.

## Chosen layer and periods, and why
- **Numbers → template `The number {a}`, layer 1** (output of block 0, K&T's h⁰), fit on 0–999.
  - Rule (declared): largest geometric mean of the family-wise margin ratios for T=10 and T=100, the knot's two
    periods, over templates and layers ≥ 1. L1 scores 7.96; the runner-up is L12 at 6.99, which has a marginally
    larger minimum ratio (6.74 vs 6.71).
  - L1 is also where T=100 is strongest and its circle share highest.
- **Helix period: T=100.** It is the strongest (9.4×) and the only one with clearly circular geometry. T=10 is the
  second period for the knot; its ring of 10 clusters is visibly ordered, but the M2 caption must say it is mostly
  clusters.
- **Days → L12 and months → L13** for single-plane previews (argmax of held-out R² minus the larger null 99%). The
  depth towers use all 29 layers.
- The margins are flat across depth, so a layer sweep (M3 film) shows persistence and slow decay rather than
  "condensing out of noise". The Qwen calendar towers do have a visible dip and recovery (days L10, months L1–10).

## Previews (all opened and checked)
- `art/number-knot/cache/preview/helix_t1_L1.png`: measured T=10, T=100 and T=5 planes against the shuffled-label
  fit. The measured planes show ordered rings of residue colours; the null is a blob about 5× larger.
- `art/number-knot/cache/preview/helix_t1_L12.png` and `helix_t0_L0.png`: the alternative layer and the embedding.
- `art/number-knot/cache/preview/fourier_t1.png`: spectra over a for 0–99 and 0–999, with shuffle 99% and
  random-token curves.
- `art/number-knot/cache/preview/numbers_margin_heatmap.png`: ΔR² and the per-cell margin for all templates, layers
  and periods.
- `art/number-knot/cache/preview/numbers_curves_t{0..3}.png`: ΔR² vs layer with null lines.
- `art/number-knot/cache/preview/calendar.png`: held-out R² vs layer with nulls, and day/month planes at the best
  and last layers.

Cache (572 MB, gitignored):
- `cache/hs/*.npy`
- `cache/fits_numbers_{100,1000}.npz`, which includes decoded helix coords for T ∈ {5, 10, 100} at every
  template × layer
- `cache/fits_qwen_{days,months}.npz`, which includes plane projections for every layer
- `cache/tables_M1.md`, `cache/summary_M1.json`

## Decisions
All are logged in `art/number-knot/NOTES.md` as `Decision: … — …`. In summary:
- replaced `{a}+`;
- prepended BOS for OLMo-2 and kept the space token;
- no BOS for Qwen3;
- defined the random-token set;
- ran on CPU in fp32 with float16 storage;
- captured pre-final-norm hooks;
- defined ΔR² and the family-wise nulls, and the "clear" rule (ΔR² ≥ 2× the larger family-wise 99% and ≥ 0.01);
- used the mean-difference plane with the held-out-template score;
- added the mod-m and poly3 comparators;
- used the exact day-order null;
- chose the primary cell by rule;
- used cyclic `twilight` for residues in previews.

## Risks and open issues
- **Small effect sizes.** A circle explains 1.5–5% of the 100-PC variance, with R²_T ≤ 0.22 in total. The helix
  render is the measured points decoded into the fitted directions: a least-squares projection that is optimistic
  in-sample. M2 should show the null panel at the same scale and print ΔR² and PCA variance (PCA-100 captures
  45–94% of the 0–999 cloud by layer).
- **T=10 is clusters, not a helix.** An honest M2 caption says "ten residue clusters, arranged in order around the
  fitted T=10 plane". The T=100 × T=10 knot is then a composition of a genuine circle and an ordered cluster ring.
- **0–99 comparability.** At N=100 the K&T periods don't clear the family-wise nulls, so the "K&T in a 1B model"
  claim has to be made on 0–999.
- **Template dependence is weak but present.** `x = {a}` has the lowest margins; layer 0 is identical across
  templates by construction.
- **Calendar method.** The mean-difference plane is supervised. Order is not used to find the plane, only the
  class labels, so an exact cyclic order is an independent signal. Days have only 7 classes, so the random-order
  null is high (0.69), which is why the exact 360-order rank is reported.
- **Reproducibility.** CPU and GPU would differ by float noise only; the hardware caption for M2 should say
  "CPU (GB10 Grace), fp32, torch 2.14.0+cu130", not CUDA.

---

# Appendix: full generated tables (`art/number-knot/cache/tables_M1.md`)



### OLMo-2 numbers 0–99, template 0: `{a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.106 (0.017) | 0.126 / **0.020** (0.017, 0.011) | 0.148 / **0.042** (0.027, 0.021) | 0.149 / **0.043** (0.033, 0.021) | 0.214 / **0.107** (0.029, 0.022) |
| 1 | 100 | 0.138 (0.019) | 0.157 / **0.019** (0.019, 0.012) | 0.178 / **0.040** (0.030, 0.022) | 0.178 / **0.040** (0.036, 0.022) | 0.260 / **0.122** (0.031, 0.022) |
| 2 | 100 | 0.164 (0.021) | 0.183 / **0.019** (0.020, 0.012) | 0.203 / **0.039** (0.031, 0.022) | 0.202 / **0.038** (0.037, 0.022) | 0.299 / **0.135** (0.033, 0.022) |
| 3 | 100 | 0.212 (0.024) | 0.230 / **0.018** (0.023, 0.012) | 0.249 / **0.037** (0.033, 0.023) | 0.248 / **0.036** (0.043, 0.022) | 0.360 / **0.148** (0.036, 0.023) |
| 4 | 100 | 0.212 (0.024) | 0.231 / **0.018** (0.023, 0.013) | 0.250 / **0.037** (0.034, 0.023) | 0.248 / **0.036** (0.042, 0.022) | 0.360 / **0.148** (0.037, 0.023) |
| 5 | 100 | 0.208 (0.023) | 0.227 / **0.019** (0.022, 0.013) | 0.247 / **0.039** (0.035, 0.023) | 0.244 / **0.036** (0.039, 0.023) | 0.357 / **0.149** (0.037, 0.024) |
| 6 | 100 | 0.217 (0.024) | 0.237 / **0.020** (0.022, 0.013) | 0.256 / **0.039** (0.035, 0.024) | 0.253 / **0.036** (0.039, 0.023) | 0.366 / **0.150** (0.037, 0.024) |
| 7 | 100 | 0.215 (0.024) | 0.238 / **0.023** (0.023, 0.013) | 0.258 / **0.043** (0.036, 0.025) | 0.252 / **0.037** (0.038, 0.024) | 0.362 / **0.147** (0.037, 0.024) |
| 8 | 100 | 0.218 (0.024) | 0.241 / **0.024** (0.023, 0.014) | 0.261 / **0.044** (0.036, 0.026) | 0.254 / **0.037** (0.036, 0.024) | 0.362 / **0.145** (0.037, 0.025) |
| 9 | 100 | 0.216 (0.024) | 0.240 / **0.023** (0.022, 0.014) | 0.260 / **0.044** (0.036, 0.026) | 0.253 / **0.037** (0.036, 0.024) | 0.360 / **0.144** (0.037, 0.025) |
| 10 | 100 | 0.218 (0.025) | 0.243 / **0.025** (0.022, 0.014) | 0.263 / **0.045** (0.036, 0.026) | 0.255 / **0.037** (0.036, 0.024) | 0.364 / **0.146** (0.038, 0.025) |
| 11 | 100 | 0.203 (0.025) | 0.226 / **0.024** (0.021, 0.014) | 0.247 / **0.045** (0.033, 0.026) | 0.242 / **0.040** (0.036, 0.024) | 0.363 / **0.160** (0.038, 0.025) |
| 12 | 100 | 0.210 (0.026) | 0.235 / **0.025** (0.022, 0.014) | 0.259 / **0.049** (0.034, 0.025) | 0.254 / **0.044** (0.036, 0.024) | 0.357 / **0.147** (0.037, 0.025) |
| 13 | 100 | 0.245 (0.030) | 0.267 / **0.022** (0.023, 0.014) | 0.288 / **0.043** (0.038, 0.026) | 0.281 / **0.036** (0.041, 0.024) | 0.387 / **0.142** (0.042, 0.024) |
| 14 | 100 | 0.254 (0.032) | 0.274 / **0.020** (0.024, 0.014) | 0.297 / **0.042** (0.038, 0.026) | 0.295 / **0.041** (0.043, 0.024) | 0.389 / **0.135** (0.042, 0.024) |
| 15 | 100 | 0.269 (0.035) | 0.287 / **0.018** (0.026, 0.015) | 0.307 / **0.039** (0.040, 0.028) | 0.306 / **0.038** (0.046, 0.026) | 0.398 / **0.129** (0.042, 0.025) |
| 16 | 100 | 0.273 (0.037) | 0.293 / **0.020** (0.029, 0.020) | 0.308 / **0.035** (0.041, 0.038) | 0.306 / **0.033** (0.049, 0.032) | 0.392 / **0.119** (0.046, 0.033) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.029, rt 0.020 | ΔR² sh 0.042, rt 0.038 | ΔR² sh 0.049, rt 0.032 | ΔR² sh 0.046, rt 0.033 |

*Circle vs. residue-class geometry, 0–99, `{a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² T=100 |
|---|---|---|---|---|---|---|
| 0 | 0.104 (0.029) | 0.160 (0.092, 0.106) | 0.27 | 0.26 | 0.12 | 0.107 |
| 1 | 0.121 (0.034) | 0.153 (0.092, 0.110) | 0.26 | 0.26 | 0.12 | 0.122 |
| 2 | 0.134 (0.038) | 0.145 (0.093, 0.113) | 0.26 | 0.27 | 0.13 | 0.135 |
| 3 | 0.150 (0.041) | 0.138 (0.093, 0.120) | 0.26 | 0.27 | 0.13 | 0.148 |
| 4 | 0.150 (0.042) | 0.139 (0.093, 0.119) | 0.26 | 0.27 | 0.13 | 0.148 |
| 5 | 0.152 (0.043) | 0.142 (0.093, 0.120) | 0.25 | 0.27 | 0.13 | 0.149 |
| 6 | 0.152 (0.045) | 0.143 (0.093, 0.121) | 0.25 | 0.27 | 0.14 | 0.150 |
| 7 | 0.149 (0.046) | 0.155 (0.093, 0.123) | 0.24 | 0.28 | 0.15 | 0.147 |
| 8 | 0.147 (0.047) | 0.155 (0.093, 0.122) | 0.24 | 0.28 | 0.15 | 0.145 |
| 9 | 0.146 (0.047) | 0.154 (0.093, 0.122) | 0.24 | 0.28 | 0.15 | 0.144 |
| 10 | 0.148 (0.047) | 0.156 (0.093, 0.121) | 0.23 | 0.29 | 0.16 | 0.146 |
| 11 | 0.162 (0.046) | 0.158 (0.093, 0.122) | 0.25 | 0.28 | 0.15 | 0.160 |
| 12 | 0.149 (0.047) | 0.179 (0.093, 0.121) | 0.25 | 0.27 | 0.14 | 0.147 |
| 13 | 0.142 (0.050) | 0.155 (0.094, 0.124) | 0.23 | 0.28 | 0.14 | 0.142 |
| 14 | 0.136 (0.051) | 0.153 (0.094, 0.126) | 0.27 | 0.28 | 0.13 | 0.135 |
| 15 | 0.129 (0.053) | 0.141 (0.095, 0.128) | 0.27 | 0.28 | 0.13 | 0.129 |
| 16 | 0.117 (0.054) | 0.132 (0.094, 0.131) | 0.25 | 0.26 | 0.15 | 0.119 |

Fourier (detrended, layer 12): top bins T=100.0 (0.141, null99 0.023), T=50.0 (0.086, null99 0.028), T=2.0 (0.061, null99 0.054), T=5.0 (0.060, null99 0.037), T=33.3 (0.058, null99 0.036), T=10.0 (0.054, null99 0.041)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0537, T=5: 0.0597, T=3.3: 0.0254, T=2.5: 0.0497, T=2: 0.0613
; at K&T periods: T=2: 0.0613 vs null99 0.0539, T=5: 0.0597 vs null99 0.0370, T=10: 0.0537 vs null99 0.0412, T=100: 0.1411 vs null99 0.0233


### OLMo-2 numbers 0–99, template 1: `The number {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.106 (0.017) | 0.126 / **0.020** (0.017, 0.011) | 0.148 / **0.042** (0.027, 0.021) | 0.149 / **0.043** (0.033, 0.021) | 0.214 / **0.107** (0.029, 0.022) |
| 1 | 100 | 0.137 (0.019) | 0.157 / **0.021** (0.018, 0.012) | 0.178 / **0.042** (0.030, 0.022) | 0.179 / **0.042** (0.034, 0.022) | 0.252 / **0.116** (0.031, 0.022) |
| 2 | 100 | 0.146 (0.020) | 0.168 / **0.022** (0.019, 0.012) | 0.190 / **0.043** (0.030, 0.022) | 0.188 / **0.042** (0.034, 0.022) | 0.271 / **0.124** (0.031, 0.023) |
| 3 | 100 | 0.160 (0.020) | 0.183 / **0.023** (0.020, 0.013) | 0.206 / **0.046** (0.031, 0.023) | 0.201 / **0.041** (0.034, 0.022) | 0.286 / **0.126** (0.032, 0.023) |
| 4 | 100 | 0.153 (0.019) | 0.176 / **0.023** (0.019, 0.013) | 0.200 / **0.047** (0.031, 0.023) | 0.193 / **0.040** (0.034, 0.023) | 0.274 / **0.122** (0.031, 0.024) |
| 5 | 100 | 0.147 (0.019) | 0.171 / **0.023** (0.019, 0.013) | 0.196 / **0.049** (0.030, 0.024) | 0.187 / **0.039** (0.032, 0.024) | 0.261 / **0.113** (0.031, 0.024) |
| 6 | 100 | 0.135 (0.019) | 0.159 / **0.024** (0.018, 0.014) | 0.184 / **0.050** (0.029, 0.024) | 0.175 / **0.041** (0.032, 0.024) | 0.241 / **0.106** (0.030, 0.026) |
| 7 | 100 | 0.124 (0.019) | 0.150 / **0.026** (0.019, 0.014) | 0.174 / **0.050** (0.029, 0.026) | 0.164 / **0.040** (0.030, 0.026) | 0.226 / **0.102** (0.029, 0.026) |
| 8 | 100 | 0.122 (0.019) | 0.148 / **0.026** (0.019, 0.014) | 0.173 / **0.050** (0.029, 0.026) | 0.163 / **0.040** (0.030, 0.026) | 0.224 / **0.102** (0.030, 0.027) |
| 9 | 100 | 0.122 (0.019) | 0.149 / **0.027** (0.019, 0.014) | 0.173 / **0.050** (0.029, 0.026) | 0.163 / **0.040** (0.030, 0.026) | 0.222 / **0.100** (0.030, 0.026) |
| 10 | 100 | 0.121 (0.019) | 0.150 / **0.029** (0.018, 0.014) | 0.174 / **0.053** (0.029, 0.027) | 0.162 / **0.041** (0.030, 0.026) | 0.221 / **0.100** (0.030, 0.026) |
| 11 | 100 | 0.123 (0.019) | 0.151 / **0.028** (0.019, 0.014) | 0.175 / **0.052** (0.030, 0.027) | 0.168 / **0.045** (0.030, 0.026) | 0.238 / **0.115** (0.030, 0.026) |
| 12 | 100 | 0.129 (0.018) | 0.159 / **0.030** (0.018, 0.013) | 0.186 / **0.057** (0.030, 0.027) | 0.178 / **0.049** (0.029, 0.026) | 0.237 / **0.108** (0.029, 0.026) |
| 13 | 100 | 0.130 (0.019) | 0.159 / **0.029** (0.019, 0.014) | 0.184 / **0.054** (0.031, 0.027) | 0.175 / **0.045** (0.030, 0.025) | 0.233 / **0.103** (0.028, 0.025) |
| 14 | 100 | 0.129 (0.019) | 0.156 / **0.028** (0.019, 0.014) | 0.181 / **0.053** (0.031, 0.027) | 0.170 / **0.042** (0.031, 0.025) | 0.227 / **0.098** (0.030, 0.026) |
| 15 | 100 | 0.131 (0.020) | 0.156 / **0.025** (0.020, 0.017) | 0.182 / **0.051** (0.031, 0.029) | 0.169 / **0.037** (0.033, 0.026) | 0.225 / **0.094** (0.030, 0.028) |
| 16 | 100 | 0.148 (0.023) | 0.171 / **0.023** (0.023, 0.029) | 0.191 / **0.043** (0.036, 0.036) | 0.182 / **0.034** (0.036, 0.038) | 0.232 / **0.084** (0.035, 0.038) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.023, rt 0.029 | ΔR² sh 0.036, rt 0.036 | ΔR² sh 0.036, rt 0.038 | ΔR² sh 0.035, rt 0.038 |

*Circle vs. residue-class geometry, 0–99, `The number {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² T=100 |
|---|---|---|---|---|---|---|
| 0 | 0.104 (0.029) | 0.160 (0.092, 0.106) | 0.27 | 0.26 | 0.12 | 0.107 |
| 1 | 0.114 (0.033) | 0.160 (0.092, 0.111) | 0.26 | 0.26 | 0.13 | 0.116 |
| 2 | 0.124 (0.035) | 0.163 (0.092, 0.114) | 0.26 | 0.27 | 0.13 | 0.124 |
| 3 | 0.126 (0.037) | 0.165 (0.093, 0.117) | 0.25 | 0.28 | 0.14 | 0.126 |
| 4 | 0.123 (0.036) | 0.166 (0.092, 0.116) | 0.24 | 0.29 | 0.14 | 0.122 |
| 5 | 0.114 (0.035) | 0.168 (0.092, 0.117) | 0.23 | 0.29 | 0.14 | 0.113 |
| 6 | 0.107 (0.034) | 0.172 (0.092, 0.114) | 0.24 | 0.29 | 0.14 | 0.106 |
| 7 | 0.104 (0.034) | 0.176 (0.092, 0.112) | 0.23 | 0.29 | 0.15 | 0.102 |
| 8 | 0.105 (0.034) | 0.178 (0.091, 0.112) | 0.23 | 0.28 | 0.15 | 0.102 |
| 9 | 0.104 (0.034) | 0.178 (0.091, 0.113) | 0.23 | 0.28 | 0.15 | 0.100 |
| 10 | 0.104 (0.033) | 0.183 (0.091, 0.112) | 0.22 | 0.29 | 0.16 | 0.100 |
| 11 | 0.117 (0.034) | 0.182 (0.091, 0.115) | 0.25 | 0.29 | 0.15 | 0.115 |
| 12 | 0.111 (0.034) | 0.209 (0.091, 0.113) | 0.24 | 0.27 | 0.14 | 0.108 |
| 13 | 0.108 (0.033) | 0.196 (0.091, 0.113) | 0.23 | 0.28 | 0.15 | 0.103 |
| 14 | 0.105 (0.034) | 0.184 (0.091, 0.113) | 0.23 | 0.29 | 0.15 | 0.098 |
| 15 | 0.102 (0.035) | 0.170 (0.092, 0.113) | 0.22 | 0.30 | 0.15 | 0.094 |
| 16 | 0.093 (0.037) | 0.147 (0.092, 0.114) | 0.23 | 0.29 | 0.15 | 0.084 |

Fourier (detrended, layer 12): top bins T=100.0 (0.092, null99 0.020), T=50.0 (0.072, null99 0.024), T=2.0 (0.067, null99 0.040), T=5.0 (0.063, null99 0.033), T=10.0 (0.054, null99 0.031), T=33.3 (0.049, null99 0.028)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0544, T=5: 0.0630, T=3.3: 0.0321, T=2.5: 0.0482, T=2: 0.0670
; at K&T periods: T=2: 0.0670 vs null99 0.0404, T=5: 0.0630 vs null99 0.0329, T=10: 0.0544 vs null99 0.0314, T=100: 0.0920 vs null99 0.0201


### OLMo-2 numbers 0–99, template 2: `x = {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.106 (0.017) | 0.126 / **0.020** (0.017, 0.011) | 0.148 / **0.042** (0.027, 0.021) | 0.149 / **0.043** (0.033, 0.021) | 0.214 / **0.107** (0.029, 0.022) |
| 1 | 100 | 0.127 (0.019) | 0.149 / **0.022** (0.018, 0.012) | 0.171 / **0.043** (0.029, 0.022) | 0.171 / **0.044** (0.034, 0.022) | 0.243 / **0.116** (0.030, 0.022) |
| 2 | 100 | 0.139 (0.020) | 0.163 / **0.024** (0.018, 0.012) | 0.185 / **0.046** (0.030, 0.022) | 0.182 / **0.043** (0.034, 0.022) | 0.265 / **0.125** (0.030, 0.023) |
| 3 | 100 | 0.175 (0.023) | 0.201 / **0.026** (0.021, 0.012) | 0.223 / **0.048** (0.033, 0.023) | 0.217 / **0.042** (0.036, 0.022) | 0.302 / **0.128** (0.033, 0.023) |
| 4 | 100 | 0.176 (0.022) | 0.204 / **0.027** (0.022, 0.013) | 0.226 / **0.050** (0.035, 0.023) | 0.219 / **0.043** (0.035, 0.023) | 0.307 / **0.130** (0.034, 0.023) |
| 5 | 100 | 0.186 (0.023) | 0.214 / **0.028** (0.023, 0.013) | 0.238 / **0.051** (0.036, 0.024) | 0.229 / **0.043** (0.033, 0.023) | 0.315 / **0.129** (0.036, 0.023) |
| 6 | 100 | 0.191 (0.024) | 0.221 / **0.031** (0.024, 0.014) | 0.244 / **0.053** (0.036, 0.025) | 0.234 / **0.043** (0.033, 0.024) | 0.318 / **0.127** (0.037, 0.025) |
| 7 | 100 | 0.189 (0.024) | 0.232 / **0.044** (0.025, 0.014) | 0.250 / **0.061** (0.036, 0.025) | 0.231 / **0.043** (0.032, 0.024) | 0.310 / **0.122** (0.037, 0.025) |
| 8 | 100 | 0.194 (0.025) | 0.238 / **0.044** (0.025, 0.015) | 0.255 / **0.061** (0.037, 0.026) | 0.235 / **0.041** (0.032, 0.025) | 0.313 / **0.120** (0.037, 0.026) |
| 9 | 100 | 0.193 (0.024) | 0.237 / **0.044** (0.025, 0.014) | 0.253 / **0.060** (0.037, 0.027) | 0.234 / **0.041** (0.032, 0.025) | 0.310 / **0.117** (0.037, 0.027) |
| 10 | 100 | 0.188 (0.024) | 0.234 / **0.046** (0.024, 0.014) | 0.248 / **0.060** (0.037, 0.027) | 0.229 / **0.041** (0.032, 0.026) | 0.306 / **0.118** (0.037, 0.027) |
| 11 | 100 | 0.173 (0.023) | 0.217 / **0.044** (0.024, 0.014) | 0.233 / **0.059** (0.034, 0.027) | 0.218 / **0.044** (0.031, 0.025) | 0.296 / **0.122** (0.035, 0.026) |
| 12 | 100 | 0.174 (0.022) | 0.218 / **0.044** (0.022, 0.014) | 0.235 / **0.061** (0.033, 0.027) | 0.221 / **0.047** (0.032, 0.025) | 0.287 / **0.113** (0.032, 0.026) |
| 13 | 100 | 0.181 (0.023) | 0.223 / **0.042** (0.022, 0.014) | 0.240 / **0.059** (0.033, 0.027) | 0.224 / **0.043** (0.033, 0.025) | 0.283 / **0.102** (0.032, 0.025) |
| 14 | 100 | 0.189 (0.023) | 0.228 / **0.039** (0.022, 0.015) | 0.245 / **0.056** (0.034, 0.026) | 0.231 / **0.042** (0.035, 0.025) | 0.282 / **0.093** (0.033, 0.025) |
| 15 | 100 | 0.184 (0.022) | 0.220 / **0.036** (0.023, 0.016) | 0.236 / **0.052** (0.033, 0.030) | 0.224 / **0.040** (0.037, 0.027) | 0.278 / **0.094** (0.032, 0.027) |
| 16 | 100 | 0.179 (0.025) | 0.209 / **0.030** (0.024, 0.023) | 0.223 / **0.044** (0.035, 0.046) | 0.223 / **0.044** (0.041, 0.039) | 0.264 / **0.085** (0.036, 0.040) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.026, rt 0.023 | ΔR² sh 0.037, rt 0.046 | ΔR² sh 0.042, rt 0.039 | ΔR² sh 0.039, rt 0.040 |

*Circle vs. residue-class geometry, 0–99, `x = {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² T=100 |
|---|---|---|---|---|---|---|
| 0 | 0.104 (0.029) | 0.160 (0.092, 0.106) | 0.27 | 0.26 | 0.12 | 0.107 |
| 1 | 0.113 (0.032) | 0.166 (0.092, 0.110) | 0.26 | 0.26 | 0.13 | 0.116 |
| 2 | 0.124 (0.034) | 0.171 (0.092, 0.112) | 0.25 | 0.27 | 0.14 | 0.125 |
| 3 | 0.128 (0.038) | 0.175 (0.092, 0.118) | 0.24 | 0.28 | 0.15 | 0.128 |
| 4 | 0.133 (0.041) | 0.178 (0.092, 0.121) | 0.24 | 0.28 | 0.15 | 0.130 |
| 5 | 0.134 (0.044) | 0.183 (0.093, 0.125) | 0.24 | 0.28 | 0.15 | 0.129 |
| 6 | 0.132 (0.044) | 0.189 (0.093, 0.127) | 0.23 | 0.28 | 0.16 | 0.127 |
| 7 | 0.127 (0.046) | 0.216 (0.093, 0.130) | 0.20 | 0.28 | 0.20 | 0.122 |
| 8 | 0.126 (0.046) | 0.215 (0.093, 0.130) | 0.19 | 0.28 | 0.21 | 0.120 |
| 9 | 0.125 (0.047) | 0.213 (0.093, 0.131) | 0.19 | 0.28 | 0.21 | 0.117 |
| 10 | 0.126 (0.046) | 0.215 (0.093, 0.129) | 0.19 | 0.28 | 0.21 | 0.118 |
| 11 | 0.127 (0.043) | 0.213 (0.093, 0.127) | 0.21 | 0.28 | 0.20 | 0.122 |
| 12 | 0.116 (0.041) | 0.227 (0.093, 0.123) | 0.21 | 0.27 | 0.19 | 0.113 |
| 13 | 0.105 (0.041) | 0.218 (0.093, 0.123) | 0.20 | 0.27 | 0.19 | 0.102 |
| 14 | 0.097 (0.042) | 0.205 (0.093, 0.124) | 0.20 | 0.27 | 0.19 | 0.093 |
| 15 | 0.099 (0.042) | 0.196 (0.093, 0.123) | 0.20 | 0.26 | 0.18 | 0.094 |
| 16 | 0.088 (0.041) | 0.181 (0.094, 0.122) | 0.24 | 0.24 | 0.16 | 0.085 |

Fourier (detrended, layer 12): top bins T=2.0 (0.101, null99 0.050), T=100.0 (0.101, null99 0.022), T=50.0 (0.072, null99 0.026), T=5.0 (0.070, null99 0.035), T=2.5 (0.058, null99 0.030), T=10.0 (0.054, null99 0.035)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0542, T=5: 0.0701, T=3.3: 0.0276, T=2.5: 0.0583, T=2: 0.1013
; at K&T periods: T=2: 0.1013 vs null99 0.0498, T=5: 0.0701 vs null99 0.0346, T=10: 0.0542 vs null99 0.0351, T=100: 0.1009 vs null99 0.0217


### OLMo-2 numbers 0–99, template 3: `Output ONLY a number. {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.106 (0.017) | 0.126 / **0.020** (0.017, 0.011) | 0.148 / **0.042** (0.027, 0.021) | 0.149 / **0.043** (0.033, 0.021) | 0.214 / **0.107** (0.029, 0.022) |
| 1 | 100 | 0.133 (0.019) | 0.154 / **0.021** (0.018, 0.012) | 0.176 / **0.043** (0.030, 0.022) | 0.176 / **0.043** (0.035, 0.022) | 0.249 / **0.116** (0.030, 0.022) |
| 2 | 100 | 0.144 (0.020) | 0.167 / **0.023** (0.019, 0.012) | 0.189 / **0.045** (0.030, 0.022) | 0.186 / **0.042** (0.035, 0.022) | 0.269 / **0.125** (0.031, 0.023) |
| 3 | 100 | 0.169 (0.022) | 0.193 / **0.024** (0.021, 0.013) | 0.215 / **0.046** (0.032, 0.023) | 0.211 / **0.042** (0.037, 0.022) | 0.299 / **0.130** (0.033, 0.023) |
| 4 | 100 | 0.170 (0.021) | 0.194 / **0.025** (0.021, 0.013) | 0.216 / **0.046** (0.033, 0.023) | 0.212 / **0.042** (0.035, 0.023) | 0.298 / **0.128** (0.033, 0.024) |
| 5 | 100 | 0.171 (0.021) | 0.195 / **0.024** (0.021, 0.013) | 0.218 / **0.046** (0.033, 0.024) | 0.211 / **0.040** (0.033, 0.024) | 0.295 / **0.124** (0.034, 0.024) |
| 6 | 100 | 0.182 (0.021) | 0.206 / **0.024** (0.022, 0.014) | 0.229 / **0.047** (0.035, 0.025) | 0.222 / **0.040** (0.033, 0.025) | 0.302 / **0.120** (0.033, 0.025) |
| 7 | 100 | 0.182 (0.021) | 0.210 / **0.029** (0.022, 0.014) | 0.234 / **0.053** (0.036, 0.025) | 0.222 / **0.040** (0.033, 0.025) | 0.299 / **0.117** (0.034, 0.025) |
| 8 | 100 | 0.189 (0.021) | 0.218 / **0.029** (0.022, 0.015) | 0.242 / **0.053** (0.039, 0.025) | 0.229 / **0.040** (0.032, 0.026) | 0.308 / **0.119** (0.035, 0.025) |
| 9 | 100 | 0.192 (0.022) | 0.222 / **0.030** (0.024, 0.014) | 0.243 / **0.051** (0.040, 0.026) | 0.232 / **0.040** (0.032, 0.026) | 0.313 / **0.121** (0.036, 0.026) |
| 10 | 100 | 0.185 (0.022) | 0.218 / **0.032** (0.023, 0.014) | 0.237 / **0.052** (0.039, 0.025) | 0.226 / **0.041** (0.032, 0.026) | 0.308 / **0.123** (0.036, 0.025) |
| 11 | 100 | 0.175 (0.022) | 0.206 / **0.031** (0.023, 0.014) | 0.227 / **0.052** (0.038, 0.026) | 0.220 / **0.044** (0.032, 0.025) | 0.308 / **0.133** (0.033, 0.026) |
| 12 | 100 | 0.176 (0.022) | 0.208 / **0.032** (0.022, 0.014) | 0.232 / **0.056** (0.037, 0.026) | 0.223 / **0.047** (0.031, 0.024) | 0.297 / **0.121** (0.033, 0.025) |
| 13 | 100 | 0.195 (0.023) | 0.226 / **0.032** (0.025, 0.013) | 0.245 / **0.050** (0.040, 0.026) | 0.236 / **0.042** (0.033, 0.025) | 0.313 / **0.118** (0.035, 0.025) |
| 14 | 100 | 0.208 (0.025) | 0.238 / **0.030** (0.026, 0.014) | 0.256 / **0.049** (0.041, 0.025) | 0.249 / **0.041** (0.034, 0.025) | 0.319 / **0.112** (0.037, 0.025) |
| 15 | 100 | 0.199 (0.024) | 0.228 / **0.029** (0.025, 0.016) | 0.245 / **0.046** (0.042, 0.028) | 0.237 / **0.038** (0.033, 0.026) | 0.307 / **0.108** (0.037, 0.026) |
| 16 | 100 | 0.190 (0.026) | 0.221 / **0.031** (0.023, 0.024) | 0.230 / **0.041** (0.044, 0.040) | 0.220 / **0.031** (0.034, 0.033) | 0.292 / **0.103** (0.041, 0.038) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.026, rt 0.024 | ΔR² sh 0.044, rt 0.040 | ΔR² sh 0.037, rt 0.033 | ΔR² sh 0.041, rt 0.038 |

*Circle vs. residue-class geometry, 0–99, `Output ONLY a number. {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² T=100 |
|---|---|---|---|---|---|---|
| 0 | 0.104 (0.029) | 0.160 (0.092, 0.106) | 0.27 | 0.26 | 0.12 | 0.107 |
| 1 | 0.114 (0.033) | 0.164 (0.092, 0.111) | 0.26 | 0.26 | 0.13 | 0.116 |
| 2 | 0.123 (0.035) | 0.168 (0.092, 0.113) | 0.25 | 0.27 | 0.14 | 0.125 |
| 3 | 0.129 (0.038) | 0.168 (0.092, 0.116) | 0.25 | 0.28 | 0.14 | 0.130 |
| 4 | 0.128 (0.039) | 0.169 (0.093, 0.117) | 0.25 | 0.27 | 0.15 | 0.128 |
| 5 | 0.125 (0.040) | 0.166 (0.093, 0.120) | 0.24 | 0.28 | 0.15 | 0.124 |
| 6 | 0.122 (0.041) | 0.166 (0.093, 0.122) | 0.24 | 0.28 | 0.15 | 0.120 |
| 7 | 0.120 (0.041) | 0.181 (0.093, 0.124) | 0.22 | 0.29 | 0.16 | 0.117 |
| 8 | 0.123 (0.042) | 0.179 (0.093, 0.126) | 0.22 | 0.29 | 0.16 | 0.119 |
| 9 | 0.126 (0.043) | 0.178 (0.093, 0.127) | 0.23 | 0.29 | 0.17 | 0.121 |
| 10 | 0.127 (0.043) | 0.183 (0.093, 0.124) | 0.22 | 0.28 | 0.18 | 0.123 |
| 11 | 0.136 (0.041) | 0.183 (0.092, 0.122) | 0.24 | 0.29 | 0.17 | 0.133 |
| 12 | 0.126 (0.041) | 0.203 (0.092, 0.120) | 0.23 | 0.27 | 0.16 | 0.121 |
| 13 | 0.125 (0.042) | 0.186 (0.092, 0.121) | 0.22 | 0.27 | 0.17 | 0.118 |
| 14 | 0.121 (0.045) | 0.178 (0.093, 0.122) | 0.23 | 0.27 | 0.17 | 0.112 |
| 15 | 0.119 (0.045) | 0.171 (0.092, 0.120) | 0.22 | 0.27 | 0.17 | 0.108 |
| 16 | 0.118 (0.048) | 0.164 (0.092, 0.125) | 0.19 | 0.25 | 0.19 | 0.103 |

Fourier (detrended, layer 12): top bins T=100.0 (0.110, null99 0.022), T=2.0 (0.075, null99 0.052), T=50.0 (0.073, null99 0.026), T=5.0 (0.065, null99 0.036), T=10.0 (0.055, null99 0.034), T=2.5 (0.054, null99 0.029)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0549, T=5: 0.0651, T=3.3: 0.0257, T=2.5: 0.0537, T=2: 0.0747
; at K&T periods: T=2: 0.0747 vs null99 0.0515, T=5: 0.0651 vs null99 0.0360, T=10: 0.0549 vs null99 0.0335, T=100: 0.1099 vs null99 0.0216


### OLMo-2 numbers 0–999, template 0: `{a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 45 | 0.078 (0.002) | 0.093 / **0.015** (0.002, 0.001) | 0.108 / **0.031** (0.003, 0.002) | 0.108 / **0.031** (0.003, 0.003) | 0.126 / **0.048** (0.003, 0.002) |
| 1 | 49 | 0.088 (0.002) | 0.103 / **0.015** (0.002, 0.001) | 0.118 / **0.029** (0.003, 0.003) | 0.117 / **0.029** (0.003, 0.003) | 0.134 / **0.046** (0.003, 0.003) |
| 2 | 56 | 0.173 (0.002) | 0.186 / **0.014** (0.002, 0.001) | 0.199 / **0.027** (0.003, 0.003) | 0.198 / **0.025** (0.003, 0.003) | 0.216 / **0.043** (0.003, 0.003) |
| 3 | 62 | 0.175 (0.002) | 0.188 / **0.013** (0.002, 0.001) | 0.200 / **0.026** (0.003, 0.003) | 0.199 / **0.024** (0.003, 0.003) | 0.218 / **0.043** (0.003, 0.003) |
| 4 | 65 | 0.170 (0.002) | 0.183 / **0.014** (0.002, 0.001) | 0.195 / **0.026** (0.003, 0.003) | 0.193 / **0.024** (0.003, 0.003) | 0.212 / **0.042** (0.003, 0.003) |
| 5 | 68 | 0.168 (0.002) | 0.182 / **0.014** (0.002, 0.001) | 0.194 / **0.025** (0.003, 0.003) | 0.192 / **0.023** (0.003, 0.003) | 0.208 / **0.040** (0.003, 0.003) |
| 6 | 71 | 0.159 (0.002) | 0.173 / **0.014** (0.002, 0.001) | 0.184 / **0.025** (0.003, 0.003) | 0.182 / **0.023** (0.003, 0.003) | 0.199 / **0.040** (0.003, 0.003) |
| 7 | 74 | 0.165 (0.002) | 0.181 / **0.016** (0.002, 0.002) | 0.190 / **0.025** (0.003, 0.003) | 0.187 / **0.023** (0.003, 0.003) | 0.203 / **0.038** (0.003, 0.003) |
| 8 | 76 | 0.155 (0.002) | 0.171 / **0.016** (0.002, 0.002) | 0.179 / **0.025** (0.003, 0.003) | 0.177 / **0.022** (0.003, 0.003) | 0.193 / **0.039** (0.003, 0.003) |
| 9 | 78 | 0.151 (0.002) | 0.167 / **0.016** (0.002, 0.002) | 0.176 / **0.024** (0.003, 0.003) | 0.173 / **0.022** (0.003, 0.003) | 0.190 / **0.039** (0.003, 0.003) |
| 10 | 79 | 0.157 (0.002) | 0.173 / **0.016** (0.002, 0.002) | 0.182 / **0.025** (0.003, 0.003) | 0.179 / **0.021** (0.003, 0.003) | 0.195 / **0.038** (0.003, 0.003) |
| 11 | 82 | 0.150 (0.002) | 0.166 / **0.016** (0.002, 0.002) | 0.176 / **0.026** (0.003, 0.003) | 0.174 / **0.024** (0.003, 0.003) | 0.189 / **0.039** (0.003, 0.003) |
| 12 | 85 | 0.155 (0.002) | 0.173 / **0.017** (0.002, 0.002) | 0.184 / **0.029** (0.003, 0.003) | 0.180 / **0.025** (0.003, 0.003) | 0.193 / **0.037** (0.003, 0.003) |
| 13 | 85 | 0.158 (0.002) | 0.173 / **0.016** (0.002, 0.002) | 0.184 / **0.026** (0.003, 0.003) | 0.179 / **0.021** (0.004, 0.003) | 0.193 / **0.035** (0.004, 0.003) |
| 14 | 85 | 0.158 (0.002) | 0.172 / **0.014** (0.002, 0.002) | 0.185 / **0.027** (0.003, 0.003) | 0.182 / **0.024** (0.004, 0.003) | 0.192 / **0.034** (0.004, 0.003) |
| 15 | 85 | 0.158 (0.002) | 0.170 / **0.013** (0.002, 0.002) | 0.182 / **0.024** (0.003, 0.003) | 0.178 / **0.021** (0.004, 0.003) | 0.190 / **0.032** (0.004, 0.003) |
| 16 | 89 | 0.154 (0.003) | 0.164 / **0.010** (0.003, 0.002) | 0.172 / **0.019** (0.004, 0.004) | 0.169 / **0.015** (0.004, 0.004) | 0.180 / **0.026** (0.004, 0.005) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.003, rt 0.002 | ΔR² sh 0.004, rt 0.004 | ΔR² sh 0.004, rt 0.004 | ΔR² sh 0.004, rt 0.005 |

*Circle vs. residue-class geometry, 0–999, `{a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² mod100 | share T=100 |
|---|---|---|---|---|---|---|---|
| 0 | 0.076 (0.003) | 0.113 (0.009, 0.010) | 0.27 | 0.27 | 0.14 | 0.299 (0.099, 0.102) | 0.161 |
| 1 | 0.087 (0.003) | 0.106 (0.009, 0.011) | 0.27 | 0.28 | 0.14 | 0.287 (0.099, 0.103) | 0.160 |
| 2 | 0.094 (0.003) | 0.095 (0.009, 0.011) | 0.26 | 0.28 | 0.14 | 0.256 (0.099, 0.105) | 0.168 |
| 3 | 0.109 (0.003) | 0.091 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.250 (0.099, 0.106) | 0.172 |
| 4 | 0.110 (0.003) | 0.091 (0.009, 0.012) | 0.26 | 0.28 | 0.15 | 0.249 (0.099, 0.105) | 0.169 |
| 5 | 0.106 (0.003) | 0.090 (0.009, 0.012) | 0.26 | 0.28 | 0.15 | 0.245 (0.099, 0.105) | 0.164 |
| 6 | 0.104 (0.003) | 0.088 (0.009, 0.012) | 0.26 | 0.28 | 0.16 | 0.245 (0.099, 0.105) | 0.161 |
| 7 | 0.097 (0.003) | 0.092 (0.009, 0.012) | 0.25 | 0.28 | 0.18 | 0.246 (0.099, 0.105) | 0.155 |
| 8 | 0.095 (0.003) | 0.091 (0.009, 0.012) | 0.24 | 0.27 | 0.18 | 0.247 (0.099, 0.105) | 0.156 |
| 9 | 0.093 (0.003) | 0.089 (0.009, 0.012) | 0.25 | 0.27 | 0.18 | 0.246 (0.099, 0.105) | 0.157 |
| 10 | 0.093 (0.003) | 0.089 (0.009, 0.012) | 0.24 | 0.28 | 0.18 | 0.243 (0.099, 0.105) | 0.156 |
| 11 | 0.092 (0.003) | 0.094 (0.009, 0.011) | 0.25 | 0.28 | 0.17 | 0.256 (0.099, 0.106) | 0.152 |
| 12 | 0.100 (0.003) | 0.105 (0.009, 0.012) | 0.24 | 0.27 | 0.17 | 0.263 (0.099, 0.107) | 0.142 |
| 13 | 0.109 (0.004) | 0.096 (0.009, 0.012) | 0.22 | 0.27 | 0.16 | 0.245 (0.099, 0.108) | 0.145 |
| 14 | 0.109 (0.004) | 0.094 (0.009, 0.012) | 0.25 | 0.28 | 0.15 | 0.235 (0.099, 0.108) | 0.144 |
| 15 | 0.115 (0.004) | 0.085 (0.009, 0.012) | 0.24 | 0.28 | 0.15 | 0.216 (0.099, 0.108) | 0.149 |
| 16 | 0.139 (0.004) | 0.065 (0.009, 0.013) | 0.23 | 0.29 | 0.16 | 0.174 (0.099, 0.108) | 0.150 |

Fourier (detrended, layer 0): top bins T=1000.0 (0.064, null99 0.002), T=100.0 (0.051, null99 0.003), T=500.0 (0.047, null99 0.003), T=333.3 (0.042, null99 0.003), T=50.0 (0.041, null99 0.003), T=2.0 (0.033, null99 0.003)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0329, T=5: 0.0327, T=3.3: 0.0129, T=2.5: 0.0251, T=2: 0.0330
; at K&T periods: T=2: 0.0330 vs null99 0.0031, T=5: 0.0327 vs null99 0.0026, T=10: 0.0329 vs null99 0.0026, T=100: 0.0510 vs null99 0.0025


### OLMo-2 numbers 0–999, template 1: `The number {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 45 | 0.078 (0.002) | 0.093 / **0.015** (0.002, 0.001) | 0.108 / **0.031** (0.003, 0.002) | 0.108 / **0.031** (0.003, 0.003) | 0.126 / **0.048** (0.003, 0.002) |
| 1 | 48 | 0.085 (0.002) | 0.101 / **0.016** (0.002, 0.001) | 0.116 / **0.031** (0.003, 0.003) | 0.116 / **0.031** (0.003, 0.003) | 0.133 / **0.048** (0.003, 0.003) |
| 2 | 54 | 0.130 (0.002) | 0.146 / **0.016** (0.002, 0.001) | 0.161 / **0.031** (0.003, 0.003) | 0.159 / **0.030** (0.003, 0.003) | 0.178 / **0.048** (0.003, 0.003) |
| 3 | 60 | 0.130 (0.002) | 0.147 / **0.017** (0.002, 0.001) | 0.162 / **0.031** (0.003, 0.003) | 0.160 / **0.030** (0.003, 0.003) | 0.176 / **0.046** (0.003, 0.003) |
| 4 | 63 | 0.127 (0.002) | 0.143 / **0.017** (0.002, 0.002) | 0.157 / **0.030** (0.003, 0.003) | 0.155 / **0.028** (0.003, 0.003) | 0.170 / **0.044** (0.003, 0.003) |
| 5 | 67 | 0.127 (0.002) | 0.143 / **0.017** (0.002, 0.002) | 0.157 / **0.030** (0.003, 0.003) | 0.154 / **0.027** (0.003, 0.003) | 0.168 / **0.041** (0.003, 0.003) |
| 6 | 70 | 0.115 (0.002) | 0.132 / **0.017** (0.002, 0.002) | 0.144 / **0.029** (0.003, 0.003) | 0.141 / **0.027** (0.003, 0.003) | 0.155 / **0.040** (0.003, 0.003) |
| 7 | 74 | 0.107 (0.002) | 0.129 / **0.022** (0.002, 0.002) | 0.136 / **0.029** (0.003, 0.003) | 0.134 / **0.027** (0.003, 0.003) | 0.144 / **0.037** (0.003, 0.003) |
| 8 | 75 | 0.104 (0.002) | 0.126 / **0.023** (0.002, 0.002) | 0.133 / **0.030** (0.003, 0.003) | 0.130 / **0.027** (0.003, 0.003) | 0.140 / **0.037** (0.003, 0.003) |
| 9 | 76 | 0.099 (0.002) | 0.122 / **0.023** (0.002, 0.002) | 0.129 / **0.030** (0.003, 0.003) | 0.125 / **0.027** (0.003, 0.003) | 0.135 / **0.036** (0.003, 0.003) |
| 10 | 79 | 0.096 (0.002) | 0.121 / **0.024** (0.002, 0.002) | 0.127 / **0.031** (0.003, 0.003) | 0.124 / **0.027** (0.003, 0.003) | 0.132 / **0.036** (0.003, 0.003) |
| 11 | 83 | 0.093 (0.002) | 0.117 / **0.024** (0.002, 0.002) | 0.125 / **0.032** (0.003, 0.003) | 0.123 / **0.030** (0.003, 0.003) | 0.132 / **0.039** (0.003, 0.003) |
| 12 | 85 | 0.097 (0.002) | 0.122 / **0.025** (0.002, 0.002) | 0.132 / **0.035** (0.003, 0.003) | 0.128 / **0.031** (0.003, 0.003) | 0.134 / **0.037** (0.003, 0.003) |
| 13 | 85 | 0.093 (0.002) | 0.117 / **0.024** (0.002, 0.002) | 0.125 / **0.032** (0.003, 0.003) | 0.121 / **0.028** (0.003, 0.003) | 0.126 / **0.034** (0.003, 0.003) |
| 14 | 83 | 0.091 (0.002) | 0.114 / **0.022** (0.002, 0.002) | 0.121 / **0.030** (0.003, 0.003) | 0.116 / **0.025** (0.003, 0.003) | 0.120 / **0.029** (0.003, 0.003) |
| 15 | 85 | 0.106 (0.002) | 0.123 / **0.018** (0.003, 0.002) | 0.129 / **0.024** (0.003, 0.003) | 0.124 / **0.019** (0.004, 0.003) | 0.128 / **0.022** (0.004, 0.003) |
| 16 | 90 | 0.123 (0.002) | 0.135 / **0.012** (0.003, 0.003) | 0.138 / **0.015** (0.004, 0.004) | 0.135 / **0.012** (0.005, 0.005) | 0.137 / **0.014** (0.005, 0.004) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.003, rt 0.003 | ΔR² sh 0.004, rt 0.004 | ΔR² sh 0.005, rt 0.005 | ΔR² sh 0.005, rt 0.004 |

*Circle vs. residue-class geometry, 0–999, `The number {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² mod100 | share T=100 |
|---|---|---|---|---|---|---|---|
| 0 | 0.076 (0.003) | 0.113 (0.009, 0.010) | 0.27 | 0.27 | 0.14 | 0.299 (0.099, 0.102) | 0.161 |
| 1 | 0.085 (0.003) | 0.113 (0.009, 0.011) | 0.27 | 0.27 | 0.14 | 0.302 (0.099, 0.103) | 0.159 |
| 2 | 0.090 (0.003) | 0.112 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.295 (0.099, 0.104) | 0.163 |
| 3 | 0.102 (0.003) | 0.112 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.292 (0.099, 0.104) | 0.158 |
| 4 | 0.105 (0.003) | 0.109 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.285 (0.099, 0.104) | 0.154 |
| 5 | 0.099 (0.003) | 0.107 (0.009, 0.011) | 0.25 | 0.28 | 0.16 | 0.280 (0.099, 0.104) | 0.147 |
| 6 | 0.093 (0.003) | 0.106 (0.009, 0.011) | 0.25 | 0.28 | 0.16 | 0.278 (0.099, 0.104) | 0.144 |
| 7 | 0.088 (0.003) | 0.110 (0.009, 0.011) | 0.24 | 0.27 | 0.20 | 0.279 (0.099, 0.104) | 0.132 |
| 8 | 0.087 (0.003) | 0.111 (0.009, 0.011) | 0.24 | 0.27 | 0.20 | 0.281 (0.099, 0.104) | 0.130 |
| 9 | 0.083 (0.003) | 0.112 (0.009, 0.011) | 0.24 | 0.27 | 0.20 | 0.282 (0.099, 0.104) | 0.128 |
| 10 | 0.082 (0.003) | 0.116 (0.009, 0.010) | 0.24 | 0.27 | 0.21 | 0.287 (0.099, 0.104) | 0.124 |
| 11 | 0.084 (0.003) | 0.119 (0.009, 0.010) | 0.25 | 0.27 | 0.20 | 0.303 (0.099, 0.104) | 0.127 |
| 12 | 0.084 (0.003) | 0.131 (0.009, 0.011) | 0.24 | 0.26 | 0.19 | 0.312 (0.099, 0.104) | 0.118 |
| 13 | 0.082 (0.003) | 0.123 (0.009, 0.011) | 0.23 | 0.26 | 0.20 | 0.296 (0.099, 0.105) | 0.113 |
| 14 | 0.081 (0.003) | 0.111 (0.009, 0.011) | 0.22 | 0.27 | 0.20 | 0.270 (0.100, 0.106) | 0.107 |
| 15 | 0.085 (0.003) | 0.088 (0.009, 0.012) | 0.21 | 0.27 | 0.20 | 0.228 (0.100, 0.107) | 0.098 |
| 16 | 0.098 (0.004) | 0.057 (0.009, 0.013) | 0.21 | 0.26 | 0.21 | 0.168 (0.100, 0.110) | 0.083 |

Fourier (detrended, layer 12): top bins T=1000.0 (0.064, null99 0.002), T=2.0 (0.054, null99 0.003), T=500.0 (0.044, null99 0.003), T=100.0 (0.040, null99 0.003), T=50.0 (0.039, null99 0.004), T=5.0 (0.037, null99 0.003)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0335, T=5: 0.0372, T=3.3: 0.0148, T=2.5: 0.0290, T=2: 0.0543
; at K&T periods: T=2: 0.0543 vs null99 0.0034, T=5: 0.0372 vs null99 0.0028, T=10: 0.0335 vs null99 0.0026, T=100: 0.0396 vs null99 0.0027


### OLMo-2 numbers 0–999, template 2: `x = {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 45 | 0.078 (0.002) | 0.093 / **0.015** (0.002, 0.001) | 0.108 / **0.031** (0.003, 0.002) | 0.108 / **0.031** (0.003, 0.003) | 0.126 / **0.048** (0.003, 0.002) |
| 1 | 48 | 0.084 (0.002) | 0.100 / **0.016** (0.002, 0.001) | 0.115 / **0.031** (0.003, 0.003) | 0.115 / **0.031** (0.003, 0.003) | 0.133 / **0.049** (0.003, 0.003) |
| 2 | 56 | 0.151 (0.002) | 0.168 / **0.017** (0.002, 0.001) | 0.182 / **0.031** (0.003, 0.003) | 0.180 / **0.029** (0.003, 0.003) | 0.199 / **0.047** (0.003, 0.003) |
| 3 | 65 | 0.163 (0.002) | 0.180 / **0.017** (0.002, 0.001) | 0.194 / **0.031** (0.003, 0.003) | 0.191 / **0.029** (0.003, 0.003) | 0.207 / **0.044** (0.003, 0.003) |
| 4 | 70 | 0.155 (0.002) | 0.173 / **0.018** (0.002, 0.002) | 0.187 / **0.032** (0.003, 0.003) | 0.184 / **0.029** (0.003, 0.003) | 0.199 / **0.044** (0.003, 0.003) |
| 5 | 76 | 0.172 (0.002) | 0.191 / **0.019** (0.002, 0.002) | 0.206 / **0.033** (0.003, 0.003) | 0.201 / **0.028** (0.003, 0.003) | 0.214 / **0.042** (0.003, 0.003) |
| 6 | 80 | 0.162 (0.002) | 0.182 / **0.020** (0.002, 0.002) | 0.197 / **0.034** (0.003, 0.003) | 0.191 / **0.029** (0.003, 0.003) | 0.204 / **0.042** (0.003, 0.003) |
| 7 | 85 | 0.161 (0.002) | 0.193 / **0.032** (0.002, 0.002) | 0.201 / **0.040** (0.003, 0.003) | 0.190 / **0.029** (0.003, 0.003) | 0.199 / **0.038** (0.003, 0.003) |
| 8 | 86 | 0.154 (0.002) | 0.187 / **0.033** (0.002, 0.002) | 0.195 / **0.041** (0.003, 0.003) | 0.182 / **0.029** (0.003, 0.003) | 0.192 / **0.038** (0.003, 0.003) |
| 9 | 87 | 0.144 (0.002) | 0.178 / **0.034** (0.002, 0.002) | 0.186 / **0.041** (0.003, 0.003) | 0.173 / **0.029** (0.003, 0.003) | 0.182 / **0.038** (0.003, 0.003) |
| 10 | 88 | 0.136 (0.002) | 0.172 / **0.036** (0.002, 0.002) | 0.180 / **0.044** (0.003, 0.003) | 0.166 / **0.029** (0.003, 0.003) | 0.174 / **0.038** (0.003, 0.003) |
| 11 | 91 | 0.122 (0.002) | 0.157 / **0.035** (0.002, 0.002) | 0.165 / **0.043** (0.003, 0.003) | 0.153 / **0.031** (0.003, 0.003) | 0.161 / **0.039** (0.003, 0.003) |
| 12 | 92 | 0.122 (0.002) | 0.158 / **0.036** (0.002, 0.002) | 0.167 / **0.045** (0.003, 0.003) | 0.154 / **0.032** (0.003, 0.003) | 0.159 / **0.037** (0.003, 0.003) |
| 13 | 91 | 0.122 (0.002) | 0.157 / **0.035** (0.002, 0.002) | 0.166 / **0.044** (0.003, 0.003) | 0.151 / **0.029** (0.003, 0.003) | 0.155 / **0.033** (0.003, 0.003) |
| 14 | 91 | 0.138 (0.002) | 0.171 / **0.032** (0.002, 0.002) | 0.180 / **0.041** (0.004, 0.003) | 0.164 / **0.025** (0.004, 0.003) | 0.166 / **0.028** (0.004, 0.003) |
| 15 | 92 | 0.192 (0.003) | 0.221 / **0.028** (0.003, 0.002) | 0.228 / **0.035** (0.004, 0.003) | 0.212 / **0.019** (0.004, 0.003) | 0.214 / **0.022** (0.004, 0.003) |
| 16 | 94 | 0.262 (0.003) | 0.282 / **0.020** (0.004, 0.003) | 0.287 / **0.025** (0.006, 0.005) | 0.276 / **0.014** (0.006, 0.005) | 0.277 / **0.015** (0.006, 0.004) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.004, rt 0.003 | ΔR² sh 0.006, rt 0.005 | ΔR² sh 0.006, rt 0.005 | ΔR² sh 0.006, rt 0.004 |

*Circle vs. residue-class geometry, 0–999, `x = {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² mod100 | share T=100 |
|---|---|---|---|---|---|---|---|
| 0 | 0.076 (0.003) | 0.113 (0.009, 0.010) | 0.27 | 0.27 | 0.14 | 0.299 (0.099, 0.102) | 0.161 |
| 1 | 0.083 (0.003) | 0.115 (0.009, 0.011) | 0.27 | 0.27 | 0.14 | 0.305 (0.099, 0.103) | 0.160 |
| 2 | 0.091 (0.003) | 0.111 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.290 (0.099, 0.105) | 0.164 |
| 3 | 0.113 (0.003) | 0.111 (0.009, 0.011) | 0.26 | 0.28 | 0.16 | 0.283 (0.099, 0.105) | 0.157 |
| 4 | 0.115 (0.003) | 0.114 (0.009, 0.011) | 0.26 | 0.28 | 0.16 | 0.288 (0.099, 0.105) | 0.152 |
| 5 | 0.110 (0.003) | 0.115 (0.009, 0.011) | 0.25 | 0.29 | 0.16 | 0.285 (0.099, 0.106) | 0.148 |
| 6 | 0.108 (0.003) | 0.119 (0.009, 0.011) | 0.24 | 0.29 | 0.17 | 0.291 (0.099, 0.106) | 0.143 |
| 7 | 0.099 (0.003) | 0.142 (0.009, 0.011) | 0.20 | 0.28 | 0.23 | 0.312 (0.099, 0.105) | 0.123 |
| 8 | 0.100 (0.003) | 0.145 (0.009, 0.011) | 0.20 | 0.29 | 0.23 | 0.314 (0.099, 0.105) | 0.122 |
| 9 | 0.098 (0.003) | 0.147 (0.009, 0.011) | 0.20 | 0.28 | 0.23 | 0.318 (0.099, 0.105) | 0.120 |
| 10 | 0.095 (0.003) | 0.153 (0.009, 0.011) | 0.19 | 0.28 | 0.23 | 0.325 (0.099, 0.105) | 0.116 |
| 11 | 0.094 (0.003) | 0.154 (0.009, 0.011) | 0.20 | 0.28 | 0.23 | 0.336 (0.099, 0.104) | 0.117 |
| 12 | 0.093 (0.003) | 0.163 (0.009, 0.011) | 0.20 | 0.28 | 0.22 | 0.341 (0.099, 0.105) | 0.109 |
| 13 | 0.096 (0.003) | 0.158 (0.009, 0.011) | 0.18 | 0.28 | 0.22 | 0.324 (0.099, 0.106) | 0.102 |
| 14 | 0.107 (0.004) | 0.144 (0.009, 0.011) | 0.18 | 0.29 | 0.22 | 0.291 (0.099, 0.106) | 0.096 |
| 15 | 0.120 (0.004) | 0.123 (0.009, 0.013) | 0.16 | 0.29 | 0.23 | 0.244 (0.099, 0.109) | 0.091 |
| 16 | 0.140 (0.006) | 0.089 (0.009, 0.016) | 0.15 | 0.28 | 0.22 | 0.179 (0.099, 0.114) | 0.084 |

Fourier (detrended, layer 12): top bins T=2.0 (0.079, null99 0.004), T=1000.0 (0.073, null99 0.002), T=5.0 (0.049, null99 0.003), T=500.0 (0.046, null99 0.003), T=50.0 (0.042, null99 0.004), T=100.0 (0.041, null99 0.003)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0349, T=5: 0.0492, T=3.3: 0.0145, T=2.5: 0.0403, T=2: 0.0793
; at K&T periods: T=2: 0.0793 vs null99 0.0037, T=5: 0.0492 vs null99 0.0032, T=10: 0.0349 vs null99 0.0030, T=100: 0.0406 vs null99 0.0029


### OLMo-2 numbers 0–999, template 3: `Output ONLY a number. {a}`

ΔR²_T = R²([1,a,cos,sin]) − R²([1,a]) on the top-100 PCs. Null columns are 99th percentiles over 200 label shuffles: `sh` = numbers with shuffled a, `rt` = 1000 random single tokens with random labels (same templates, same pipeline). PCA% = variance of the layer's cloud captured by the 100 PCs.

| layer | PCA% | R²_lin (sh99) | T=2: R² / ΔR² (sh99, rt99) | T=5: R² / ΔR² (sh99, rt99) | T=10: R² / ΔR² (sh99, rt99) | T=100: R² / ΔR² (sh99, rt99) |
|---|---|---|---|---|---|---|
| 0 | 45 | 0.078 (0.002) | 0.093 / **0.015** (0.002, 0.001) | 0.108 / **0.031** (0.003, 0.002) | 0.108 / **0.031** (0.003, 0.003) | 0.126 / **0.048** (0.003, 0.002) |
| 1 | 48 | 0.086 (0.002) | 0.102 / **0.016** (0.002, 0.001) | 0.117 / **0.031** (0.003, 0.003) | 0.117 / **0.031** (0.003, 0.003) | 0.134 / **0.048** (0.003, 0.003) |
| 2 | 54 | 0.130 (0.002) | 0.147 / **0.016** (0.002, 0.001) | 0.161 / **0.031** (0.003, 0.003) | 0.160 / **0.029** (0.003, 0.003) | 0.178 / **0.048** (0.003, 0.003) |
| 3 | 60 | 0.133 (0.002) | 0.150 / **0.017** (0.002, 0.001) | 0.164 / **0.031** (0.003, 0.003) | 0.162 / **0.029** (0.003, 0.003) | 0.180 / **0.047** (0.003, 0.003) |
| 4 | 63 | 0.126 (0.002) | 0.143 / **0.017** (0.002, 0.002) | 0.156 / **0.030** (0.003, 0.003) | 0.154 / **0.029** (0.003, 0.003) | 0.171 / **0.045** (0.003, 0.003) |
| 5 | 67 | 0.124 (0.002) | 0.140 / **0.017** (0.002, 0.001) | 0.153 / **0.030** (0.003, 0.003) | 0.151 / **0.027** (0.003, 0.003) | 0.166 / **0.042** (0.003, 0.003) |
| 6 | 70 | 0.114 (0.002) | 0.130 / **0.017** (0.002, 0.002) | 0.143 / **0.029** (0.003, 0.003) | 0.140 / **0.027** (0.003, 0.003) | 0.154 / **0.040** (0.003, 0.003) |
| 7 | 75 | 0.108 (0.002) | 0.129 / **0.021** (0.002, 0.002) | 0.139 / **0.031** (0.003, 0.003) | 0.135 / **0.027** (0.003, 0.003) | 0.147 / **0.039** (0.003, 0.003) |
| 8 | 77 | 0.103 (0.002) | 0.124 / **0.021** (0.002, 0.002) | 0.135 / **0.031** (0.003, 0.003) | 0.130 / **0.027** (0.003, 0.003) | 0.142 / **0.039** (0.003, 0.003) |
| 9 | 79 | 0.103 (0.002) | 0.124 / **0.021** (0.002, 0.002) | 0.134 / **0.031** (0.003, 0.003) | 0.130 / **0.027** (0.003, 0.003) | 0.142 / **0.039** (0.003, 0.003) |
| 10 | 81 | 0.102 (0.002) | 0.126 / **0.024** (0.002, 0.002) | 0.134 / **0.032** (0.003, 0.003) | 0.129 / **0.028** (0.003, 0.003) | 0.140 / **0.038** (0.003, 0.003) |
| 11 | 85 | 0.099 (0.002) | 0.122 / **0.024** (0.002, 0.002) | 0.131 / **0.033** (0.003, 0.003) | 0.128 / **0.030** (0.003, 0.003) | 0.138 / **0.039** (0.003, 0.003) |
| 12 | 86 | 0.100 (0.002) | 0.125 / **0.025** (0.002, 0.002) | 0.136 / **0.035** (0.003, 0.003) | 0.131 / **0.031** (0.003, 0.003) | 0.137 / **0.037** (0.003, 0.003) |
| 13 | 86 | 0.102 (0.002) | 0.126 / **0.024** (0.002, 0.002) | 0.134 / **0.032** (0.003, 0.003) | 0.129 / **0.027** (0.003, 0.003) | 0.136 / **0.033** (0.003, 0.003) |
| 14 | 86 | 0.109 (0.002) | 0.131 / **0.022** (0.002, 0.002) | 0.139 / **0.030** (0.003, 0.003) | 0.134 / **0.025** (0.003, 0.003) | 0.139 / **0.030** (0.003, 0.003) |
| 15 | 87 | 0.136 (0.002) | 0.154 / **0.017** (0.003, 0.002) | 0.160 / **0.023** (0.004, 0.003) | 0.155 / **0.019** (0.004, 0.003) | 0.161 / **0.025** (0.004, 0.003) |
| 16 | 92 | 0.173 (0.003) | 0.185 / **0.012** (0.004, 0.002) | 0.187 / **0.014** (0.005, 0.004) | 0.184 / **0.011** (0.005, 0.004) | 0.190 / **0.018** (0.005, 0.004) |
| family-wise 99% (max over layers) | | | ΔR² sh 0.004, rt 0.002 | ΔR² sh 0.005, rt 0.004 | ΔR² sh 0.005, rt 0.004 | ΔR² sh 0.005, rt 0.004 |

*Circle vs. residue-class geometry, 0–999, `Output ONLY a number. {a}`.* ΔR²_mod m = R²([1, a, one-hot(a mod m)]) − R²_lin (in brackets: shuffle mean, 99%). share_T = ΔR²_T / ΔR²_mod: an isotropic set of residue clusters (no circle) gives share 2/(m−1) per circle (T=10, 5: 0.22; T=2: 0.11 for m=10; T=100: 0.02 for m=100); a pure circle gives 1. poly3 = ΔR²([1, a, a², a³]), a non-periodic comparator with the circle's column count.

| layer | ΔR² poly3 | ΔR² mod10 | share T=10 | share T=5 | share T=2 | ΔR² mod100 | share T=100 |
|---|---|---|---|---|---|---|---|
| 0 | 0.076 (0.003) | 0.113 (0.009, 0.010) | 0.27 | 0.27 | 0.14 | 0.299 (0.099, 0.102) | 0.161 |
| 1 | 0.084 (0.003) | 0.113 (0.009, 0.011) | 0.27 | 0.27 | 0.14 | 0.302 (0.099, 0.103) | 0.160 |
| 2 | 0.089 (0.003) | 0.112 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.295 (0.099, 0.104) | 0.164 |
| 3 | 0.100 (0.003) | 0.110 (0.009, 0.011) | 0.26 | 0.28 | 0.15 | 0.291 (0.099, 0.104) | 0.160 |
| 4 | 0.102 (0.003) | 0.110 (0.009, 0.011) | 0.26 | 0.28 | 0.16 | 0.290 (0.099, 0.104) | 0.154 |
| 5 | 0.096 (0.003) | 0.105 (0.009, 0.011) | 0.26 | 0.28 | 0.16 | 0.281 (0.099, 0.104) | 0.149 |
| 6 | 0.092 (0.003) | 0.104 (0.009, 0.011) | 0.25 | 0.28 | 0.16 | 0.282 (0.099, 0.104) | 0.142 |
| 7 | 0.087 (0.003) | 0.112 (0.009, 0.011) | 0.24 | 0.28 | 0.19 | 0.290 (0.099, 0.104) | 0.134 |
| 8 | 0.088 (0.003) | 0.112 (0.009, 0.011) | 0.24 | 0.28 | 0.19 | 0.290 (0.099, 0.105) | 0.133 |
| 9 | 0.090 (0.003) | 0.112 (0.009, 0.011) | 0.24 | 0.28 | 0.19 | 0.291 (0.099, 0.105) | 0.133 |
| 10 | 0.090 (0.003) | 0.117 (0.009, 0.011) | 0.24 | 0.27 | 0.20 | 0.297 (0.099, 0.105) | 0.129 |
| 11 | 0.093 (0.003) | 0.121 (0.009, 0.011) | 0.25 | 0.27 | 0.20 | 0.307 (0.099, 0.104) | 0.127 |
| 12 | 0.094 (0.003) | 0.132 (0.009, 0.011) | 0.23 | 0.27 | 0.19 | 0.314 (0.099, 0.104) | 0.118 |
| 13 | 0.100 (0.003) | 0.122 (0.009, 0.011) | 0.22 | 0.26 | 0.20 | 0.292 (0.100, 0.105) | 0.115 |
| 14 | 0.103 (0.004) | 0.111 (0.009, 0.011) | 0.22 | 0.27 | 0.20 | 0.268 (0.100, 0.106) | 0.111 |
| 15 | 0.106 (0.004) | 0.089 (0.009, 0.012) | 0.21 | 0.26 | 0.20 | 0.229 (0.100, 0.107) | 0.108 |
| 16 | 0.127 (0.005) | 0.058 (0.009, 0.012) | 0.19 | 0.24 | 0.21 | 0.175 (0.100, 0.111) | 0.101 |

Fourier (detrended, layer 0): top bins T=1000.0 (0.064, null99 0.002), T=100.0 (0.051, null99 0.003), T=500.0 (0.047, null99 0.003), T=333.3 (0.042, null99 0.003), T=50.0 (0.041, null99 0.003), T=2.0 (0.033, null99 0.003)
; harmonics of 10 (bins 1–5 × N/10): T=10: 0.0329, T=5: 0.0327, T=3.3: 0.0129, T=2.5: 0.0251, T=2: 0.0330
; at K&T periods: T=2: 0.0330 vs null99 0.0031, T=5: 0.0327 vs null99 0.0026, T=10: 0.0329 vs null99 0.0026, T=100: 0.0510 vs null99 0.0025


### Qwen3-0.6B days (K=7, 24 templates)

Supervised mean-difference plane (top-2 PCs of the class means). R²_in: circle regression P = c + A[cos 2πk/K, sin 2πk/K] on all points; R²_ho: plane + circle fitted on even templates, scored on odd (and swapped). Nulls: 99th pct over shuffles of the class order (`ord`) and of all point labels (`pts`), through the identical pipeline. cyclic: the class means' angular order equals the calendar order.

| layer | plane var% | R²_in (ord99, pts99) | R²_ho (ord99, pts99) | cyclic |
|---|---|---|---|---|
| 0 | 50 | 0.866 (0.684, 0.059) | **0.866** (0.684, 0.017) | yes |
| 1 | 49 | 0.789 (0.666, 0.076) | **0.789** (0.666, 0.011) | yes |
| 2 | 49 | 0.778 (0.649, 0.084) | **0.777** (0.648, -0.000) | yes |
| 3 | 48 | 0.766 (0.634, 0.097) | **0.765** (0.633, -0.007) | yes |
| 4 | 48 | 0.786 (0.654, 0.091) | **0.785** (0.654, -0.012) | yes |
| 5 | 47 | 0.780 (0.647, 0.098) | **0.778** (0.645, -0.028) | yes |
| 6 | 47 | 0.852 (0.678, 0.105) | **0.849** (0.676, -0.020) | yes |
| 7 | 48 | 0.839 (0.676, 0.130) | **0.833** (0.670, -0.043) | yes |
| 8 | 49 | 0.860 (0.670, 0.122) | **0.849** (0.657, -0.045) | yes |
| 9 | 49 | 0.857 (0.664, 0.099) | **0.843** (0.649, -0.085) | yes |
| 10 | 48 | 0.759 (0.705, 0.106) | **0.752** (0.697, -0.068) | yes |
| 11 | 55 | 0.917 (0.704, 0.102) | **0.905** (0.695, -0.058) | yes |
| 12 | 53 | 0.929 (0.693, 0.099) | **0.920** (0.685, -0.054) | yes |
| 13 | 52 | 0.920 (0.693, 0.097) | **0.911** (0.684, -0.056) | yes |
| 14 | 52 | 0.912 (0.691, 0.102) | **0.903** (0.683, -0.063) | yes |
| 15 | 52 | 0.911 (0.687, 0.107) | **0.897** (0.675, -0.064) | yes |
| 16 | 52 | 0.906 (0.680, 0.103) | **0.889** (0.665, -0.101) | yes |
| 17 | 50 | 0.852 (0.742, 0.100) | **0.847** (0.737, -0.053) | yes |
| 18 | 49 | 0.861 (0.732, 0.105) | **0.855** (0.726, -0.065) | yes |
| 19 | 48 | 0.887 (0.693, 0.106) | **0.877** (0.685, -0.079) | yes |
| 20 | 47 | 0.899 (0.727, 0.106) | **0.891** (0.722, -0.077) | yes |
| 21 | 46 | 0.862 (0.677, 0.109) | **0.854** (0.670, -0.070) | yes |
| 22 | 47 | 0.807 (0.714, 0.109) | **0.800** (0.709, -0.069) | no |
| 23 | 46 | 0.838 (0.702, 0.109) | **0.829** (0.693, -0.069) | yes |
| 24 | 47 | 0.836 (0.703, 0.112) | **0.824** (0.691, -0.078) | yes |
| 25 | 47 | 0.848 (0.704, 0.111) | **0.831** (0.687, -0.087) | yes |
| 26 | 47 | 0.853 (0.697, 0.109) | **0.832** (0.676, -0.114) | yes |
| 27 | 50 | 0.772 (0.591, 0.100) | **0.688** (0.508, -0.142) | yes |
| 28 | 51 | 0.618 (0.507, 0.099) | **0.495** (0.386, -0.143) | yes |
| family-wise 99% (max over layers) | | | ord 0.740, pts 0.017 | |

Exact order null (all 360 cyclic orders up to rotation/reflection), rank of the true order by held-out R² per layer: L0:1, L1:1, L2:1, L3:1, L4:1, L5:1, L6:1, L7:1, L8:1, L9:1, L10:2, L11:1, L12:1, L13:1, L14:1, L15:1, L16:1, L17:1, L18:1, L19:1, L20:1, L21:1, L22:2, L23:1, L24:1, L25:1, L26:1, L27:1, L28:1

### Qwen3-0.6B months (K=12, 24 templates)

Supervised mean-difference plane (top-2 PCs of the class means). R²_in: circle regression P = c + A[cos 2πk/K, sin 2πk/K] on all points; R²_ho: plane + circle fitted on even templates, scored on odd (and swapped). Nulls: 99th pct over shuffles of the class order (`ord`) and of all point labels (`pts`), through the identical pipeline. cyclic: the class means' angular order equals the calendar order.

| layer | plane var% | R²_in (ord99, pts99) | R²_ho (ord99, pts99) | cyclic |
|---|---|---|---|---|
| 0 | 36 | 0.762 (0.396, 0.045) | **0.762** (0.396, 0.007) | yes |
| 1 | 32 | 0.730 (0.422, 0.047) | **0.730** (0.421, -0.010) | no |
| 2 | 31 | 0.830 (0.426, 0.053) | **0.827** (0.424, -0.045) | no |
| 3 | 31 | 0.911 (0.464, 0.051) | **0.908** (0.462, -0.079) | yes |
| 4 | 31 | 0.846 (0.464, 0.047) | **0.833** (0.454, -0.150) | no |
| 5 | 32 | 0.831 (0.457, 0.051) | **0.814** (0.441, -0.107) | no |
| 6 | 33 | 0.862 (0.447, 0.054) | **0.854** (0.441, -0.036) | no |
| 7 | 33 | 0.844 (0.439, 0.059) | **0.838** (0.435, -0.043) | no |
| 8 | 34 | 0.832 (0.436, 0.059) | **0.827** (0.432, -0.034) | no |
| 9 | 37 | 0.810 (0.436, 0.064) | **0.794** (0.420, -0.065) | no |
| 10 | 37 | 0.778 (0.422, 0.059) | **0.770** (0.413, -0.049) | no |
| 11 | 49 | 0.886 (0.436, 0.057) | **0.883** (0.434, -0.045) | yes |
| 12 | 45 | 0.903 (0.445, 0.056) | **0.902** (0.444, -0.015) | yes |
| 13 | 44 | 0.900 (0.439, 0.055) | **0.899** (0.438, -0.031) | yes |
| 14 | 44 | 0.897 (0.438, 0.054) | **0.896** (0.437, -0.039) | yes |
| 15 | 44 | 0.895 (0.436, 0.056) | **0.894** (0.436, -0.036) | yes |
| 16 | 44 | 0.894 (0.437, 0.053) | **0.893** (0.436, -0.057) | yes |
| 17 | 42 | 0.884 (0.432, 0.054) | **0.883** (0.431, -0.025) | yes |
| 18 | 41 | 0.884 (0.446, 0.057) | **0.883** (0.445, -0.039) | yes |
| 19 | 37 | 0.881 (0.436, 0.057) | **0.879** (0.435, -0.032) | yes |
| 20 | 38 | 0.838 (0.452, 0.061) | **0.837** (0.451, -0.027) | yes |
| 21 | 35 | 0.880 (0.483, 0.062) | **0.880** (0.483, -0.031) | yes |
| 22 | 36 | 0.915 (0.467, 0.062) | **0.915** (0.467, -0.023) | yes |
| 23 | 35 | 0.905 (0.467, 0.058) | **0.904** (0.466, -0.027) | yes |
| 24 | 35 | 0.905 (0.468, 0.056) | **0.903** (0.467, -0.027) | yes |
| 25 | 36 | 0.908 (0.470, 0.053) | **0.905** (0.468, -0.027) | yes |
| 26 | 36 | 0.909 (0.470, 0.050) | **0.904** (0.466, -0.025) | yes |
| 27 | 37 | 0.774 (0.442, 0.041) | **0.756** (0.424, -0.034) | no |
| 28 | 38 | 0.719 (0.429, 0.044) | **0.697** (0.407, -0.031) | no |
| family-wise 99% (max over layers) | | | ord 0.489, pts 0.007 | |
