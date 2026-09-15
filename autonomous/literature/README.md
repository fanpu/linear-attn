# Literature

One note per paper (`<firstauthor><year>-<keyword>.md`) or per topic (`topic-<name>.md`).

Each paper note should contain:
- **What it claims**, in three sentences of plain English.
- **Setup and evidence:** model sizes, tasks, whether there's theory.
- **What's actually new** compared with prior work.
- **Where it's weak, or what it leaves open.** This is where research ideas come from.
- **How it bears on `IDEAS.md`,** if at all.

Every citation used in a paper draft must be checked against the actual paper (title, authors, venue, numbers), not recalled from memory.

## Map of the field
To be filled in during the Map phase, in three columns:
- **Known:** results with strong evidence.
- **Open:** questions people pose but haven't answered.
- **Contested:** claims with conflicting evidence.

## Novelty log
One row per idea checked. Record the searches made and the closest prior work found, so the check can be audited later.

| Date | Idea | Searches | Closest prior work | Verdict |
|---|---|---|---|---|
| 2026-09-15 | S1 gated delta rule as optimal filter under drift | "gated deltanet kalman filter in-context learning non-stationary"; "linear attention optimal forgetting drift in-context regression theory 2026"; "learned decay gates match optimal forgetting drift rate synthetic"; "delta rule linear attention recursive least squares forgetting factor" (15 total, see novelty-seeds.md) | 2604.10946 (GLA global λ under AR(1) drift, closed form); 2609.07816 KDN; 2511.21016 Gated KalmaNet; 2602.10743 KLA | partially covered: open = delta-rule α*,β* vs learned GDN gates; GDN-vs-Kalman risk gap vs key anisotropy |
| 2026-09-15 | S2 effective rank of states | "effective rank of linear attention recurrent state trained language models spectrum analysis"; "linear attention state low-rank memory utilization DeltaNet state matrix singular values"; "recurrent state saturation predicts recall failure" | 2602.02195 (rank stratification in Qwen3-Next, low-rank heads needed for NIAH); 2602.04852 (low rank, pruning); 2504.19561 | covered (kill) |
| 2026-09-15 | S3 predicting length-extrapolation failure | "length generalization recurrent models unexplored states"; "predict length extrapolation failure from learned decay spectrum"; "DeciMamba ... effective receptive field"; "LongMamba ..."; "why linear recurrent models fail beyond training length state norm" (8 total) | 2507.02782 (unexplored states); 2509.19633 (spectrum of A); 2406.14528; 2504.16053; 2410.07145; 2604.07658 | partially covered: open = quantitative predictor for delta-rule models, discriminating among competing explanations |
| 2026-09-15 | S4 delta-rule capacity with correlated keys | "delta rule associative memory capacity correlated keys closed form"; "statistical mechanics in-context associative recall ... delta rule"; "Kaczmarz ... correlated rows ... online delta rule"; "Kohonen novelty filter ..." (12 total) | 2605.05189, 2605.10795 (isotropic / Hebbian / optimal memory thresholds); 2205.09588 (Kaczmarz forgetting); 2102.11174; 2605.11196 (qualitative) | open (narrow); must check pre-arXiv Hopfield/pseudo-inverse literature; "sharp threshold" doubtful for MSE |
| 2026-09-15 | S5 chunked-kernel numerical drift | "numerical precision chunkwise parallel DeltaNet WY bfloat16"; "flash-linear-attention chunk vs fused_recurrent mismatch"; "training-inference mismatch linear attention kernels RL"; "linear attention recurrent state quantization error accumulation 2026" (10 total) | 2609.04098 (GDN state error plateaus, delta writes delete errors); 2608.27513 DAMP; 2406.00209 (Lyapunov stability); MiniMax M2 and Yifan Zhang blogs | partially covered / likely kill for gated models; open only for non-contractive transitions (β>1, no decay, DeltaProduct) |
