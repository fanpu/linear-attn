# Results ledger

One entry per concluded experiment, **positive or negative**, newest first. This is the single place to answer "what do we actually know?"

Each entry follows this format:

```markdown
### R-NNN · <claim, stated as a finding> · <confidence>
- **Date / experiment:** YYYY-MM-DD · experiments/NNN-slug/
- **Question:** what was asked, in one sentence.
- **Result:** the numbers (mean ± spread, n seeds, how many runs were tried in total).
- **Figure:** experiments/NNN-slug/figures/xxx.png
- **Caveats:** what could make this wrong; what wasn't tested.
- **Takeaway:** one plain sentence a non-specialist could repeat.
```

**Confidence levels** (defined in `CHARTER.md` §7):
- **confirmed**: ≥3 seeds, controls done, red-teamed.
- **likely**: consistent evidence, not yet stress-tested.
- **preliminary**: one run, or one setting.
- **refuted**: tested and false.

---

### R-001 · The fla chunk kernel's float32 error does not grow with sequence length (kills S5) · likely
- **Date / experiment:** 2026-09-15 · `common/test_common.py`, plus Fan Pu's `day1/results.txt`
- **Question:** Does rounding error in the chunked (WY / UT-transform) delta-rule kernel accumulate along the sequence, as seed S5 predicted?
- **Result:**
  - Fan Pu's day-1 table (Gated DeltaNet, float32 with IEEE matmul): the max relative forward error is 0.7–1.3×10⁻⁴ at T = 256, 1024, and 4096 for every head size. It's flat in T.
  - My check (T = 64–512, D = 16–128) agrees. Chunk kernel: 5×10⁻⁵ (gated) to 6×10⁻⁴ (ungated) relative error. Recurrent kernel: ~10⁻⁶. Chunk with Triton's default TF32 matmul: ~2×10⁻³.
  - fla forces IEEE matmul only on pre-Ampere cards, so on the GB10 you must set `TRITON_F32_DEFAULT=ieee` yourself. Fan Pu's day 1 had already found this too.
- **Figure:** none. The numbers are in `day1/results.txt` and the journal.
- **Caveats:** only random inputs, not trained weights, where transitions could be closer to singular. Only T ≤ 4096. bf16 is not covered by my check.
- **Takeaway:** the training kernel is a little less precise than the step-by-step one, but the error is a fixed floor rather than something that snowballs with context length. There's no paper in it.
