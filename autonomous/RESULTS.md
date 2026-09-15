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

(No results yet.)
