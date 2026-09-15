# Two-week plan

**Start:** Tue 2026-09-15 (day 1). **Deadline:** Tue 2026-09-29, a complete draft with all figures.

The plan has three phases separated by **gates**. At each gate I stop and make an explicit decision in the journal.
The phases exist to force two things researchers are bad at: killing ideas early, and starting to write before feeling ready.

```
 days 1–2   ░░ MAP       read, build test harness, shortlist ideas
 days 3–5   ▒▒ PROBE     ≤1-day probe per idea, kill most        ── GATE A (end of day 5): pick ONE direction
 days 6–10  ▓▓ DEEPEN    mechanism → theory → scale-up           ── GATE B (end of day 10): freeze the claim
 days 11–14 ██ WRITE     paper draft, red-team, final figures    ── deadline 09-29
```

## Phase 1 — Map (days 1–2: Sep 15–16)

- [ ] Literature sweep of linear attention and its relatives, focused on 2024–2026 work.
  - Families: linear attention, DeltaNet / Gated DeltaNet, Mamba-2 and other SSMs, RWKV-7, TTT / Titans / MesaNet, log-linear attention, hybrids.
  - Theory: associative recall capacity, in-context learning as online optimization, expressivity (state tracking), length generalization.
  - Output: `literature/` notes + a "known / open / contested" map in `literature/README.md`.
- [ ] Read Fan Pu's linear-attention work in the repo (`day*/`, `theory/08-icl-linear-attention/`). Note what they've found and what it suggests.
- [ ] Build `common/`: a small, tested library.
  - Recurrent reference implementations of each mixer in float64.
  - Checks that `fla` kernels match them.
  - Synthetic task generators (multi-query associative recall, in-context regression, state tracking).
  - A minimal LM training loop with pause/checkpoint support.
- [ ] Shortlist 4–6 ideas in `IDEAS.md`, each with a prediction, a cheap probe, a kill criterion, and a novelty status.

## Phase 2 — Probe (days 3–5: Sep 17–19)

- [ ] Run one probe per shortlisted idea: small model, synthetic data, or a derivation. ≤1 day each; several can share a day.
- [ ] Each probe's outcome goes in `RESULTS.md`, including kills.
- [ ] Deeper novelty check on the survivors (a subagent literature sweep, if needed).

### GATE A (end of Fri Sep 19): choose one direction
Write the decision in the journal. The chosen direction must have:
1. A probe result that already shows the effect (even noisily).
2. No prior paper making the same claim.
3. A believable path to a *mechanism or formula*, not just a benchmark number.
4. A compute plan that fits the remaining GPU share.

If nothing passes, take the most promising idea, spend day 6 probing it further, and re-gate. Never slide more than one day.

## Phase 3 — Deepen (days 6–10: Sep 20–24)

- [ ] Explain the effect on the smallest setting where it appears. Find the mechanism.
- [ ] Theory: derive a prediction (closed form, scaling law, or stability condition) and overlay it on measurements.
- [ ] Robustness: seeds, model sizes, sequence lengths, a second task or data distribution.
- [ ] Scale-up: one realistic setting (e.g. a small LM run) showing the effect matters outside toys.
- [ ] Start `paper/` on day 8 with an outline and the key figure.

### GATE B (end of Thu Sep 24): freeze the claim
Write the abstract in the journal. From here on, experiments only fill gaps the draft has already exposed.

## Phase 4 — Write (days 11–14: Sep 25–28, + Sep 29)

- [ ] Full draft: intro (why care) → background (just enough) → result → mechanism/theory → experiments → limitations → related work.
- [ ] Final figures, designed with the `dataviz` skill.
- [ ] Red-team: one subagent reviews the draft as a hostile reviewer. Address or concede each point.
- [ ] Reproducibility: a single command per figure; README in `paper/`.
- [ ] Sep 29: the draft is done. Final summary for Fan Pu in `SYNC.md` and the journal.

## Change log
Record every change to this plan with the date and the reason.

- 2026-09-15: plan created.
