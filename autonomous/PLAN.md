# Plan

**Outer deadline:** a complete draft with all figures by 2026-09-29. Earlier is better.

**The real constraint is tokens, not days.** Phases advance when their exit criteria are met, not when the calendar says so. If the GPU is busy on a long run, I work on something else or stop and wait cheaply; I don't burn tokens to fill time.

The plan has four phases separated by two **gates**, where I stop and write an explicit decision in the journal.
The structure exists to force two things researchers are bad at: killing ideas early, and starting to write before feeling ready.

```
 ░░ MAP      read, build test harness, shortlist ideas         ~15% of token budget
 ▒▒ PROBE    one cheap probe per idea, kill most               ~25%   ── GATE A: pick ONE direction
 ▓▓ DEEPEN   mechanism → theory → scale-up                     ~35%   ── GATE B: freeze the claim
 ██ WRITE    paper draft, red-team, final figures              ~25%   ── done
```

**Budget tracking.** The token shares above are guides, not quotas. At each phase exit, note in the journal roughly how much budget has been used. If a phase runs past ~1.5× its share, I must either pass its gate with what I have or explicitly re-plan (logged below).

## Phase 1 — Map

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

**Exit when:** 4–6 ideas are shortlisted in `IDEAS.md`, each with a prediction, a cheap probe, a kill criterion, and a novelty status, and `common/` tests pass.

## Phase 2 — Probe

- [ ] Run one probe per shortlisted idea: small model, synthetic data, or a derivation. Launch GPU probes back to back so the GPU works while I read or derive.
- [ ] Each probe's outcome goes in `RESULTS.md`, including kills.
- [ ] Deeper novelty check on the survivors (a subagent literature sweep, if needed).

### GATE A: choose one direction
Write the decision in the journal. The chosen direction must have:
1. A probe result that already shows the effect (even noisily).
2. No prior paper making the same claim.
3. A believable path to a *mechanism or formula*, not just a benchmark number.
4. A compute plan that fits the remaining GPU share and token budget.

If nothing passes, give the most promising idea one more probe and re-gate once. After that, choose anyway.

## Phase 3 — Deepen

- [ ] Explain the effect on the smallest setting where it appears. Find the mechanism.
- [ ] Theory: derive a prediction (closed form, scaling law, or stability condition) and overlay it on measurements.
- [ ] Robustness: seeds, model sizes, sequence lengths, a second task or data distribution.
- [ ] Scale-up: one realistic setting (e.g. a small LM run) showing the effect matters outside toys.
- [ ] Start `paper/` as soon as the key figure exists: outline plus that figure.

### GATE B: freeze the claim
Write the abstract in the journal. From here on, experiments only fill gaps the draft has already exposed.

## Phase 4 — Write

- [ ] Full draft: intro (why care) → background (just enough) → result → mechanism/theory → experiments → limitations → related work.
- [ ] Final figures, designed with the `dataviz` skill.
- [ ] Red-team: one subagent reviews the draft as a hostile reviewer. Address or concede each point.
- [ ] Reproducibility: a single command per figure; README in `paper/`.
- [ ] Final summary for Fan Pu in `SYNC.md` and the journal.

## Change log
Record every change to this plan with the date and the reason.

- 2026-09-15: plan created.
- 2026-09-15: switched from calendar-day phases to milestone gates with token-budget shares. Fan Pu pointed out that tokens, not wall-clock time, are the bottleneck.
