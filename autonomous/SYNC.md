# Sync: notes between Claude and Fan Pu

A shared whiteboard. Newest first. Keep entries short, and date them.

- **For Fan Pu:** papers, ideas, or observations relevant to your work; questions; requests (e.g. "I'd like ~6 GPU hours tomorrow night").
- **From Fan Pu:** anything you'd like me to know or look at. Add a line under "Inbox", and I'll read it at the start of each session.

To pause my GPU use: tell me in chat, or run `touch autonomous/GPU_PAUSE`. To resume: `rm autonomous/GPU_PAUSE`.

---

## Inbox (from Fan Pu)

(empty)

## Outbox (from Claude)

- **2026-09-15 08:35: Paused as requested.** `autonomous/GPU_PAUSE` is set, and none of my jobs are running. To let me use the GPU again: `autonomous/tools/resume.sh` (or tell me).
  - Status: Gate A passed. Direction: how much Kalman-style memories can beat a well-configured gated delta rule. Results are R-002–R-004 in `RESULTS.md`.
  - Experiment 003 (trained models) finished before the pause but is unanalyzed.
  - Possibly relevant to your theory/08: a delta-rule memory's best key metric looks like Σ^(−½), not the Σ⁻¹ that linear attention learns.

- **2026-09-15:** Set up `autonomous/`. Plan: read and shortlist → probe ideas → pick one direction (Gate A) → deepen → freeze the claim (Gate B) → write. Paced by milestones and token budget; draft by Sep 29 at the latest. My GPU jobs honour `GPU_PAUSE` and checkpoint within about a minute. I commit only paths under `autonomous/`, locally, and never push.
