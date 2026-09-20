# dl-alchemy: live state

Read `COURSE.md` first (standing instructions are in its §1). Keep this file current; update it before ending
any session that changed something.

## Now

- **Current unit:** 1 (Basics: hyperparameter tuning and scaling).
- **Current phase:** P1 done 2026-09-19. `unit1-basics/notes.pdf` (v1.0, 9 pages, every citation looked up) and
  `unit1-basics/handout.pdf` (**draft v0.95**, 4 pages: baseline measured; best LR and noise floor pending).
  Fan Pu can read both now. P0 check the same day: CS 312 site unchanged, no A1 material public.
- **GPU permission (2026-09-19, later the same day):** "the currently running job doesnt require gpu exclusivity.
  feel free to run jobs. just dont oom the machine". His day4 MQAR sweep shares the GPU. Rules in force: one
  dl-alchemy job at a time, `mem_fraction=0.25` (~30 GB cap) in `alchemy/train.py`, `alchemy/queue.py` waits for
  MemAvailable >= 35 GB before each run, `expandable_segments`. Loss numbers are valid under sharing; **timings are
  not** (shared speed is roughly half of exclusive) and must be re-measured on an idle GPU.
- **P2 progress:**
  1. DONE `alchemy/` built and smoke-tested: `model.py` (CS 312 Block/Transformer), `data.py` (stateless
     (seed, step) batches from llm.c shards), `train.py` (all handout knobs, jsonl log, resume, result.json),
     `queue.py` (sequential runner, STOP file, memory check), `calibrate.py`, `retokenize.py`.
  2. DONE vocab decision: measured on the shared GPU, width 384: GPT-2 50k vocab 26k tok/s vs 8k vocab 75k tok/s
     (head was 55% of FLOPs). Retokenizing to byte-level BPE 8192 → `data/fineweb_edu_bpe8k/` (CPU job,
     `logs/retokenize.log`).
  3. DONE baseline fixed in `unit1-basics/baseline.toml`: depth 8, **width 256** (4 heads), vocab 8192, T 512,
     **120M tokens**, batch 64 seqs = 32,768 tokens (~3,660 steps), AdamW(0.9, 0.95), wd 0.1, clip 1, 100 warmup
     steps, cosine to 10%. 6.3M body + 2.1M head params. Why width 256: measured shared-GPU throughput was
     105k/89k/77k/61k tok/s at widths 256/320/384/512 (pure compute), so a ~20 tok/param run is ~25 min at 256 but
     ~60 min at 384. Real train loop at width 256: 77k tok/s shared → ~28 min/run, peak 6.4 GB. Train-loop syncs are
     batched (one per 10 steps) because each sync waits a time-slice on a shared GPU.
  4. **RUNNING** calibration, launched 2026-09-19 04:06 detached (setsid): six-point LR grid then 4 extra seeds at
     the best LR, 10 runs, ~4.5-5 h shared. Log `logs/calibrate.log`; per-run `stdout.log` under
     `unit1-basics/quiz/sealed/calibration/<run>/`. Check: `pgrep -af alchemy.calibrate`; `tail logs/calibrate.log`
     ("calibration finished" at the end). To stop: `touch unit1-basics/quiz/sealed/calibration/STOP` (stops after the
     current run) or kill the PIDs. It is resumable: re-run the command in `alchemy/calibrate.py`'s docstring.
     Output: `unit1-basics/quiz/sealed/calibration/calibration.json`; its `public` block (best LR, noise floor,
     timing) may be shown to him, its `sealed` block (the whole LR bowl = his lr-bowl problem) may not. If he later
     asks for the baseline LR grid as his own experiment, take his prediction first, then reveal these runs (no GPU).
  5. THEN (next session, when he says go): sealed pilots (`unit1-basics/quiz/sealed/pilots/`): batch-size effect over
     16x, cooldown drop (constant vs cosine vs wsd), IsoFLOP bowls at C0/16, C0/4, C0 bracket a minimum; status only
     in chat. Then re-issue `handout.pdf` as v1.0 (currently draft v0.95: baseline numbers are final, best LR and
     noise floor missing) and start P3.
  6. TODO once the GPU is idle: re-time the baseline to get exclusive minutes/run.
- Before P2 he may already want a tutorial on the notes (no GPU). Practice questions are in notes §8.
- **GPU jobs running for this project:** the calibration chain above (one process at a time, ~6.4 GB).
- **File delivery:** SendUserFile failed in the first session ("not on a project thread"); give him the PDF paths.

## Unit status

| # | Unit | Phase | Real CS 312 material available? | Quiz score |
|---|---|---|---|---|
| 1 | Basics: tuning and scaling | P1 done (notes v1.0, handout draft v0.9); P2 waits for GPU | no (checked 2026-09-19) | |
| 2 | Optimization 1: hyperparameter invariants | not started | no | |
| 3 | Optimization 2: sharp and flat basins | not started | no | |
| 4 | Architecture 1: extending model capacity | not started | example problem only (in `source/`) | |
| 5 | Architecture 2: stability across depth/time | not started | no | |
| 6 | Generalization: data-efficient algorithms | not started | no | |
| 7 | Exotic: ??? | not started | topic unannounced | |

Final grade = mean of best 5 of 7 quiz scores.

## Decisions

| Date | Decision |
|---|---|
| 2026-09-19 | Follow CS 312's units, order, domains (LM 1-4, DNA 5-7), unit loop and quiz format as-is. No tailoring to the rest of the repo. |
| 2026-09-19 | Pacing on demand: one unit at a time, each phase only when asked. Their calendar is reference only. |
| 2026-09-19 | GPU only when Fan Pu says it is idle; state GPU-hours before launching. |
| 2026-09-19 | Baseline architecture copied from their snippet; only width / tokens scaled for the GB10. |
| 2026-09-19 | When real course material is published it supersedes Claude's reconstruction for that unit. |
| 2026-09-19 | Division of labour: Fan Pu designs experiments, predicts, interprets, writes the report; Claude writes all code/configs, launches and monitors runs, plots (`COURSE.md` §1.6, §3). Menu mode only when he asks. |
| 2026-09-19 | Deliverables (handout, notes, quiz, debrief) are PDFs compiled from Typst with `template/template.typ` (day4 handout look, style only). Build with `template/build.sh`. |

## Open / pending

- Calibration (Unit 1 P2, needs GPU): baseline width, token count, vocab/tokenizer, sequence length, minutes per
  run, seed-noise floor (defines `=` in quizzes). Not yet measured.
- LM dataset: `../day3/data/fineweb_edu` checked 2026-09-19: 15 train shards + 1 val shard, 100M tokens each (uint16,
  GPT-2 tokenizer, llm.c format) = 1.5B train / 100M val. Enough for units 1-4. Open question for calibration:
  keep the ~50k GPT-2 vocab (embedding + LM head cost about as much as the body at our width) or retokenize smaller (CPU job).
- DNA dataset: decide at Unit 5 P0, preferably whatever CS 312 uses.
- Unit 5 "time" ambiguity (training time vs sequence length): resolve from their handout if available.
- Nothing in `dl-alchemy/` is committed to git yet (Fan Pu has not asked for a commit).

## Log

- 2026-09-19: Read the CS 312 site, saved snapshot to `source/`. Brainstormed the course with Fan Pu. First draft
  leaned on his current linear-attention work; he asked to follow their curriculum instead. He set pacing (one
  unit at a time, on request) and asked for all instructions/state to live here. Wrote `COURSE.md`, `STATE.md`.
  No GPU used (his training job was running).
- 2026-09-19: He asked for PDFs in the day4 handout style. Added `template/` (template.typ, build.sh, handout.csl,
  style_sample). Verified the sample compiles and renders; sent `template/style_sample.pdf`. Still no GPU use.
- 2026-09-19: He asked whether each unit comes with teaching on how to think about the questions (papers, prior
  results) so he is not guessing blind. Yes: expanded the `notes.pdf` spec in `COURSE.md` §4 (mental models, known
  results with verified citations, worked predictions, practice questions; read before the handout; 6-10 pages).
- 2026-09-19: He asked why he must run experiments himself; he is mid research sprint and wants to think, not code
  or babysit runs. Agreed split recorded in `COURSE.md` §1.6 and §3: he designs/predicts/interprets, Claude executes.
- 2026-09-19: Confirmed staff-side GPU work (calibration, sealed pilots, held-out runs with a baseline control,
  staff predictions recorded before each run); written into `COURSE.md` §4. Expect ~2-3 GPU nights per unit in total.
- 2026-09-19: He confirmed: follow CS 312 on DNA for units 5-7. Told him nothing blocks Unit 1; assumptions stated
  (notes pitched at someone who knows transformers/AdamW/muP basics; do not wait for Stanford's A1 release; GPU window
  of ~4-6 h needed after the P1 draft for calibration + pilots). Waiting for an explicit "start unit 1".
- 2026-09-19: "start unit 1". P0: site byte-identical to the morning snapshot, GitHub org has only the website repo.
  P1: wrote `unit1-basics/notes.typ/.pdf` (mental models: noisy quadratic, gradient noise scale, warmup, two-bottleneck
  scaling law; table of 14 checked papers; 3 worked predictions; practice questions) and `handout.typ/.pdf` (five
  problems: lr-bowl, batch, schedule, knobs, scaling; provisional baseline depth 8 / width 384 / T 512 / 300M tokens /
  batch 131k tokens). Template fixes: tables may break across pages; box headers stick to their body. GPU was at 96%
  from his own job all session; not touched.
- 2026-09-19 (later): he allowed GPU jobs alongside his MQAR sweep ("just dont oom the machine"). Built and tested
  `alchemy/`; measured throughput; retokenized FineWeb-Edu to BPE-8192 (`data/fineweb_edu_bpe8k/`, 1.78B train
  tokens, round trip verified); fixed the baseline at width 256 / 120M tokens; launched calibration 04:06. Updated
  handout (v0.95) and notes (v1.1) to the measured baseline.
