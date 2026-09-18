# testbed: one repository for the sprint (Day 4 snapshot)

One repository, one package (`testbed/`), one snapshot per day. Day 4 adds the
synthetic-recall (MQAR) harness on top of the Day 3 checkout: the MQAR data
generator, a two-layer model with a pluggable token mixer (attention, additive
linear attention, DeltaNet, gated DeltaNet), the reference recurrences you
write, the training script, and the day's sweep. Companion handouts are in
`handout/` (`day3_assignment.pdf`, `day4_assignment.pdf`).

`tests/adapters.py` carries your five filled-in Day 3 adapters and four new
Day 4 stubs (`run_naive_linear_attn`, `run_naive_delta_rule`,
`run_mqar_accuracy`, `run_state_elements`).

## Setup (Day 4 additions)

```
uv sync
export TORCH_CUDA_ARCH_LIST="12.1a"     # GB10 (sm_121)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

Pinned: `torch 2.14.0+cu130`, `flash-linear-attention 0.5.2` (the version in
which the chunked delta-rule kernels were verified on this box), Python 3.12.

## Run the tests (Day 4)

```
uv run pytest -m "not gpu"     # CPU subset: the five Day 3 tests pass; the four new [you] tests fail with NotImplementedError
uv run pytest                  # also the GPU tests: your references against fla's kernels, and determinism
uv run pytest -k test_naive_delta_rule
```

Tests never import your code directly; they go through `tests/adapters.py`.

## Day 4 scripts

| command | what it does | GB10 time (estimate) |
|---|---|---|
| `python scripts/look_at_mqar.py --config experiments/day4_mqar/attn_d64.toml` | prints two examples and ten random queries with their gaps | seconds |
| `python scripts/look_at_mqar.py --run runs/<name>` | ten random failed queries of a finished run, with gap and key index statistics | seconds |
| `python scripts/train_mqar.py --config <toml> --seed 0 [--out runs/<name>]` | one training run; writes `runs/<mixer>_d<d>_lr<lr>_s<seed>/` | 1–10 min, see the handout |
| `python experiments/day4_mqar/sweep.py <stage>` | lists the runs of a stage and writes their TOML files to `experiments/day4_mqar/generated/` | seconds |
| `nohup experiments/day4_mqar/run.sh <stage> > run_<stage>.log 2>&1 &` | runs a stage; stages `smoke`, `lr`, `scan`, `scan512` (the two $d=512$ scan runs only), `seeds`, `gate`, `pivot` (attention at d=64 with 100k examples, the handout's pivot) | smoke ≤ 10 min, lr ≤ 75 min, scan ≤ 60 min, seeds ≤ 25 min, gate ≤ 45 min |
| `python experiments/day4_mqar/plot.py [--runs runs]` | the table and `experiments/day4_mqar/mqar_recall.png` | seconds |
| `python examples/retrieval_toy.py` | the additive-versus-delta retrieval toy | seconds |

A run directory holds `config.toml` (verbatim copy), `env.json` (torch,
fla, triton versions, GPU, source hash), `metrics.jsonl` (a `meta` record,
then one `train` and one `eval` record per epoch), `eval_preds.npz` (the
last test pass, for the failure inspection), `model.pt` (the final weights), `stdout.log`, and
`summary.json` when the run completes. A run whose `summary.json` exists is
skipped, so a stage can be re-launched after an interruption.

## Day 4 layout and where each piece comes from

| path | counterpart |
|---|---|
| `testbed/tasks/mqar.py` | `zoology/data/multiquery_ar.py` (Zoology keeps it under `data/`; ours is `tasks/` because `testbed/data.py` is the Day 3 shard loader) |
| `testbed/ops/<op>/naive.py` (you), `chunk.py` (wrapper on fla) | `fla/ops/<op>/naive.py` beside `chunk.py` |
| `testbed/layers/attn.py`, `layers/linear_attn.py` | `zoology/mixers/attention.py`; `fla/layers/delta_net.py` |
| `testbed/models/mqar_lm.py` | `zoology/model.py` |
| `testbed/evals/mqar_accuracy.py` (you) | `zoology/train.py::compute_metrics` |
| `testbed/analysis/state_elements.py` (you) | `zoology/model.py::_compute_state_size` |
| `scripts/train_mqar.py` | `zoology/train.py` protocol in `nanoGPT/train.py` shape |
| `scripts/look_at_mqar.py` | none; the "look at the data / look at the failures" step of the handout |
| `experiments/day4_mqar/*.toml`, `sweep.py`, `run.sh`, `plot.py` | torchtitan run files; `zoology/experiments/.../configs.py`; `nanochat/speedrun.sh`; `zoology/analysis/` |
| `experiments/day4_mqar/sample_runs/` | Day 3's `sample_runs/`: synthetic runs in the exact output format, for warming up `plot.py` |
| `tests/` | `cs336` starters (`adapters.py`) and `fla/tests/ops/` (parameterized, explicit tolerances, determinism) |
| `handout/` | the Typst template bundle (unchanged except `refs.bib`) plus both days' handouts |


Companion to `day3_assignment.pdf`. One repository for the whole sprint; each
day's zip is the previous day's plus that day's additions (see the changelog).
Everything below runs from this directory.

## Day 3: transformer baselines and the seed-variance yardstick

### Setup

We use `uv` to manage dependencies, as the CS336 starters do.

    curl -LsSf https://astral.sh/uv/install.sh | sh     # once
    uv sync                                              # creates .venv from pyproject.toml
    export TORCH_CUDA_ARCH_LIST="12.1a"                  # GB10 (sm_121); also TRITON_PTXAS_PATH if your box needs it

Run every command below with `uv run <command>`, or activate `.venv` first.
The tests run on CPU and need neither `fla` nor a GPU. On the DGX Spark, `torch 2.14.0+cu130` and
`flash-linear-attention 0.5.2` are the versions the kernel check and the throughput benchmark used.

### Run the tests

    pytest                       # all five
    pytest -k test_lr_at         # one Problem; likewise test_evaluate, test_ref_nll, test_check_disjoint, test_seed_stats

Initially all five tests fail with `NotImplementedError`. To connect the tests to your code, complete the
functions in `tests/adapters.py`; the tests themselves are never edited. The tests are the ones the
handout names in each Problem's "To test your implementation" line.

### Layout, and which public codebase each piece follows

    testbed/                  the package (stable across the sprint). Layout after fla-org/flash-linear-attention and karpathy/nanoGPT.
      model.py                shapes, config builder, SDPA shim                       [AI]  (fla/models/transformer, nanoGPT model.py)
      data.py                 llm.c shard format, windows, seeded batch order          [AI]  (llm.c dev/data, nanoGPT get_batch)
      config.py               TOML run configs and the source hash                     [AI]  (torchtitan config_manager.py, modded-nanogpt)
      schedule.py             lr_at                                                    [you] (nanoGPT get_lr)
      data_check.py           check_disjoint                                           [you]
      evals/val_loss.py       evaluate                                                 [you] (nanoGPT estimate_loss)
      evals/naive.py          ref_nll, the reference beside the fast version           [you] (fla ops/<op>/naive.py pattern)
      analysis/seed_stats.py  seed_stats                                               [you] (zoology/analysis)
    tests/                    one test per [you] function, plus adapters.py           (cs336 assignment1-basics tests/)
    scripts/                  verbs: prepare_data, look_at_shards, check_data, train, summarize   (nanochat scripts/, nanoGPT train.py)
    experiments/day3_seed_variance/   today's runs: one TOML per run, the sweep, run.sh,
                              the log-entry skeleton, and sample_runs/ (synthetic)      (zoology experiments/, torchtitan train_configs/, nanochat speedrun.sh)
    examples/                 shard_roundtrip.py and simulate_shat.py, the worked examples the handout walks through
                              (no exemplar counterpart: CS336 puts worked examples in the PDF; we ship them runnable as well)
    handout/                  the Typst template bundle the handout was built from (no exemplar counterpart; shipped so the PDF is reproducible)
    pyproject.toml            uv-managed, pinned                                       (cs336 assignment1-basics)

Run directories (written by `scripts/train.py`, after modded-nanogpt's logs): `runs/<mixer>_<size>_s<seed>/` holds
`config.toml` (a copy of the launch config), `env.json` (torch, triton, fla versions, GPU name, source hash),
`metrics.jsonl` (one record per logged step; field names are the handout's), `stdout.log`, checkpoints, and
`summary.json` when the run completes. A number that cannot be traced to a run directory does not go in a handout.

Config files follow torchtitan's section and key names where a counterpart exists (`training.local_batch_size`,
`training.seq_len`, `training.max_norm`, `optimizer.lr`, `metrics.log_freq`, `job.dump_folder`). Where torchtitan
counts in steps we count in fractions of the run (`validation.every_frac`, `checkpoint.every_frac`,
`lr_scheduler.warmup_frac`), because the three sizes have different step counts and one fraction gives them the
same schedule shape. `attn_30M_smoke.toml` plays the role of torchtitan's `debugmodel` preset: it is `attn_30M.toml`
with `training.steps = 300`, so a smoke run is the exact prefix of the real run.

### Data

    nohup python scripts/prepare_data.py --out data/fineweb_edu --num-shards 16 > prep.log 2>&1 &   # 20-40 min (estimate), 3.2 GB
    python scripts/look_at_shards.py --data data/fineweb_edu --stats     # seconds
    python scripts/check_data.py --data data/fineweb_edu                 # about 10 s (1.46M fingerprints), needs your check_disjoint

### Scripts, in the order the handout uses them

    python examples/shard_roundtrip.py                                   # seconds
    python examples/simulate_shat.py                                     # seconds
    python scripts/summarize.py --runs experiments/day3_seed_variance/sample_runs   # warm-up on the synthetic runs, needs your seed_stats
    python scripts/train.py --config experiments/day3_seed_variance/attn_30M_smoke.toml --seed 0 --out runs/smoke_a   # about 3 min (estimate)
    python scripts/train.py --config experiments/day3_seed_variance/attn_30M_smoke.toml --seed 0 --out runs/smoke_b
    nohup experiments/day3_seed_variance/run.sh > run.log 2>&1 &       # six runs, about 31 GPU-hours total (estimate)
    python scripts/summarize.py                                          # any time; running runs appear in the plot

Times marked "estimate" are computed from the throughput benchmark's measured attention throughput
(77,546 / 59,511 / 32,926 tok/s at 30M / 60M / 125M with a 32,000 vocabulary) scaled by the larger head today:
30M about 1.2 h per seed, 60M about 3.1 h, 125M about 11 h. Nothing in this repository has yet been timed on the
GB10 with today's vocabulary; the smoke run is where the estimates are checked.

Checkpoints: `ckpt_XX.pt` (weights only, bf16) every 10% of a run, 0.17 GB each at 30M and 0.41 GB at 125M,
about 17 GB for all six runs; `ckpt_latest.pt` (weights + optimizer, fp32) for resuming, deleted when a run completes.
`nvidia-smi` shows N/A for memory on the GB10 (unified memory); read `peak_mem_GB` from `metrics.jsonl` instead.
If the box dies, re-launch the same `run.sh` command: finished runs are skipped and the interrupted run resumes
from its last 10% checkpoint.

## What lives where

The repository carries code, configs, and one results figure per day (`experiments/dayN_*/`). Run directories (`runs/`), data shards, checkpoints, and logs stay on the box and are gitignored; every number in a handout traces to a run directory there. Plots and scripts made for a blog post live in a `blog/` folder beside the repository, not in it.

## Changelog

- **2026-09-18 (Day 4, cleanup).** Removed from the tree: `prep.log`, `run.log`, `runs-smoke/` (now `runs/smoke_a`, `runs/smoke_b` on the box), and the blog-only figures and `scripts/plot_pred_vs_actual.py` (moved to `blog/`). Added `.gitignore`.
- **2026-09-18 (Day 4).** Added `testbed/tasks/mqar.py`, `testbed/ops/{linear_attn,delta_rule,gated_delta_rule}/`, `testbed/layers/{attn,linear_attn}.py`, `testbed/models/mqar_lm.py`, `testbed/evals/mqar_accuracy.py`, `testbed/analysis/state_elements.py`, `scripts/train_mqar.py`, `scripts/look_at_mqar.py`, `examples/retrieval_toy.py`, `experiments/day4_mqar/`, the tests `test_naive_linear_attn`, `test_naive_delta_rule`, `test_chunk_ops` (GPU), `test_mqar_accuracy`, `test_state_elements`, `test_mqar_data`, `tests/__init__.py`, four `run_*` adapters, the `gpu` pytest marker, `handout/day4_assignment.{typ,pdf}`; `refs.bib` gains three entries. `pyproject.toml` gains `einops` and `tomli-w`, version 0.4.0. Nothing moved or renamed.
- 2026-09-15 (Day 3). Package created as `testbed/` (the name is fixed for the sprint) with `model.py` and `data.py`
  carried over from the throughput benchmark, plus `config.py`, `schedule.py`, `data_check.py`, `evals/`, and
  `analysis/`. Added `scripts/` (five verbs), `tests/` with `adapters.py`, `experiments/day3_seed_variance/`
  (three run configs, the smoke preset, `sweep.py`, `run.sh`, `log_entry_template.md`, `sample_runs/`),
  `examples/`, `pyproject.toml`. Run directories are `runs/<mixer>_<size>_s<seed>/` with `config.toml`,
  `env.json`, `metrics.jsonl`, `stdout.log`.
