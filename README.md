# Linear Attention — 30-Day Research Sprint

A day-by-day sprint on linear attention: kernel correctness, numerics, and throughput of
gated linear-attention variants (Gated DeltaNet via [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention))
measured against standard attention.

Each `dayN/` directory is a self-contained experiment — its own script, its own results,
no shared framework. That is deliberate: kernel risk, framework risk, and training risk are
separate failure surfaces and shouldn't be debugged on the same day.

## Reference environment

The pinned versions in `pyproject.toml` / `requirements.lock.txt` reflect the machine this
work actually runs on:

| | |
|---|---|
| GPU | NVIDIA **GB10** (Grace Blackwell, compute capability **12.1** → `sm_121a`) |
| Arch / OS | `aarch64` Linux, driver 580.173.02 |
| CUDA toolkit | 13.0 (`/usr/local/cuda`, `ptxas` V13.0.88) |
| Python | 3.12.3 (`/usr/bin/python3.12`) |
| torch | 2.14.0+cu130 |
| triton | 3.8.0 |
| flash-linear-attention | 0.5.2 |

Nothing here is GB10-specific in principle, but the numerics results are, and the
`torch` wheel index below is CUDA-13.0-specific.

## Setup

### With `uv` (recommended)

```bash
uv sync                      # creates .venv/ and installs the pinned dependency set
uv sync --extra dev          # ...plus pytest + ruff
```

`pyproject.toml` declares only the five *direct* dependencies (`flash-linear-attention`,
`torch`, `numpy`, `pandas`, `matplotlib`); the other ~65 packages in the environment are
transitive. `torch` is pulled from an explicitly-scoped PyTorch CUDA 13.0 index — see
`[[tool.uv.index]]` — because `2.14.0+cu130` for aarch64 is not on PyPI. Every other
package resolves from PyPI as usual.

Run things with `uv run`, which activates the environment for you:

```bash
uv run python day1/kernel_check.py
```

### Reproducing the exact environment

`uv sync` re-resolves transitive dependencies, so it may pick up newer `transformers`,
`tokenizers`, etc. than the ones the recorded results were produced with. To reconstruct
the environment byte-for-byte, use the frozen list instead:

```bash
uv venv --python 3.12
uv pip install -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
```

### Without `uv` (plain pip)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

### Verify

```bash
python -c "import torch, fla; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
# -> 2.14.0+cu130 NVIDIA GB10 (12, 1)
```

### Per-session environment

Source this before running anything that compiles Triton kernels — it pins the target arch
to `sm_121a`, points Triton at the system `ptxas`, and caps build parallelism so
from-source builds don't OOM:

```bash
source day1/env.sh
```

## Running the experiments

### Day 1 — kernel correctness

Checks the `fla` chunked Gated DeltaNet kernel (forward + backward) against a naive fp32
sequential reference across dtypes, head dims, and sequence lengths.

```bash
source day1/env.sh
python day1/kernel_check.py | tee /tmp/log_tf32.txt
```

To separate TF32 error from true-fp32 error, run it a second time with TF32 disabled
globally (`NVIDIA_TF32_OVERRIDE=0`) and tag the two logs when plotting:

```bash
NVIDIA_TF32_OVERRIDE=0 python day1/kernel_check.py | tee /tmp/log_ieee.txt

python day1/plot.py --log tf32=/tmp/log_tf32.txt --log ieee=/tmp/log_ieee.txt \
                    --out day1/results
python day1/plot.py --csv day1/results.csv --out day1/results   # replot, no rerun needed
```

`plot.py` writes `*_summary.png`, `*_detail.png`, `*.csv` and `*_tables.md`.
Committed findings live in `day1/results.txt`.

### Day 2 — throughput and roofline

Attention (SDPA) vs Gated DeltaNet tokens/sec, against a measured compute/bandwidth roofline.

```bash
source day1/env.sh
python day2/throughput_benchmark.py roofline      # measured matmul TFLOP/s and copy GB/s
python day2/throughput_benchmark.py sdpa-check    # which SDPA backends actually run on sm_121
python day2/throughput_benchmark.py sweep         # 2 mixers x 4 sizes, fresh process per row
python day2/throughput_benchmark.py plot          # log-log tok/s vs params
```

## Layout

```
pyproject.toml           direct dependencies + uv index config
requirements.lock.txt    full `pip freeze` of the working environment
day1/
  env.sh                 CUDA/Triton env vars — source before running
  kernel_check.py        chunked GDN kernel vs naive fp32 reference
  plot.py                parses kernel_check logs -> figures, CSV, markdown tables
  results.txt            committed Day 1 error tables
day2/
  throughput_benchmark.py  roofline + tok/s sweep, attention vs GDN
```

## Notes

- `.gitignore` excludes `*.csv`, `*.jsonl` and checkpoints, so generated data artifacts
  (`day1/results.csv`, `day2/results.jsonl`) stay local. Figures are *not* ignored, so
  `git add` them deliberately. Anything worth keeping gets promoted into a committed
  `results.txt` / markdown table.
- The `.venv/` in this tree was originally created at `day1/.venv` and moved up a level,
  so the console-script shebangs under `.venv/bin/` (`pip`, `torchrun`, `transformers`, …)
  point at a path that no longer exists. `.venv/bin/python -m pip` works fine; bare
  `.venv/bin/pip` does not. A fresh `uv sync` fixes it.
