# Linear Attention — 30-Day Research Sprint

A day-by-day research sprint on linear attention.

Each `dayN/` directory is a self-contained experiment — its own script, its own results,
no shared framework. 

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
