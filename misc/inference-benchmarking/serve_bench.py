"""Phase 2: online serving latency under load.

The offline sweep measures throughput, but TTFT, TPOT and tail latency only mean
something when requests *arrive* over time rather than as one batch handed to the
engine at once. So this phase runs a real server and drives it with vLLM's own
`vllm bench serve`, which is the tool these numbers are normally quoted from --
using it keeps the results comparable with published figures instead of depending
on a client I wrote myself.

Memory safety is the same contract as the offline sweep: the server's
gpu_memory_utilization comes from budget.py, and a Watchdog kills the whole
process group if system memory availability falls below the floor.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import budget
import guard

RESULTS = Path("results/serving.jsonl")
PORT = 8077
OUT_LEN = 128
IN_LEN = 1024

# Concurrency levels to probe. Each is a separate client run against one server.
CONCURRENCIES = [1, 4, 16, 64]


def wait_for_health(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    """Poll /health until the server answers, the process dies, or we give up."""
    import httpx

    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def start_server(model: str, util: float, max_model_len: int, max_num_seqs: int):
    argv = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
        "--port", str(PORT),
        "--gpu-memory-utilization", f"{util:.4f}",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        # Same reason as the offline sweep: prefix caching would serve repeated
        # prompts from cache and flatter the latency numbers. ("--disable-log-requests"
        # does not exist in vLLM 0.23 and would abort the server at startup.)
        "--no-enable-prefix-caching",
    ]
    # See sweep.py: cap JIT build parallelism so kernel compilation cannot
    # exhaust host memory on a unified-memory machine.
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", MAX_JOBS="4")
    return subprocess.Popen(argv, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_client(model: str, concurrency: int, num_prompts: int, tmpdir: Path) -> dict:
    """One `vllm bench serve` run; returns its parsed result JSON."""
    fname = f"c{concurrency}.json"
    argv = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--model", model,
        "--host", "127.0.0.1", "--port", str(PORT),
        "--dataset-name", "random",
        "--random-input-len", str(IN_LEN),
        "--random-output-len", str(OUT_LEN),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result",
        "--result-dir", str(tmpdir),
        "--result-filename", fname,
    ]
    res = guard.run_guarded(argv, timeout_s=1800)
    path = tmpdir / fname
    if path.exists():
        return json.loads(path.read_text())
    return {"error": res.status(), "tail": res.output[-1000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrencies", default=",".join(map(str, CONCURRENCIES)))
    ap.add_argument("--num-prompts", type=int, default=200)
    args = ap.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    spec = budget.ModelSpec.from_hf_cache(args.model)

    max_conc = max(concurrencies)
    plan = budget.plan_cell(spec, batch=max_conc, total_len=IN_LEN + OUT_LEN)
    if not plan.fits:
        print(f"refusing to serve {args.model} at concurrency {max_conc}: {plan.reason}")
        return 1
    print(f"{args.model}: {plan.summary()}")

    proc = start_server(args.model, plan.gpu_memory_utilization,
                        IN_LEN + OUT_LEN, max_conc)
    tmpdir = Path("results/serving_raw")
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        with guard.Watchdog(proc) as wd:
            if not wait_for_health(PORT, proc, timeout_s=600):
                print("server failed to become healthy"
                      + (" (watchdog tripped)" if wd.tripped else ""))
                return 1
            print("server healthy; running client sweeps")

            RESULTS.parent.mkdir(exist_ok=True)
            for c in concurrencies:
                # Enough requests to reach steady state at each concurrency
                # without spending half an hour on c=1.
                n = max(8, min(args.num_prompts, 4 * c))
                r = run_client(args.model, c, n, tmpdir)
                row = {"model": args.model, "concurrency": c,
                       "in_len": IN_LEN, "out_len": OUT_LEN,
                       "num_prompts": n,
                       "gpu_memory_utilization": plan.gpu_memory_utilization,
                       "watchdog_tripped": wd.tripped,
                       "result": r}
                with RESULTS.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                tp = r.get("request_throughput")
                ttft = r.get("p99_ttft_ms")
                print(f"  c={c:<4d} throughput={tp if tp is None else f'{tp:.2f}'} req/s"
                      f"  p99 TTFT={ttft if ttft is None else f'{ttft:.0f}'} ms")
                if wd.tripped:
                    print("  watchdog tripped; stopping")
                    break
    finally:
        if proc.poll() is None:
            guard._kill_group(proc)
        print("server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
