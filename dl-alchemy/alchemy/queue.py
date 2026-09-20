"""Run a list of jobs one at a time.  python -m alchemy.queue jobs.txt

Each non-empty, non-# line of jobs.txt:   <out_dir> <config.toml or -> [section.key=value ...]
Finished runs (result.json) are skipped, so re-launching the queue resumes it. Before each job the queue waits
until MemAvailable >= MIN_FREE_GB (other projects share this machine's unified memory) and stops if a file named
STOP exists next to jobs.txt. Launch detached:
    setsid nohup .venv/bin/python -m alchemy.queue path/jobs.txt > path/queue.log 2>&1 < /dev/null &
"""
import os
import subprocess
import sys
import time

MIN_FREE_GB = 35


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return 0.0


def main():
    jobs_path = os.path.abspath(sys.argv[1])
    stop = os.path.join(os.path.dirname(jobs_path), "STOP")
    jobs = [l.split() for l in open(jobs_path) if l.strip() and not l.startswith("#")]
    for i, (out, config, *sets) in enumerate(jobs):
        if os.path.exists(stop):
            print("STOP file found; exiting", flush=True)
            return
        if os.path.exists(os.path.join(out, "result.json")):
            continue
        while mem_available_gb() < MIN_FREE_GB:
            print(f"waiting for memory ({mem_available_gb():.0f} GB available)", flush=True)
            time.sleep(60)
        cmd = [sys.executable, "-m", "alchemy.train", "--out", out]
        if config != "-":
            cmd += ["--config", config]
        if sets:
            cmd += ["--set", *sets]
        print(f"[{i + 1}/{len(jobs)}] {time.strftime('%H:%M:%S')} {out}", flush=True)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "stdout.log"), "a") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
        print(f"    exit {rc}", flush=True)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
