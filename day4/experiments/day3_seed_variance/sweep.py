#!/usr/bin/env python
"""The six-run seed-variance sweep as a list of (config, seed) pairs.  [AI-owned]
# after HazyResearch/zoology zoology/experiments/*.py: a sweep is a Python file
# that builds a list of configs; run.sh executes it top to bottom.

    python experiments/day3_seed_variance/sweep.py     # prints one "config seed" line per run

Order matters: the 30M pair first so that it finishes inside the working day.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ["30M", "60M", "125M"]
SEEDS = [0, 1]

configs = [(os.path.join(HERE, f"attn_{size}.toml"), seed) for size in SIZES for seed in SEEDS]

if __name__ == "__main__":
    for path, seed in configs:
        print(os.path.relpath(path), seed)
