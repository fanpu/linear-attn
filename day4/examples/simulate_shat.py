#!/usr/bin/env python
"""Toy: how much does a two-run standard-deviation estimate wobble?  [complete and runnable]

python examples/simulate_shat.py
"""

import numpy as np

rng = np.random.default_rng(0)
sigma = 1.0
x1, x2 = rng.normal(0, sigma, 10_000), rng.normal(0, sigma, 10_000)
s_hat = np.abs(x1 - x2) / np.sqrt(2)
p10, p90 = np.percentile(s_hat, [10, 90])
print(f"true sigma = {sigma}")
print(f"10th percentile of s_hat = {p10:.3f}, 90th = {p90:.3f}  (ratio {p90/p10:.1f}x)")
print(
    f"fraction of pairs with s_hat < 0.5 sigma: {(s_hat < 0.5 * sigma).mean():.3f}; > 1.5 sigma: {(s_hat > 1.5 * sigma).mean():.3f}"
)
