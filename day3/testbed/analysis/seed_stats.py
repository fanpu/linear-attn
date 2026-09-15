"""Seed statistics.  [you]"""


def seed_stats(losses_by_size: dict) -> dict:
    """Pooled seed standard deviation, its 80% range, and the minimum detectable difference.

    losses_by_size: {"30M": [x_0, x_1, ...], "60M": [...], ...}; each list holds the
        final validation losses (nats) of the seeds at that size. A size with fewer
        than 2 entries contributes nothing and is skipped.
    Returns {"s_pooled": float, "nu": int, "lo": float, "hi": float, "mdd": float} with
        s_pooled = sqrt( sum_i sum_j (x_ij - mean_i)^2 / sum_i (n_i - 1) ),
        nu       = sum_i (n_i - 1)                       (degrees of freedom),
        lo       = s_pooled / sqrt(chi2.ppf(0.9, nu) / nu),
        hi       = s_pooled / sqrt(chi2.ppf(0.1, nu) / nu),
        mdd      = 2.5 * s_pooled,
    where chi2.ppf is scipy.stats.chi2.ppf.
    """
    raise NotImplementedError
