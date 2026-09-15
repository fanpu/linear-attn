"""testbed: a small-scale controlled testbed for linear-attention mixers.

One package for the whole sprint; days add modules, never rename them.
Layout after fla-org/flash-linear-attention (reference beside fast, layers/
models/ops split) and karpathy/nanoGPT (model.py, data loading, train.py):

    model.py              model shapes, config builder, the SDPA shim for fla's attention   [AI]
    data.py               llm.c shard format, window loading, the seeded batch order       [AI]
    config.py             TOML run configs (torchtitan form) and the source hash           [AI]
    schedule.py           lr_at, the warmup-then-cosine schedule                            [you]
    data_check.py         check_disjoint, validation windows found in training              [you]
    evals/val_loss.py     evaluate, the batched token-weighted validation loss              [you]
    evals/naive.py        ref_nll, the unbatched reference evaluate is tested against       [you]
    analysis/seed_stats.py seed_stats, the pooled seed standard deviation and its range     [you]

[you] modules hold silent-failure code (a wrong schedule, a wrong loss average,
a wrong count, or a wrong standard deviation runs fine and hands you a
plausible number) and ship as stubs raising NotImplementedError. [AI] modules
hold loud-failure code and ship complete.
"""
