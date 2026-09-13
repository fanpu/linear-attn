# Attention as Weather

![The final learned attention field](outputs/attention-weather-final.png)

## What you are looking at

This is a single attention head in a character-level transformer trained from scratch on a small text corpus. Rows are the characters asking a question: *which earlier characters should shape my representation?* Columns are the earlier characters available to answer. The triangular blank region is causality: a character cannot attend to the future.

Each frame is captured during training. At the beginning, attention is almost uniform and uncommitted. As the loss falls, concentrated weather systems form along meaningful past locations. The colors are a literal heatmap of attention weight, not a simulated fluid.

## Why this matters

Attention is a routing mechanism. It lets the model draw information from a context-dependent place rather than use only the immediately preceding character. Even this tiny transformer discovers that language is not just local: recurring letters, spaces, and familiar fragments pull attention across distance.

## Reading the work

Cool navy means negligible attention. Cyan and amber indicate growing relevance; coral marks the most strongly selected context. The matrix is deliberately left legible, because the artwork's texture is also evidence.

## Reproduce

```bash
/home/fzeng/ml/research/.venv/bin/python render.py
```

The renderer saves the animation, final frame, raw attention tensors, and the training metadata in `outputs/`.

## Files

- [Looping animation](outputs/attention-weather.gif)
- [Final still](outputs/attention-weather-final.png)
- [Raw attention tensors](outputs/attention-data.npz)

## Production Edition

![The production master: four attention heads as a woven weather field](outputs/master.png)

This edition abandons the matrix as the hero object while preserving every relation it contained. A token's horizontal position records its place in the fixed 64-character context. For each query and each of the transformer's four learned attention heads, the three strongest causal links become curved filaments.

- Turquoise, gold, coral, and periwinkle distinguish the four real attention heads.
- A filament's opacity and width are its measured attention weight.
- The luminous pressure around a token is the total attention it receives from all later queries and heads.
- Node size uses the same received-attention quantity at a finer scale.
- The quiet gold line at the bottom advances through actual training checkpoints, not generated time.

The piece therefore shows attention as a changing routing field: relationships accumulate, fade, and specialize as the language model learns.

### Production Files

- [2560 x 1440 master still](outputs/master.png)
- [32-frame production loop](outputs/loop-production.gif)
- [All captured four-head attention tensors](outputs/production-data.npz)
- [Exact visual mapping and run metadata](outputs/production-metadata.json)

Render this edition with:

```bash
/home/fzeng/ml/research/.venv/bin/python render_production.py
```
