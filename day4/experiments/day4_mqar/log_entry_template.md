### Day 4, 2026-09-18
Question: Does our MQAR harness reproduce the published picture at 512 tokens and 64 pairs: attention solves it at d=64, DeltaNet (delta rule) far above additive linear attention at d=128, and both linear mixers improving with state size?
Prediction (written BEFORE running): what number do I expect, and how confident?
Setup: MQAR, vocab 8192, 512 tokens, 64 key-value pairs, 20k train / 3k test examples (data seeds 123 / 124, fixed); 2-layer models, 2 heads, no MLP, no short conv, tied embeddings; attention at d=64, additive and delta at d=128, four learning rates {1e-4, 4.6e-4, 2.2e-3, 1e-2}, AdamW wd 0.1, cosine over 32 epochs, batch 128, early stop at test acc > 0.99; then d in {64, 256, 512} for both linear mixers at the best lr; seed 1 of the three best runs; bf16 autocast
Result: (one plot/table, path in repo)
Prediction correct? (yes / no / partially, and what surprised me)
What I now believe:
Spawned question(s):
Hours spent:
