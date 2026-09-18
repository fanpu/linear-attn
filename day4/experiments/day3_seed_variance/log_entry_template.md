### Day 3, 2026-09-15
Question: How much does the random seed alone move a small transformer's final validation loss, at 30M/300M tokens today and at 60M/600M and 125M/1.2B in the queue?
Prediction (written BEFORE running): what number do I expect, and how confident?
Setup: attention baseline (fla TransformerForCausalLM, SDPA Flash), 30M/60M/125M (d=512/768/768, L=10/9/18), GPT-2 tokenizer, vocab 50304; AdamW 6e-4 peak, 3% warmup, cosine to 0.1x, B=32, T=1024; budgets 300M/600M/1.2B tokens; seeds {0, 1}; the seed sets the init and the order of a fixed window set; eval on the first 2^24 tokens of shard 0
Result: (one plot/table, path in repo)
Prediction correct? (yes / no / partially, and what surprised me)
What I now believe:
Spawned question(s):
Hours spent:
