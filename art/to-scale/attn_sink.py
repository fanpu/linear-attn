"""Attention-sink check for the softmax models: how much attention mass lands on position 0?

For each layer, mean over heads and over query positions >= 32 of the attention weight on key 0,
on 4 fineweb-edu documents (512 tokens each). Eager attention, bf16.
"""
import argparse
import json
import os

import numpy as np
import torch

from common import CACHE, SHORT, fineweb_docs, pasar_job
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n_docs", type=int, default=4)
args = ap.parse_args()
name = SHORT[args.model]
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
docs = fineweb_docs(0, args.n_docs)
per_layer = None
with torch.no_grad():
    for d, t in enumerate(docs):
        ids = tok(t, return_tensors="pt").input_ids[:, :512].cuda()
        out = model(input_ids=ids, output_attentions=True)
        m = np.array([a[0, :, 32:, 0].float().mean().item() for a in out.attentions])
        per_layer = m if per_layer is None else per_layer + m
        pasar_job.progress(d + 1, len(docs))
per_layer /= len(docs)
print(name, "attention on position 0 per layer:", np.round(per_layer, 3), "mean", per_layer.mean())
json.dump({"model": args.model, "attn_to_pos0": per_layer.tolist()}, open(os.path.join(CACHE, f"sink_{name}.json"), "w"))
