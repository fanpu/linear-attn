"""TinyStories -> uint16 token streams in cache/ (Qwen3 tokenizer, remapped to the 8,192 most frequent ids).

    .venv/bin/python 04-lazy-rich-mup/prep_data.py
"""
import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
C = f"{HERE}/cache"
V = 8192
files = {"train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
         "val": "data/validation-00000-of-00001-869c898b519ad725.parquet"}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
eos = tok.convert_tokens_to_ids("<|endoftext|>")
raw = {}
for split, f in files.items():
    p = hf_hub_download("roneneldan/TinyStories", f, repo_type="dataset", local_dir=f"{C}/tinystories")
    texts = pd.read_parquet(p)["text"].tolist()
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    raw[split] = np.concatenate([np.array(x + [eos], dtype=np.int64) for x in ids])
    print(split, len(texts), "stories", len(raw[split]), "tokens")
counts = np.bincount(raw["train"])
top = np.argsort(-counts)[: V - 1]
remap = np.full(max(counts.size, raw["val"].max() + 1), V - 1, dtype=np.int64)  # V-1 = <unk>
remap[top] = np.arange(V - 1)
for split in raw:
    out = remap[np.minimum(raw[split], remap.size - 1)].astype(np.uint16)
    print(split, "unk rate", (out == V - 1).mean())
    out.tofile(f"{C}/ts_{split}.bin")
np.save(f"{C}/ts_vocab.npy", top)
