"""Coordinate check (Tensor Programs V, Appendix / mup package): activation sizes vs width during the first steps.

For SP and muP GPTs at widths 128..4096 we take a few Adam steps at one fixed learning rate and record, for a fixed
probe batch, the RMS coordinate size of each tapped activation x_t and of its change x_t - x_0.
Under muP every curve is flat in width; under SP the changes grow with width.

    _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/coord_check.py
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mup_gpt import GPT, Cfg
from train_lm import batch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.cuda.set_per_process_memory_fraction(0.08)
dev = "cuda"
WIDTHS = [128, 256, 512, 1024, 2048, 4096]
STEPS = 5
LR = 2 ** -8  # 3.9e-3, near the base-width optimum
TAPS = ["embed", "attn0", "mlp0", "attn3", "mlp3", "logits"]


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


out = {}
for param in ["sp", "mup"]:
    for d in WIDTHS:
        for seed in range(3):
            model = GPT(Cfg(d=d, L=4, param=param, ctx=128)).to(dev)
            model.reset(seed)
            opt = torch.optim.AdamW(model.param_groups(LR), betas=(0.9, 0.95), weight_decay=0.0)
            xp, _ = batch("val", 99_000 + seed, 8, 128)
            rec = {k: [] for k in TAPS}
            drec = {k: [] for k in TAPS}
            base = None
            for t in range(STEPS + 1):
                taps = {}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(xp, taps)
                taps = {k: v.float() for k, v in taps.items()}
                if base is None:
                    base = taps
                for k in TAPS:
                    rec[k].append(rms(taps[k]))
                    drec[k].append(rms(taps[k] - base[k]))
                if t == STEPS:
                    break
                x, y = batch("train", 500 + t, 16, 128, seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            out[f"{param}_{d}_{seed}"] = dict(param=param, d=d, seed=seed, rms=rec, drms=drec)
            print(param, d, seed, {k: f"{v[-1]:.3g}" for k, v in drec.items()}, flush=True)
            del model, opt
            torch.cuda.empty_cache()
json.dump(dict(widths=WIDTHS, steps=STEPS, lr=LR, taps=TAPS, runs=out), open(f"{HERE}/cache/coord_check.json", "w"))
