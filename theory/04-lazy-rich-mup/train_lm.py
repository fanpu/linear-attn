"""Train the SP / muP GPT on TinyStories. Resumable; one JSON result per run in cache/lm/.

    .venv/bin/python 04-lazy-rich-mup/train_lm.py smoke
    _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/train_lm.py sweep width     # main muTransfer sweep
    _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/train_lm.py sweep depth     # depth extension
"""
import json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mup_gpt import GPT, Cfg, lr_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/cache/lm"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

_data = {}


def data(split):
    if split not in _data:
        _data[split] = torch.from_numpy(np.fromfile(f"{HERE}/cache/ts_{split}.bin", dtype=np.uint16).astype(np.int64))
    return _data[split]


def batch(split, step, B, T, seed=0):
    """Deterministic batch for (split, step): identical data order for every run (common random numbers)."""
    d = data(split)
    g = torch.Generator().manual_seed((0 if split == "train" else 1) * 10 ** 9 + step * 1000 + seed)
    ix = torch.randint(0, len(d) - T - 1, (B,), generator=g)
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i + 1:i + T + 1] for i in ix])
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


@torch.no_grad()
def evaluate(model, n_batches=40, B=32, T=128):
    model.eval()
    tot = 0.0
    for i in range(n_batches):
        x, y = batch("val", 10_000_000 + i, B, T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return tot / n_batches


def run_name(param, d, L, lr, steps, seed=0, depth_mup=False):
    return f"{param}{'_dmup' if depth_mup else ''}_d{d}_L{L}_lr{lr:.2e}_T{steps}_s{seed}"


def train(param, d, L, lr, steps, seed=0, depth_mup=False, B=32, T=128, ckpt_every=250, evals=(), log=print,
          compile=True, _crash_at=None):
    name = run_name(param, d, L, lr, steps, seed, depth_mup)
    res_path, ck_path = f"{OUT}/{name}.json", f"{OUT}/{name}.ckpt"
    if os.path.exists(res_path):
        return json.load(open(res_path))
    torch.manual_seed(seed)
    c = Cfg(d=d, L=L, param=param, depth_mup=depth_mup, ctx=T)
    model = GPT(c).to(dev)
    model.reset(seed)
    W0 = {n: p.detach().clone() for n, p in model.hidden_matrices()}
    W0["head"] = model.head.weight.detach().clone()
    groups = model.param_groups(lr)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    step_loss = torch.compile(loss_fn) if compile else loss_fn
    hist = {"step": [], "train": [], "val_step": [], "val": []}
    start = 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        hist, start, W0 = ck["hist"], ck["step"], ck["W0"]
        log(f"resumed {name} at step {start}")
    t0 = time.time()
    diverged = False
    ema = None
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_at(step, steps)
        x, y = batch("train", step, B, T, seed)
        loss = step_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if _crash_at is not None and step == _crash_at:
            raise RuntimeError("simulated crash")
        if step % 10 == 0 or step == steps - 1:
            lv = loss.item()
            if not math.isfinite(lv) or lv > 12:
                diverged = True
                log(f"{name}: diverged at step {step} (loss {lv})")
                break
            hist["step"].append(step); hist["train"].append(lv)
        if (step + 1) in evals:
            hist["val_step"].append(step + 1); hist["val"].append(evaluate(model, n_batches=10))
        if (step + 1) % ckpt_every == 0 and step + 1 < steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "step": step + 1, "W0": W0},
                       ck_path)
    res = dict(name=name, param=param, d=d, L=L, lr=lr, steps=steps, seed=seed, depth_mup=depth_mup,
               tokens=steps * B * T, hist=hist, diverged=diverged, secs=time.time() - t0)
    if not diverged:
        res["val"] = evaluate(model)
        move = {}
        with torch.no_grad():
            cur = dict(model.hidden_matrices()); cur["head"] = model.head.weight
            for n, w0 in W0.items():
                dw = (cur[n].float() - w0.float())
                move[n] = dict(dop=torch.linalg.matrix_norm(dw, ord=2).item(),
                               w0op=torch.linalg.matrix_norm(w0.float(), ord=2).item(),
                               dfro=dw.norm().item(), w0fro=w0.float().norm().item())
        res["move"] = move
    else:
        res["val"] = float("nan")
    json.dump(res, open(res_path, "w"))
    if os.path.exists(ck_path):
        os.remove(ck_path)
    log(f"{name}: val {res['val']:.4f} ({res['secs']:.0f}s)")
    return res


def sweep(kind, shard=0, n_shards=1):
    logf = open(f"{OUT}/sweep_{kind}_{shard}.log", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    runs = []
    if kind == "width":
        lrs = [2 ** k for k in range(-12, -4)]  # 2.4e-4 ... 3.1e-2
        for d in [128, 256, 512, 1024]:
            for param in ["mup", "sp"]:
                if d == 128 and param == "sp":
                    continue  # identical to muP at the base width
                runs += [dict(param=param, d=d, L=4, lr=lr, steps=1000) for lr in lrs]
    elif kind == "depth":
        lrs = [2 ** k for k in range(-11, -4)]  # 4.9e-4 ... 3.1e-2
        for L in [2, 4, 8, 16]:
            for dm in [True, False]:
                if L == 4 and dm:
                    continue  # identical at the base depth (multiplier 1, same LR): reuse the width-sweep runs
                runs += [dict(param="mup", d=128, L=L, lr=lr, steps=1000, depth_mup=dm) for lr in lrs]
    elif kind == "budget":
        lrs = [2 ** k for k in range(-13, -5)]
        for steps in [375, 6000]:
            runs += [dict(param=p, d=256, L=4, lr=lr, steps=steps) for p in ["mup"] for lr in lrs]
    for r in runs[shard::n_shards]:
        train(**r, log=log)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.08)
    if sys.argv[1] == "smoke":
        r = train("mup", 256, 2, 1e-3, 30, ckpt_every=10, evals=(20,))
        print(r["val"], r["hist"]["val"], list(r["move"].items())[:2])
        os.remove(f"{OUT}/{r['name']}.json")
        # resume path: crash at step 25 (checkpoint at 20), resume, and compare with an uninterrupted run
        try:
            train("sp", 128, 2, 3e-3, 40, ckpt_every=10, _crash_at=25)
        except RuntimeError as e:
            print("crashed as planned:", e)
        r1 = train("sp", 128, 2, 3e-3, 40, ckpt_every=10)
        os.remove(f"{OUT}/{r1['name']}.json")
        r2 = train("sp", 128, 2, 3e-3, 40, ckpt_every=1000)
        os.remove(f"{OUT}/{r2['name']}.json")
        print("resumed val", r1["val"], "uninterrupted val", r2["val"], "hist equal:", r1["hist"]["train"][-1], r2["hist"]["train"][-1])
    elif sys.argv[1] == "time":
        for d in [128, 256, 512, 1024]:
            r = train("mup", d, 4, 1e-3, 60)
            print(d, r["secs"] / 60, "s/step")
            os.remove(f"{OUT}/{r['name']}.json")
    elif sys.argv[1] == "sweep":
        sweep(sys.argv[2], *(int(a) for a in sys.argv[3:5]))
