"""Train a sequence model (softmax / linear / delta / gdelta) on in-context linear regression, predicting every y_t
from the pairs before it.  One model per invocation; run several in parallel through _shared/gpu_run.sh.

  .venv/bin/python 08-icl-linear-attention/train_seq.py --kind delta --layers 2 --sigma 0
"""
import argparse, json, math, pathlib, time
import numpy as np
import torch
from seqmodels import ICLModel

p = argparse.ArgumentParser()
p.add_argument("--kind", default="delta")
p.add_argument("--layers", type=int, default=2)
p.add_argument("--width", type=int, default=64)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--d", type=int, default=10)
p.add_argument("--N", type=int, default=40)
p.add_argument("--sigma", type=float, default=0.0)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--bench", action="store_true")
a = p.parse_args()
torch.cuda.set_per_process_memory_fraction(0.08)
dev = "cuda"
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
model = ICLModel(a.kind, a.d, a.width, a.heads, a.layers).to(dev)
nparam = sum(t.numel() for t in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, fused=True)
warm = max(1, a.steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
name = f"seq_{a.kind}_L{a.layers}_w{a.width}_d{a.d}_N{a.N}_s{a.sigma}_seed{a.seed}"
cache = pathlib.Path(__file__).parent / "cache"


def batch(B):
    X = torch.randn(B, a.N, a.d, device=dev, generator=gen)
    w = torch.randn(B, a.d, device=dev, generator=gen)
    f = torch.einsum("bnd,bd->bn", X, w)
    return X, f + a.sigma * torch.randn(B, a.N, device=dev, generator=gen), f


step_fn = torch.compile(model)
log = {"args": vars(a), "nparam": nparam, "loss": [], "excess": []}
t0 = time.time()
for step in range(a.steps):
    X, y, f = batch(a.batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = step_fn(X, y).float()
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if a.bench and step == 20:
        torch.cuda.synchronize(); t1 = time.time()
    if a.bench and step == 80:
        torch.cuda.synchronize(); print(f"{name}: {(time.time()-t1)/60*1000:.1f} ms/step, {nparam} params"); raise SystemExit
    if step % 100 == 0:
        with torch.no_grad():
            log["loss"].append((step, loss.item()))
            log["excess"].append((step, ((pred - f) ** 2).mean(0).tolist()))
    if step % 1000 == 0:
        print(f"{name} step {step} loss {loss.item():.4f} excess(last) {((pred[:, -1]-f[:, -1])**2).mean().item():.4f} {time.time()-t0:.0f}s", flush=True)
torch.save({"args": vars(a), "state": model.state_dict()}, cache / f"{name}.pt")
json.dump(log, open(cache / f"{name}.json", "w"))
print("saved", name, f"{time.time()-t0:.0f}s")
