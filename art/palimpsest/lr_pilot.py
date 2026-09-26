"""Learning-rate pilot: fit page A for --steps steps at a grid of learning rates, report PSNR and
seconds per step. One job per (arch, optimiser). Used only to choose the sweep's learning rates."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import pasar_job
from pages import load
from train import make_model, coords

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--opt", default="sgd")
    p.add_argument("--lrs", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--sigma", type=float, default=16.0)
    p.add_argument("--nff", type=int, default=256)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--mom", type=float, default=0.9)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()
    dev = "cuda" if pasar_job.job_id() is not None and torch.cuda.is_available() else "cpu"
    pasar_job.apply_memory_limit()
    n = a.res
    A = torch.as_tensor(load(n)["A"], device=dev).reshape(-1, 1)
    X = coords(n, a, device=dev)
    lrs = [float(x) for x in a.lrs.split(",")]
    res = []
    for i, lr in enumerate(lrs):
        torch.manual_seed(0)
        model = make_model(a, torch.Generator().manual_seed(0)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr) if a.opt == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=a.mom)
        t0 = time.time()
        curve = []
        for s in range(a.steps):
            opt.zero_grad()
            loss = ((model(X) - A) ** 2).mean()
            loss.backward()
            opt.step()
            if s % 100 == 99:
                curve.append(-10 * math.log10(max(loss.item(), 1e-12)))
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / a.steps
        r = dict(lr=lr, psnr_curve=[round(c, 2) for c in curve], sec_per_step=dt)
        print(json.dumps(r), flush=True)
        res.append(r)
        pasar_job.progress(i + 1, len(lrs))
    tag = {"ff": f"ff{a.sigma:g}", "siren": f"siren{a.omega:g}", "relu": "relu"}[a.arch]
    os.makedirs(os.path.join(HERE, "cache", "pilot"), exist_ok=True)
    with open(os.path.join(HERE, "cache", "pilot", f"lr_{tag}_w{a.width}_{a.opt}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
