"""Train a coordinate network on page A (or C, or nothing), then on page B; watch A fade.

phase 1: fit `--first` (A | C | none) for --steps1 full-batch steps.
phase 2: fresh optimiser, fit B for --steps2 steps. The output on the pixel grid is saved at
         ~--nsnap log-spaced steps, and the per-band ghost of every template page is measured
         in float32/float64 at each of them (metrics.Ghost).
phase 3: "savings". From the final B network, fresh optimiser, fit A for --relearn steps and log
         PSNR(A) at every step. Compared across A->B, C->B and none->B this says whether the
         weights still hold A even where the output does not.

Every run is one pasar job. Device is CPU unless running under pasar.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import pasar_job
from pages import load
from metrics import Ghost

HERE = os.path.dirname(os.path.abspath(__file__))


class FF(nn.Module):
    """Gaussian random Fourier features (Tancik et al. 2020), x in [0,1]^2, B ~ N(0, sigma^2)."""

    def __init__(self, sigma, nff, gen):
        super().__init__()
        self.register_buffer("B", torch.randn(nff, 2, generator=gen) * sigma)

    def forward(self, x01):
        p = 2 * math.pi * x01 @ self.B.T
        return torch.cat([torch.cos(p), torch.sin(p)], -1)


class Sine(nn.Module):
    def __init__(self, lin, w0):
        super().__init__()
        self.lin, self.w0 = lin, w0

    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))


def make_model(a, gen):
    if a.arch == "siren":
        layers = []
        d_in = 2
        for i in range(a.depth):
            lin = nn.Linear(d_in, a.width)
            with torch.no_grad():
                bound = 1 / d_in if i == 0 else math.sqrt(6 / d_in) / a.omega
                lin.weight.uniform_(-bound, bound, generator=gen)
                lin.bias.uniform_(-1 / math.sqrt(d_in), 1 / math.sqrt(d_in), generator=gen)
            layers.append(Sine(lin, a.omega))
            d_in = a.width
        out = nn.Linear(d_in, 1)
        with torch.no_grad():
            b = math.sqrt(6 / d_in) / a.omega
            out.weight.uniform_(-b, b, generator=gen)
            out.bias.zero_()
        layers.append(out)
        return nn.Sequential(*layers)
    # ReLU MLP, optionally behind Fourier features. PyTorch default (Kaiming-uniform) init, drawn
    # after seeding the global RNG so runs are reproducible.
    layers = []
    if a.arch == "ff":
        layers.append(FF(a.sigma, a.nff, gen))
        d_in = 2 * a.nff
    else:
        d_in = 2
    for _ in range(a.depth):
        layers += [nn.Linear(d_in, a.width), nn.ReLU()]
        d_in = a.width
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)


def coords(n, a, offset=0.0, device="cpu"):
    t = (torch.arange(n, dtype=torch.float32, device=device) + 0.5 + offset) / n  # pixel centres in (0,1)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    x01 = torch.stack([xx, yy], -1).reshape(-1, 2)
    return x01 if a.arch == "ff" else 2 * x01 - 1


def make_opt(a, params):
    if a.opt == "adam":
        return torch.optim.Adam(params, lr=a.lr)
    return torch.optim.SGD(params, lr=a.lr, momentum=a.mom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["relu", "ff", "siren"], required=True)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--sigma", type=float, default=16.0)
    p.add_argument("--nff", type=int, default=256)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mom", type=float, default=0.9)
    p.add_argument("--first", choices=["A", "C", "none"], default="A")
    p.add_argument("--second", default="B")
    p.add_argument("--steps1", type=int, default=3000)
    p.add_argument("--steps2", type=int, default=3000)
    p.add_argument("--relearn", type=int, default=300)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--nsnap", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--outdir", default=os.path.join(HERE, "cache", "runs"))
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.lr is None:
        a.lr = {("adam", "siren"): 1e-4}.get((a.opt, a.arch), 1e-3 if a.opt == "adam" else 1e-2)
    dev = a.device or ("cuda" if pasar_job.job_id() is not None and torch.cuda.is_available() else "cpu")
    pasar_job.apply_memory_limit()
    tag = {"ff": f"ff{a.sigma:g}", "siren": f"siren{a.omega:g}", "relu": "relu"}[a.arch]
    name = a.name or f"{tag}_w{a.width}_{a.opt}{a.lr:g}_{a.first}{a.second}_s{a.seed}_n{a.res}"
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, name + ".npz")
    print("run", name, "device", dev, flush=True)

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = make_model(a, gen).to(dev)
    P = load(a.res)
    n = a.res
    X = coords(n, a, device=dev)
    Xoff = coords(n, a, offset=0.5, device=dev)  # half-pixel diagonal offset: between the samples
    tgt = {k: torch.as_tensor(v, device=dev).reshape(-1, 1) for k, v in P.items()}
    templates = {k: P[k] for k in ["A", "B", "C", "D1", "D2", "D3", "D4"]}
    G = Ghost(templates, n, device=dev)

    total = (a.steps1 if a.first != "none" else 0) + a.steps2 + a.relearn
    done = 0
    t0 = time.time()

    def fit(target, steps, log_every=None, snaps=None, on_snap=None):
        nonlocal done
        opt = make_opt(a, model.parameters())
        losses = np.zeros(steps, np.float32)
        snaps = set(snaps or [])
        for s in range(steps + 1):
            if s in snaps and on_snap:
                on_snap(s)
            if s == steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = ((model(X) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses[s] = loss.item()
            done += 1
            if done % 200 == 0:
                pasar_job.progress(done, total, loss=float(losses[s]))
        return losses

    @torch.no_grad()
    def output(Xg=X):
        return model(Xg).reshape(n, n).float()

    rec = {}
    # ---- phase 1
    if a.first != "none":
        l1 = fit(tgt[a.first], a.steps1)
        f1 = output()
        rec["loss1"] = l1
        rec["f_end1"] = f1.cpu().numpy()
        rec["f_end1_off"] = output(Xoff).cpu().numpy()
        psnr1 = -10 * math.log10(float(((f1 - tgt[a.first].reshape(n, n)) ** 2).mean()))
        print(f"phase1 {a.first} psnr {psnr1:.2f} dB  t={time.time()-t0:.1f}s", flush=True)
        rec["psnr1"] = psnr1

    state1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({k: v.cpu() for k, v in state1.items()}, out.replace(".npz", "_phase1.pt"))
    pnames = [k for k, _ in model.named_parameters()]

    # ---- phase 2
    snaps = sorted(set([0] + list(np.unique(np.round(np.logspace(0, math.log10(a.steps2), a.nsnap)).astype(int)))))
    snap_f, snap_g, snap_gall, snap_pA, snap_pB, snap_dw = [], [], [], [], [], []
    Bimg = tgt[a.second].reshape(n, n)
    Aimg = tgt["A"].reshape(n, n)

    def on_snap(s):
        f = output()
        g, gall = G(f - Bimg)
        snap_f.append(f.cpu().numpy().astype(np.float16))
        snap_g.append(g)
        snap_gall.append(gall)
        snap_pA.append(-10 * math.log10(float(((f - Aimg) ** 2).mean())))
        snap_pB.append(-10 * math.log10(float(((f - Bimg) ** 2).mean())))
        # per-tensor distance travelled from the end of phase 1 (for the matched random-kick control)
        sd = dict(model.named_parameters())
        snap_dw.append([float((sd[k].detach() - state1[k]).norm()) for k in pnames])

    l2 = fit(tgt[a.second], a.steps2, snaps=snaps, on_snap=on_snap)
    fT = output()
    rec.update(
        loss2=l2, snap_steps=np.array(snaps), snap_f=np.stack(snap_f), ghost=np.stack(snap_g),
        ghost_all=np.stack(snap_gall), psnrA=np.array(snap_pA), psnrB=np.array(snap_pB),
        f_final=fT.cpu().numpy(), f_final_off=output(Xoff).cpu().numpy(),
        snap_dw=np.array(snap_dw), param_names=np.array(pnames),
    )
    print(f"phase2 B psnr {snap_pB[-1]:.2f} dB; ghost(A) bands {np.round(snap_g[-1][0], 3)}  t={time.time()-t0:.1f}s", flush=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ---- phase 3: relearn A from the final B network
    if a.relearn > 0:
        rsteps = [s for s in (1, 3, 10, 30, 100, 300) if s <= a.relearn]
        rl_f = []
        l3 = fit(tgt["A"], a.relearn, snaps=rsteps, on_snap=lambda s: rl_f.append(output().cpu().numpy().astype(np.float16)))
        rec["loss3"] = l3
        rec["relearn_steps"] = np.array(rsteps)
        rec["relearn_f"] = np.stack(rl_f)
        model.load_state_dict(state)
    np.savez_compressed(
        out, **rec, template_names=np.array(G.names), band_labels=np.array(G.labels),
        args=json.dumps(vars(a)), wall=time.time() - t0,
    )
    torch.save(state, out.replace(".npz", "_final.pt"))
    pasar_job.progress(total, total)
    print(f"saved {out}  wall {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
