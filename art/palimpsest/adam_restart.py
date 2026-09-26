"""Follow-up (2026-09-26 evening): is the Fourier network's blank page, and the return of page A,
an artefact of restarting Adam?

One job = one phase-1 network (first page A or C, one seed). Phase 1 is train.py's protocol
(FF sigma 32, width 256, depth 4, Adam lr 3e-4, 2,000 full-batch steps at 256 px). Then several
phase-2 runs on page B branch from the *identical* phase-1 weights, differing only in the optimiser:

  fresh        Adam, zeroed moments (the original protocol)
  carry        Adam, moment estimates and step count carried over from phase 1
  warmup100    fresh Adam, lr ramped linearly 0 -> lr over the first 100 steps
  lr0.1x       fresh Adam at lr/10 throughout (a warm-up that never ends)
  adamw        fresh AdamW, weight decay 0.01 (torch default)
  eps1e-5/1e-4/1e-3   fresh Adam with a larger epsilon (B's gradients at the start: median 2e-4)

At every step up to --dense, and at log-spaced steps after, we record: the per-band ghost of every
template page (metrics.Ghost), PSNR vs A and B, the output's spatial std (a blank page has ~0),
and for each of the 4 ReLU layers the fraction of pixels on which each unit is active (a unit
with 0 is dead on the whole page). Outputs are stored as images at a subset of steps.

Writes cache/followup/<first>B_s<seed>_<variant>.npz
"""
import argparse, argparse as _ap, math, os, time, json
import numpy as np
import torch
import pasar_job
from train import make_model, coords
from pages import load
from metrics import Ghost

HERE = os.path.dirname(os.path.abspath(__file__))


def make_opt(kind, params, lr, state=None):
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01), None
    eps = 1e-8
    if kind.startswith("eps"):
        eps = float(kind[3:])
    opt = torch.optim.Adam(params, lr=lr * (0.1 if kind == "lr0.1x" else 1.0), eps=eps)
    if kind == "carry":
        import copy
        opt.load_state_dict(copy.deepcopy(state))
    sched = None
    if kind.startswith("warmup"):
        W = int(kind[6:])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / W))
    return opt, sched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", default="A", choices=["A", "C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", default="fresh,carry,warmup100,lr0.1x,adamw,eps1e-5,eps1e-4,eps1e-3")
    p.add_argument("--steps1", type=int, default=2000)
    p.add_argument("--steps2", type=int, default=1000)
    p.add_argument("--dense", type=int, default=300)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--outdir", default=os.path.join(HERE, "cache", "followup"))
    a = p.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() and pasar_job.job_id() is not None else "cpu")
    pasar_job.apply_memory_limit()
    os.makedirs(a.outdir, exist_ok=True)
    arch = _ap.Namespace(arch="ff", width=256, depth=4, sigma=32.0, nff=256, omega=30.0)
    variants = a.variants.split(",")
    n = a.res
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = make_model(arch, gen).to(dev)
    P = load(n)
    X = coords(n, arch, device=dev)
    tgt = {k: torch.as_tensor(v, device=dev).reshape(-1, 1) for k, v in P.items()}
    G = Ghost({k: P[k] for k in ["A", "B", "C", "D1", "D2", "D3", "D4"]}, n, device=dev)
    total = a.steps1 + len(variants) * a.steps2
    done = 0
    t0 = time.time()

    # ---- phase 1 (identical to train.py)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for s in range(a.steps1):
        opt.zero_grad(set_to_none=True)
        loss = ((model(X) - tgt[a.first]) ** 2).mean()
        loss.backward(); opt.step(); done += 1
        if done % 200 == 0:
            pasar_job.progress(done, total, loss=float(loss))
    with torch.no_grad():
        f1 = model(X).reshape(n, n)
    psnr1 = -10 * math.log10(float(((f1 - tgt[a.first].reshape(n, n)) ** 2).mean()))
    print(f"phase1 {a.first} s{a.seed} psnr {psnr1:.2f} dB t={time.time()-t0:.0f}s", flush=True)
    state1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ostate1 = opt.state_dict()
    torch.save(dict(model=state1, opt=ostate1), os.path.join(a.outdir, f"{a.first}_s{a.seed}_phase1.pt"))

    relus = [m for m in model if isinstance(m, torch.nn.ReLU)]
    acts = {}
    hooks = [r.register_forward_hook(lambda mod, i, o, k=k: acts.__setitem__(k, o.detach())) for k, r in enumerate(relus)]

    snaps = sorted(set(range(a.dense + 1)) | set(np.unique(np.round(np.logspace(0, math.log10(a.steps2), 60)).astype(int)).tolist()))
    img_snaps = set([s for s in snaps if s <= 300 and (s <= 20 or s % 3 == 0)] + [s for s in snaps if s > 300])
    Bimg = tgt["B"].reshape(n, n); Aimg = tgt["A"].reshape(n, n)
    for v in variants:
        model.load_state_dict(state1)
        # fresh optimiser state object each time (load_state_dict copies tensors)
        opt, sched = make_opt(v, model.parameters(), a.lr, state=ostate1)
        rec = dict(steps=[], ghost=[], psnrA=[], psnrB=[], fstd=[], act=[], img_steps=[], imgs=[], loss=[], lr=[])
        sset = set(snaps)
        for s in range(a.steps2 + 1):
            if s in sset:
                with torch.no_grad():
                    f = model(X).reshape(n, n)
                g, _ = G(f - Bimg)
                rec["steps"].append(s); rec["ghost"].append(g)
                rec["psnrA"].append(-10 * math.log10(float(((f - Aimg) ** 2).mean())))
                rec["psnrB"].append(-10 * math.log10(float(((f - Bimg) ** 2).mean())))
                rec["fstd"].append(float(f.std()))
                rec["act"].append(np.stack([(acts[k] > 0).float().mean(0).cpu().numpy() for k in range(len(relus))]).astype(np.float16))
                if s in img_snaps:
                    rec["img_steps"].append(s); rec["imgs"].append(f.cpu().numpy().astype(np.float16))
            if s == a.steps2:
                break
            opt.zero_grad(set_to_none=True)
            loss = ((model(X) - tgt["B"]) ** 2).mean()
            loss.backward()
            rec["lr"].append(opt.param_groups[0]["lr"])
            opt.step()
            if sched:
                sched.step()
            rec["loss"].append(float(loss)); done += 1
            if done % 200 == 0:
                pasar_job.progress(done, total, loss=float(loss))
        out = {k: np.array(vv) for k, vv in rec.items()}
        out.update(template_names=np.array(G.names), band_labels=np.array(G.labels), psnr1=psnr1,
                   args=json.dumps(dict(vars(a), variant=v)))
        path = os.path.join(a.outdir, f"{a.first}B_s{a.seed}_{v}.npz")
        np.savez_compressed(path, **out)
        L = out["ghost"][:, 0] - out["ghost"][:, 3:7].mean(1)
        st = out["steps"]; late = st >= 10
        dead = [(out["act"][:, k] == 0).mean(1) for k in range(4)]
        print(f"{v:10s} blank(fstd<0.05) steps={int((out['fstd'][st <= 300] < 0.05).sum())}  "
              f"A-specific peak(s>=10, bands 8-192 mean)={L[late][:, 3:].mean(1).max():.3f} at {st[late][L[late][:, 3:].mean(1).argmax()]}  "
              f"max dead frac per layer={[round(float(d.max()), 3) for d in dead]}  final psnrB={out['psnrB'][-1]:.1f}  t={time.time()-t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    pasar_job.progress(total, total)
    print("done", flush=True)


if __name__ == "__main__":
    main()
