"""Combed M1 compute: train one MLP per N, sample closed-form and trained fields, measure memorisation.

  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/compute.py train   [--threads 4]
  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/compute.py sample closed|mlp [--threads 4]
  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/compute.py converge closed|mlp
  OMP_NUM_THREADS=4 art/.venv/bin/python art/combed/compute.py limit closed|mlp
  art/.venv/bin/python art/combed/compute.py summary

Every stage checkpoints one file per N in cache/ and skips files that exist, so a rerun resumes.
Device: CPU (declared; see NOTES.md).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import combed_common as C

TRAIN = dict(steps=20_000, batch=1024, lr=1e-3, schedule="cosine to 0", optimizer="Adam(0.9,0.999)", seed=0,
             t_dist="U[0,1]", dtype="float32", width=256, depth=4, time_freqs=16)


def excess_loss_eval(n: int, data: torch.Tensor):
    """Fixed probe set for E||v_theta - v*||^2 (the reducible part of the CFM loss), t ~ U[0, 0.99]."""
    g = torch.Generator().manual_seed(99)
    x1 = data[torch.randint(len(data), (4096,), generator=g)]
    x0 = torch.randn(4096, 3, generator=g, dtype=torch.float64)
    t = torch.rand(4096, generator=g, dtype=torch.float64) * 0.99
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    vstar = torch.stack([C.closed_form_velocity(xt[i:i + 1], float(t[i]), data)[0] for i in range(4096)])
    return xt, t, vstar


def train(n: int):
    out = C.CACHE / f"mlp_N{n}.pt"
    if out.exists():
        print(f"[train] N={n} exists, skip", flush=True)
        return
    data64 = torch.tensor(C.training_set(n), dtype=torch.float64)
    data = data64.float()
    xt_p, t_p, vstar_p = excess_loss_eval(n, data64)
    torch.manual_seed(TRAIN["seed"])
    model = C.VelocityMLP(TRAIN["width"], TRAIN["depth"], TRAIN["time_freqs"])
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN["steps"])
    hist = []
    t0 = time.time()
    for step in range(1, TRAIN["steps"] + 1):
        x1 = data[torch.randint(n, (TRAIN["batch"],))]
        x0 = torch.randn(TRAIN["batch"], 3)
        t = torch.rand(TRAIN["batch"])
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                ex = ((model(xt_p.float(), t_p.float()).double() - vstar_p) ** 2).sum(1).mean().item()
            hist.append((step, loss.item(), ex))
            print(f"[train] N={n} step {step} cfm {loss.item():.4f} excess {ex:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(state=model.state_dict(), hist=hist, cfg=TRAIN, wall_s=time.time() - t0), out)


def load_mlp(n: int) -> C.VelocityMLP:
    ck = torch.load(C.CACHE / f"mlp_N{n}.pt", weights_only=False)
    cfg = ck["cfg"]
    m = C.VelocityMLP(cfg["width"], cfg["depth"], cfg["time_freqs"])
    m.load_state_dict(ck["state"])
    return m.double().eval()


def sample(kind: str, n: int):
    out = C.CACHE / f"flow_{kind}_N{n}.npz"
    if out.exists():
        print(f"[sample] {kind} N={n} exists, skip", flush=True)
        return
    data_np = C.training_set(n)
    data = torch.tensor(data_np, dtype=torch.float64)
    if kind == "closed":
        field = lambda x, t: C.closed_form_velocity(x, t, data)
    else:
        model = load_mlp(n)
        field = lambda x, t: C.mlp_velocity(model, x, t)
    x0 = torch.tensor(C.noise_seeds(), dtype=torch.float64)
    keep = C.keep_indices(256)
    res = {}
    t0 = time.time()
    for steps in (256, 512):
        xe, traj = C.rk4(field, x0, steps, keep=keep if steps == 256 else None)
        ve = field(xe, C.T_STOP)
        xhat = xe + (1 - C.T_STOP) * ve  # one Euler step to t = 1; for v* this is E[x1 | x_t]
        res[f"end{steps}"] = xe.numpy()
        res[f"xhat{steps}"] = xhat.numpy()
        if traj is not None:
            res["traj"] = traj.numpy().astype(np.float32)
        print(f"[sample] {kind} N={n} {steps} steps done ({time.time() - t0:.0f}s)", flush=True)
    for key in ("end256", "xhat256"):
        i, d1, d2 = C.nn_two(res[key], data_np)
        res[f"{key}_nn"], res[f"{key}_d1"], res[f"{key}_d2"] = i, d1, d2
    if kind == "closed":
        w = torch.softmax(C.closed_form_logits(torch.tensor(res["end256"]), C.T_STOP, data), 1)
        res["wmax256"] = w.max(1).values.numpy()
    np.savez(out, t_keep=C.time_grid(256)[keep], keep=keep, data=data_np, x0=x0.numpy(),
             wall_s=time.time() - t0, **res)


def converge(kind: str, n: int):
    """Order check on the 256 seeds with the largest 256-vs-512 endpoint gap plus 256 random seeds:
    integrate with 1024 and 2048 steps; 4th-order RK4 should shrink the gap ~16x per doubling."""
    out = C.CACHE / f"converge_{kind}_N{n}.json"
    if out.exists():
        return
    z = np.load(C.CACHE / f"flow_{kind}_N{n}.npz")
    data = torch.tensor(z["data"], dtype=torch.float64)
    if kind == "closed":
        field = lambda x, t: C.closed_form_velocity(x, t, data)
    else:
        model = load_mlp(n)
        field = lambda x, t: C.mlp_velocity(model, x, t)
    gap = np.linalg.norm(z["end256"] - z["end512"], axis=1)
    worst = np.argsort(gap)[-256:]
    rand = np.random.default_rng(5).choice(len(gap), 256, replace=False)
    res = {}
    for name, idx in (("worst", worst), ("random", rand)):
        x0 = torch.tensor(z["x0"][idx])
        e = {256: z["end256"][idx], 512: z["end512"][idx]}
        for steps in (1024, 2048):
            e[steps] = C.rk4(field, x0, steps)[0].numpy()
        res[name] = {f"{a}v{b}": float(np.linalg.norm(e[a] - e[b], axis=1).max())
                     for a, b in ((256, 512), (512, 1024), (1024, 2048))}
    out.write_text(json.dumps(res))
    print(f"[converge] {kind} N={n} {res}", flush=True)


def limit(kind: str, n: int, n_tail: int = 96):
    """Continue the 256-step endpoints from t = 1-1e-3 to t = 1-1e-6 on a geometric grid in (1 - t)
    (h/(1-t) = 0.07 per step), to separate the field's t -> 1 behaviour from the declared stop.
    The same run with 2 n_tail steps gives its own step-doubling gap."""
    out = C.CACHE / f"limit_{kind}_N{n}.npz"
    if out.exists():
        return
    z = np.load(C.CACHE / f"flow_{kind}_N{n}.npz")
    data_np = z["data"]
    data = torch.tensor(data_np, dtype=torch.float64)
    if kind == "closed":
        field = lambda x, t: C.closed_form_velocity(x, t, data)
    else:
        model = load_mlp(n)
        field = lambda x, t: C.mlp_velocity(model, x, t)
    t0 = time.time()
    res = {}
    for k in (n_tail, 2 * n_tail):
        ts = C.tail_grid(k)
        xe = C.rk4_grid(field, torch.tensor(z["end256"]), ts)
        res[f"end_tail{k}"] = xe.numpy()
        res[f"xhat_tail{k}"] = (xe + (1 - ts[-1]) * field(xe, float(ts[-1]))).numpy()
    for key in (f"end_tail{n_tail}", f"xhat_tail{n_tail}"):
        i, d1, d2 = C.nn_two(res[key], data_np)
        res[f"{key}_nn"], res[f"{key}_d1"], res[f"{key}_d2"] = i, d1, d2
    np.savez(out, t_end=C.tail_grid(n_tail)[-1], wall_s=time.time() - t0, **res)
    print(f"[limit] {kind} N={n} done ({time.time() - t0:.0f}s)", flush=True)


def summary():
    from scipy.spatial import cKDTree

    knot = cKDTree(C.trefoil(np.linspace(0, 2 * math.pi, 400_000, endpoint=False)))  # spacing 2.4e-5
    rows = []
    for n in C.NS:
        for kind in ("closed", "mlp"):
            f = C.CACHE / f"flow_{kind}_N{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            r = dict(N=n, field=kind)
            for key in ("end256", "xhat256"):
                ratio = z[f"{key}_d1"] / z[f"{key}_d2"]
                r[f"mem_{key}"] = float((ratio < C.MEM_RATIO).mean())
                r[f"median_ratio_{key}"] = float(np.median(ratio))
                r[f"median_d1_{key}"] = float(np.median(z[f"{key}_d1"]))
                r[f"distinct_nn_{key}"] = int(len(np.unique(z[f"{key}_nn"][ratio < C.MEM_RATIO])))
            dk = knot.query(z["xhat256"])[0]
            r.update(knot_dist_median=float(np.median(dk)), knot_dist_p90=float(np.quantile(dk, 0.9)))
            dd = np.linalg.norm(z["end256"] - z["end512"], axis=1)
            dx = np.linalg.norm(z["xhat256"] - z["xhat512"], axis=1)
            r.update(step2_max_end=float(dd.max()), step2_p999_end=float(np.quantile(dd, 0.999)),
                     step2_median_end=float(np.median(dd)), step2_max_xhat=float(dx.max()),
                     step2_frac_gt_1em3_end=float((dd > 1e-3).mean()),
                     wall_s=float(z["wall_s"]))
            if kind == "closed":
                r["frac_wmax_gt_0.99"] = float((z["wmax256"] > 0.99).mean())
            m = C.CACHE / f"mlp_N{n}.pt"
            if kind == "mlp" and m.exists():
                ck = torch.load(m, weights_only=False)
                r["train_wall_s"] = ck["wall_s"]
                r["final_cfm"], r["final_excess"] = ck["hist"][-1][1], ck["hist"][-1][2]
            lf = C.CACHE / f"limit_{kind}_N{n}.npz"
            if lf.exists():
                L = np.load(lf)
                for key in ("end_tail96", "xhat_tail96"):
                    r[f"mem_{key}"] = float((L[f"{key}_d1"] < C.MEM_RATIO * L[f"{key}_d2"]).mean())
                r["tail_step2_max_end"] = float(np.linalg.norm(L["end_tail96"] - L["end_tail192"], axis=1).max())
                r["tail_step2_max_xhat"] = float(np.linalg.norm(L["xhat_tail96"] - L["xhat_tail192"], axis=1).max())
            cj = C.CACHE / f"converge_{kind}_N{n}.json"
            if cj.exists():
                r["converge"] = json.loads(cj.read_text())
            rows.append(r)
    # null: fresh points on the knot (perfect generalisation, s ~ U[0, 2pi), seed 7) through the same criterion
    fresh = C.trefoil(np.random.default_rng(7).uniform(0, 2 * math.pi, C.N_SAMPLES))
    for n in C.NS:
        _, d1, d2 = C.nn_two(fresh, C.training_set(n))
        rows.append(dict(N=n, field="null_fresh_knot", mem_xhat256=float((d1 < C.MEM_RATIO * d2).mean()),
                         median_ratio_xhat256=float(np.median(d1 / d2))))
    (C.CACHE / "summary.json").write_text(json.dumps(rows, indent=1))
    (C.ROOT / "logs" / "summary.json").write_text(json.dumps(rows, indent=1))
    hdr = "knotd = median distance of x_hat to the continuous trefoil\nN field knotd mem(xhat) mem(raw) mem(xhat,1-1e-6) med_d1/d2(xhat) distinctNN(xhat) step2_max step2_p999 wmax>.99"
    print(hdr)
    for r in rows:
        if r["field"].startswith("null"):
            print(f"{r['N']:5d} null(fresh knot points) mem {r['mem_xhat256']:.4f} median d1/d2 {r['median_ratio_xhat256']:.3g}")
            continue
        print(f"{r['N']:5d} {r['field']:6s} {r['knot_dist_median']:.1e} {r['mem_xhat256']:.4f} {r['mem_end256']:.4f} {r.get('mem_xhat_tail96', float('nan')):.4f} "
              f"{r['median_ratio_xhat256']:.3g} {r['distinct_nn_xhat256']:5d} "
              f"{r['step2_max_end']:.2e} {r['step2_p999_end']:.2e} {r.get('frac_wmax_gt_0.99', float('nan')):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "sample", "converge", "limit", "summary"])
    ap.add_argument("kind", nargs="?", default="closed", choices=["closed", "mlp"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="*", default=list(C.NS))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    C.CACHE.mkdir(exist_ok=True)
    if a.stage == "train":
        for n in a.ns:
            train(n)
    elif a.stage == "sample":
        for n in a.ns:
            sample(a.kind, n)
    elif a.stage == "limit":
        for n in a.ns:
            limit(a.kind, n)
    elif a.stage == "converge":
        for n in a.ns:
            converge(a.kind, n)
    else:
        summary()
