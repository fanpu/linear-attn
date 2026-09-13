"""Evaluate trained sequence models (train_seq.py) against classical in-context algorithms.  CPU only.

For every model: excess risk at every context length t, prediction distance to each algorithm, and behaviour under
test-time covariate scaling (x -> c x) and label-noise shift.  Algorithms (all tuned on in-distribution prompts):
  ridge (Bayes-optimal, = RLS), GD-k with step sizes tuned separately for every t (k = 1..4),
  online SGD / delta rule with one tuned step size (plain LMS and normalized LMS = DeltaNet with L2-normalized keys).

  .venv/bin/python 08-icl-linear-attention/analysis_seq.py
"""
import glob, json, math, pathlib, time
import numpy as np
import torch
from seqmodels import ICLModel
from icl_core import ridge_w, gd_w, lms_prefix_w

torch.set_num_threads(4)
HERE = pathlib.Path(__file__).parent
D64 = torch.float64
d, N, KMAX = 10, 40, 4
SCALES = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
NOISES = [0.0, 0.25, 0.5, 1.0]
g = torch.Generator().manual_seed(1234)


def prompts(B, sigma, c=1.0):
    X = torch.randn(B, N, d, generator=g, dtype=D64) * c
    w = torch.randn(B, d, generator=g, dtype=D64)
    f = torch.einsum("bnd,bd->bn", X, w)
    return X, f + sigma * torch.randn(B, N, generator=g, dtype=D64), f


def prefix(fn, X, y):
    """predictions for y_t from pairs < t, for estimator fn(X[:, :t], y[:, :t]) -> w."""
    out = torch.zeros(X.shape[0], N, dtype=D64)
    for t in range(1, N):
        out[:, t] = (fn(X[:, :t], y[:, :t], t) * X[:, t]).sum(-1)
    return out


# ------------------------------------------------------------------ tune classical baselines per training noise
def tune(sigma, Btune=4096):
    X, y, f = prompts(Btune, sigma)
    lrs = {}
    for k in range(1, KMAX + 1):
        lrs[k] = [None]
        for t in range(1, N):
            eta = torch.full((k,), 1.0 / (1 + (d + 1) / t + sigma**2 / t), dtype=D64)
            if k > 1:
                eta = eta * torch.linspace(0.6, 1.4, k, dtype=D64)
            eta.requires_grad_()
            opt = torch.optim.Adam([eta], lr=0.02)
            for it in range(150):
                wv = gd_w(X[:, :t], y[:, :t], list(eta))
                loss = ((wv * X[:, t]).sum(-1) - f[:, t]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            lrs[k].append(eta.detach())
    betas = {}
    for norm in (True, False):
        grid = np.geomspace(0.02, 1.5, 40) if norm else np.geomspace(0.002, 0.3, 40)
        best = None
        for b in grid:
            W = lms_prefix_w(X, y, float(b), normalize=norm)
            r = ((W[:, :N] * X).sum(-1) - f).pow(2)[:, 1:].mean().item()
            if np.isfinite(r) and (best is None or r < best[0]):
                best = (r, float(b))
        betas[norm] = best[1]
    return lrs, betas


def algorithms(X, y, sigma, lrs, betas):
    lam = max(sigma**2, 1e-8)
    out = {"ridge": prefix(lambda A, b, t: ridge_w(A, b, lam), X, y)}
    for k in range(1, KMAX + 1):
        out[f"gd{k}"] = prefix(lambda A, b, t: gd_w(A, b, list(lrs[k][t])), X, y)
    for norm, name in ((True, "nlms"), (False, "lms")):
        W = lms_prefix_w(X, y, betas[norm], normalize=norm)
        out[name] = (W[:, :N] * X).sum(-1)
    return out


@torch.no_grad()
def model_preds(model, X, y):
    preds = []
    for i in range(0, X.shape[0], 1024):
        preds.append(model(X[i:i + 1024].float(), y[i:i + 1024].float()).double())
    return torch.cat(preds)


BASE = HERE / "cache" / "seq_baselines.pt"
if BASE.exists():
    res, EV = torch.load(BASE, weights_only=False)
    res["models"] = {}
    print("loaded cached baselines", flush=True)
else:
    res = {"t": list(range(1, N + 1)), "scales": SCALES, "noises": NOISES, "models": {}, "algs": {}}
    tuned = {}
    for sig in (0.0, 0.5):
        t0 = time.time()
        tuned[sig] = tune(sig)
        res["algs"][str(sig)] = {"betas": {"nlms": tuned[sig][1][True], "lms": tuned[sig][1][False]},
                                 "gd_lrs": {k: [None] + [v.tolist() for v in tuned[sig][0][k][1:]] for k in tuned[sig][0]}}
        print(f"tuned baselines for sigma={sig} in {time.time()-t0:.0f}s  betas={tuned[sig][1]}", flush=True)

    # shared evaluation prompts
    EV = {}
    for sig in (0.0, 0.5):
        X, y, f = prompts(4096, sig)
        EV[("in", sig)] = (X, y, f)
        EV[("algs", sig)] = algorithms(X, y, sig, *tuned[sig])
        res["algs"][str(sig)]["risk"] = {k: ((v - f) ** 2).mean(0).tolist() for k, v in EV[("algs", sig)].items()}
        shift = {}
        for c in SCALES:
            Xc, yc, fc = prompts(1024, sig, c)
            A = algorithms(Xc, yc, sig, *tuned[sig])
            shift[c] = (Xc, yc, fc)
            res["algs"][str(sig)].setdefault("scale", {})[str(c)] = {k: (((v - fc) ** 2).mean(0) / c**2).tolist() for k, v in A.items()}
        EV[("scale", sig)] = shift
        noise = {}
        for s2 in NOISES:
            Xn, yn, fn = prompts(1024, s2)
            A = algorithms(Xn, yn, sig, *tuned[sig])
            noise[s2] = (Xn, yn, fn)
            res["algs"][str(sig)].setdefault("noise", {})[str(s2)] = {k: ((v - fn) ** 2).mean(0).tolist() for k, v in A.items()}
        EV[("noise", sig)] = noise
        print(f"baselines evaluated for sigma={sig}", flush=True)
    torch.save((res, EV), BASE)

OUT = HERE / "cache" / "seq_eval.json"
if OUT.exists():
    res["models"] = json.load(open(OUT))["models"]
for path in sorted(glob.glob(str(HERE / "cache" / "seq_*_seed*.pt"))):
    if pathlib.Path(path).stem in res["models"]:
        continue
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    model = ICLModel(a["kind"], a["d"], a["width"], a["heads"], a["layers"])
    model.load_state_dict(ck["state"]); model.eval(); model.set_impl("par")
    sig = a["sigma"]
    name = pathlib.Path(path).stem
    t0 = time.time()
    X, y, f = EV[("in", sig)]
    p = model_preds(model, X, y)
    entry = {"kind": a["kind"], "L": a["layers"], "sigma": sig, "steps": a["steps"],
             "risk": ((p - f) ** 2).mean(0).tolist(),
             "dist": {k: ((p - v) ** 2).mean(0).tolist() for k, v in EV[("algs", sig)].items()}}
    entry["scale"] = {}
    for c, (Xc, yc, fc) in EV[("scale", sig)].items():
        pc = model_preds(model, Xc, yc)
        entry["scale"][str(c)] = (((pc - fc) ** 2).mean(0) / c**2).tolist()
    entry["noise"] = {}
    for s2, (Xn, yn, fn) in EV[("noise", sig)].items():
        pn = model_preds(model, Xn, yn)
        entry["noise"][str(s2)] = ((pn - fn) ** 2).mean(0).tolist()
    res["models"][name] = entry
    r = entry["risk"]
    json.dump(res, open(OUT, "w"))
    print(f"{name}: risk t=10 {r[9]:.3f} t=20 {r[19]:.3f} t=40 {r[39]:.4f}   ({time.time()-t0:.0f}s)", flush=True)

json.dump(res, open(HERE / "cache" / "seq_eval.json", "w"))
print("saved cache/seq_eval.json")
