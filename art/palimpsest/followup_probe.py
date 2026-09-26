"""Follow-up (2026-09-26 evening), CPU: is page A still inside the network while its output is blank?

Re-runs the first 150 steps of phase 2 (on page B) from the saved phase-1 state of adam_restart.py
(fresh Adam, or Adam with carried-over state), on the CPU. At chosen steps it fits a ridge readout from the
last hidden layer's activations (65,536 pixels x 256 units, + bias) to page A, to a decoy page D1 (same hand,
shuffled words) and to page B, and reports the fraction of each page's variance the layer can still express
(R^2), plus the A-specific part R^2(A) - R^2(D1). A blank output with high A-specific R^2 means the letters
are still in the representation and only the readout has cancelled them.
Writes cache/followup/probe_<first>B_s<seed>_<variant>.json
"""
import argparse, argparse as _ap, copy, json, os
import numpy as np
import torch
from train import make_model, coords
from pages import load

HERE = os.path.dirname(os.path.abspath(__file__))


def r2(H, y, lam=1e-3, fit=False):
    Hc = torch.cat([H, torch.ones(H.shape[0], 1, dtype=H.dtype)], 1)
    A = Hc.T @ Hc
    A += lam * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=H.dtype)
    w = torch.linalg.solve(A, Hc.T @ y)
    yh = Hc @ w
    res = y - yh
    v = float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return (v, yh) if fit else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", default="A"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="fresh"); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe", default="0,1,2,3,5,10,20,40,60,80,100,120,150")
    a = p.parse_args()
    torch.set_num_threads(6)
    arch = _ap.Namespace(arch="ff", width=256, depth=4, sigma=32.0, nff=256, omega=30.0)
    model = make_model(arch, torch.Generator().manual_seed(a.seed))
    ck = torch.load(os.path.join(HERE, "cache", "followup", f"{a.first}_s{a.seed}_phase1.pt"), map_location="cpu")
    model.load_state_dict(ck["model"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.variant == "carry":
        opt.load_state_dict(copy.deepcopy(ck["opt"]))
    P = load(256); X = coords(256, arch)
    T = {k: torch.tensor(P[k], dtype=torch.float64).reshape(-1, 1) for k in ["A", "B", "D1", "C"]}
    B = T["B"].float()
    body = model[:-1]
    probe = [int(s) for s in a.probe.split(",")]
    out = []; recon = {}; outs = {}
    for s in range(max(probe) + 1):
        if s in probe:
            with torch.no_grad():
                H = body(X).double(); f = model[-1](H.float()).double()
            row = dict(step=s, fstd=float(f.std()), dead=float(((H > 0).sum(0) == 0).double().mean()),
                       **{f"r2_{k}": r2(H, T[k]) for k in T})
            recon[s] = r2(H, T["A"], fit=True)[1].reshape(256, 256).float().numpy().astype(np.float16)
            outs[s] = f.reshape(256, 256).float().numpy().astype(np.float16)
            row["r2_A_specific"] = row["r2_A"] - row["r2_D1"]
            # how much of the output's own variance is A: corr of output with A
            fa = f - f.mean(); ta = T["A"] - T["A"].mean()
            row["corr_out_A"] = float((fa * ta).sum() / (fa.norm() * ta.norm() + 1e-12))
            out.append(row); print(row, flush=True)
        opt.zero_grad(set_to_none=True)
        loss = ((model(X) - B) ** 2).mean(); loss.backward(); opt.step()
    np.savez_compressed(os.path.join(HERE, "cache", "followup", f"probe_{a.first}B_s{a.seed}_{a.variant}.npz"),
                        steps=np.array(sorted(recon)), recon_A=np.stack([recon[k] for k in sorted(recon)]),
                        out=np.stack([outs[k] for k in sorted(outs)]))
    json.dump(out, open(os.path.join(HERE, "cache", "followup", f"probe_{a.first}B_s{a.seed}_{a.variant}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
