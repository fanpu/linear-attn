"""Width sweep for the lazy regime: NTK-parameterized 2-layer ReLU nets on binary CIFAR (airplane vs automobile).

For each width m and seed: full-batch GD (float64), log-spaced checkpoints of train loss, outputs on all
train+test points, relative weight change; empirical NTK (torch.func, float32) at init on (train+test, train)
and at the end on (train, train); a 96-point kernel sub-block at every checkpoint (for animations).

    _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/compute_ntk.py widths
    _shared/gpu_run.sh .venv/bin/python 04-lazy-rich-mup/compute_ntk.py alpha
"""
import math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntk_core import cifar_binary, empirical_ntk, init_params, net

torch.cuda.set_per_process_memory_fraction(0.08)
dev = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/cache"
os.makedirs(CACHE, exist_ok=True)

xtr, ytr, xte, yte = cifar_binary()
n = len(ytr)
Xtr = torch.tensor(xtr, device=dev, dtype=torch.float32); Xall = torch.tensor(np.concatenate([xtr, xte]), device=dev, dtype=torch.float32)
Y = torch.tensor(ytr, device=dev, dtype=torch.float32)
SUB = np.concatenate([np.where(ytr == -1)[0][:48], np.where(ytr == 1)[0][:48]])  # 96 pts sorted by class


def run(m, seed, alpha=None, lr=2.0, steps=3000, n_ck=40, tag=""):
    """alpha=None: plain NTK parameterization. alpha>0: Chizat-Oyallon-Bach model alpha*(f - f0), loss / alpha^2."""
    out_path = f"{CACHE}/{tag}.npz"
    if os.path.exists(out_path):
        print("skip", out_path); return
    t0 = time.time()
    p = init_params(192, m, seed, dev, dtype=torch.float32)  # float32 is 10x faster here; f64 spot-check in test
    p0 = {k: v.clone() for k, v in p.items()}
    f0_all = net(p0, Xall).detach()
    scale = 1.0 if alpha is None else alpha

    def model_all(q):
        f = net(q, Xall)
        return f if alpha is None else alpha * (f - f0_all)

    cks = np.unique(np.round(np.geomspace(1, steps, n_ck)).astype(int))
    cks = np.concatenate([[0], cks])
    rec = dict(step=[], loss=[], f_all=[], dW=[], da=[], Ksub=[])
    p32 = lambda q: {k: v.float() for k, v in q.items()}
    Xs32 = Xtr[SUB].float()
    for t in range(steps + 1):
        if t in cks:
            with torch.no_grad():
                f = model_all(p)
                rec["step"].append(t); rec["f_all"].append(f.cpu().numpy())
                rec["loss"].append(0.5 * ((f[:n] - Y) ** 2).mean().item())
                rec["dW"].append(((p["W"] - p0["W"]).norm() / p0["W"].norm()).item())
                rec["da"].append(((p["a"] - p0["a"]).norm() / p0["a"].norm()).item())
            rec["Ksub"].append((scale ** 2 * empirical_ntk(p32(p), Xs32)).cpu().numpy())
            if not np.isfinite(rec["loss"][-1]):
                print("diverged", tag); break
        if t == steps:
            break
        p["W"].requires_grad_(True); p["a"].requires_grad_(True)
        f = model_all(p)[:n]
        loss = 0.5 * ((f - Y) ** 2).mean() / scale ** 2
        gW, ga = torch.autograd.grad(loss, (p["W"], p["a"]))
        with torch.no_grad():
            p = {"W": p["W"] - lr * gW, "a": p["a"] - lr * ga}
    K0 = (scale ** 2 * empirical_ntk(p32(p0), Xall.float(), Xtr.float())).cpu().numpy()
    KT = (scale ** 2 * empirical_ntk(p32(p), Xtr.float())).cpu().numpy()
    np.savez_compressed(out_path, m=m, seed=seed, alpha=-1 if alpha is None else alpha, lr=lr,
                        **{k: np.array(v) for k, v in rec.items()}, K0=K0.astype(np.float32), KT=KT.astype(np.float32),
                        sub=SUB)
    print(f"{tag}: m={m} seed={seed} alpha={alpha} loss={rec['loss'][-1]:.4f} dW={rec['dW'][-1]:.2e} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "smoke":
        run(64, 0, steps=20, n_ck=5, tag="smoke_w"); run(256, 0, alpha=0.1, steps=20, n_ck=5, tag="smoke_a")
    elif what == "alpha_probe":
        for alpha in [0.01, 0.03, 0.1, 0.3]:
            for lr in [0.5, 2.0]:
                run(1024, 0, alpha=alpha, lr=lr, steps=int(sys.argv[2]), n_ck=6, tag=f"probe_a{alpha}_lr{lr}")
    elif what == "alpha":
        for seed in range(2):
            for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
                run(1024, seed, alpha=alpha, lr=LR_ALPHA, steps=STEPS_ALPHA, n_ck=60, tag=f"alpha_a{alpha}_s{seed}")
    elif what == "widths":
        for seed in range(3):
            for m in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
                run(m, seed, tag=f"width_m{m}_s{seed}")
