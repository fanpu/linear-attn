"""Train one white-box model and record its learned key metric. Usage:
    python train.py --model delta|linattn --T 128 --qd 0 --init I|sqrt --seed 0
Writes cache/<tag>.json (final metrics) and cache/<tag>.pt (checkpoint, for pause/resume)."""
import argparse, json, math, sys, time
from pathlib import Path
import torch
from models import DeltaNetWB, LinAttnWB, make_batch, spectrum, fit_exponent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pause import PAUSE_EXIT, should_pause

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--T", type=int, required=True)
p.add_argument("--qd", type=float, default=0.0); p.add_argument("--init", default="I")
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=4000)
p.add_argument("--d", type=int, default=32); p.add_argument("--kappa", type=float, default=100)
p.add_argument("--sigma2", type=float, default=0.1); p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--batch", type=int, default=256)
args = p.parse_args()
tag = f"{args.model}_T{args.T}_qd{args.qd}_{args.init}_s{args.seed}"
out_json, ckpt = Path("cache") / f"{tag}.json", Path("cache") / f"{tag}.pt"
if out_json.exists():
    print("done already:", tag); sys.exit(0)

dev, d = "cuda", args.d
torch.manual_seed(args.seed)
lam = spectrum(d, args.kappa).to(dev)
q = args.qd / d
A0 = torch.eye(d, device=dev) if args.init == "I" else torch.diag(lam ** -0.5)   # A^T A = Sigma^-s0 with s0 = 0 or 1
if args.model == "delta":
    model = DeltaNetWB(d, A0, learn_alpha=q > 0, alpha0=0.99).to(dev)
else:
    model = LinAttnWB(d, A0).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
start, hist = 0, []
if ckpt.exists():
    st = torch.load(ckpt)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    start, hist = st["step"], st["hist"]
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state(st["rng_cuda"])
    print(f"resumed {tag} at step {start}")

t0 = time.time()
for step in range(start, args.steps):
    x, y, _ = make_batch(args.batch, args.T, d, lam, q, args.sigma2, dev)
    loss = (model(x, y)[:, 1:] - y[:, 1:]).pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 250 == 0 or step == args.steps - 1:
        s_hat, r2, off = fit_exponent(model.metric(), lam)
        hist.append(dict(step=step, loss=loss.item(), s=s_hat))
        print(f"{tag} step {step} loss {loss.item():.4f} s_hat {s_hat:.3f} r2 {r2:.3f} off {off:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if should_pause() or (step + 1) % 500 == 0:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1,
                        hist=hist, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), ckpt)
        if should_pause():
            sys.exit(PAUSE_EXIT)

# evaluation on fresh sequences: context-averaged excess risk (subtract the noise floor)
with torch.no_grad():
    g = torch.Generator(device=dev).manual_seed(12345)
    errs = []
    for _ in range(16):
        x, y, W = make_batch(256, args.T, d, lam, q, args.sigma2, dev, gen=g)
        yh = model(x, y)
        clean = (W * x).sum(-1)
        errs.append((yh[:, 1:] - clean[:, 1:]).pow(2).mean().item())
s_hat, r2, off = fit_exponent(model.metric(), lam)
res = dict(tag=tag, **vars(args), s_hat=s_hat, r2=r2, offdiag=off, excess_risk=sum(errs) / len(errs), hist=hist,
           diagM=torch.diagonal(model.metric()).tolist(), lam=lam.tolist())
if args.model == "delta":
    res.update(beta=model.beta.item(), alpha=model.alpha.item(), alpha_star=math.sqrt(1 - q))
json.dump(res, open(out_json, "w"))
ckpt.unlink(missing_ok=True)
print("FINAL", json.dumps({k: res[k] for k in ["tag", "s_hat", "r2", "offdiag", "excess_risk"] + (["alpha", "beta"] if args.model == "delta" else [])}))
