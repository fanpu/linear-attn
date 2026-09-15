"""Stage B (GPU): rerun the edge-of-stability hero net (2/eta = 80) with exact weight-space records.

Training is the main4.npz protocol of art/edge-of-stability/eos_batch.py with M = 1: full-batch GD,
top-3 Hessian eigenpairs by warm-started block subspace iteration refreshed every step (1 iteration),
cold-start audit every --check-every steps. Added records:

  bank_vecs.npy   (NB, 3, P) float32  top-3 eigenvectors every --bank-every steps, each refined by
                  --bank-iters extra subspace iterations (refinement does NOT feed back into the
                  per-step warm start, so the per-step lambda trace keeps the main4 method)
  theta_f32.npy   (T/10, P) float32   theta_t at t = 0, 10, 20, ... (exact anchors for replay)
  thetabar_f16.npy(T/10, P) float16   21-step centred mean of theta at the same t (NaN where the
                  window is incomplete); float16 is a declared precision reduction
  log.npz         per-step lambda, residuals, loss, projections onto the moving u_j(t), sketches
  replay.npz      (--replay) proj_bank (T, 3*NB): <theta_t - theta_0, bank vector> for EVERY step,
                  rebuilt by replaying <= 9 gradient steps from each float32 anchor; the anchor
                  mismatch after 10 replayed steps is logged as the replay error.

Checkpoints (ckpt.pt) every --ckpt-every steps; rerunning the same command resumes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eosnet as N  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--inv", type=float, default=80.0)
p.add_argument("--n", type=int, default=5000)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--k", type=int, default=3)
p.add_argument("--half", type=int, default=10)
p.add_argument("--sketch", type=int, default=1024)
p.add_argument("--check-every", type=int, default=500)
p.add_argument("--check-iters", type=int, default=60)
p.add_argument("--bank-every", type=int, default=250)
p.add_argument("--bank-iters", type=int, default=30)
p.add_argument("--theta-every", type=int, default=10)
p.add_argument("--ckpt-every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--mem", type=float, default=0.10)
p.add_argument("--out-dir", required=True)
p.add_argument("--stop-after", type=int, default=-1, help="testing: exit right after this step")
p.add_argument("--replay", action="store_true")
args = p.parse_args()

torch.cuda.set_per_process_memory_fraction(args.mem)
dev, dt = "cuda", torch.float32
os.makedirs(args.out_dir, exist_ok=True)
O = lambda f: os.path.join(args.out_dir, f)  # noqa: E731
T_, k, S, W, TE = args.steps, args.k, args.sketch, 2 * args.half + 1, args.theta_every
P = N.P
eta = 2.0 / args.inv
NB = (T_ + args.bank_every - 1) // args.bank_every
NA = (T_ + TE - 1) // TE
X, Y, yl = N.load_data(args.n, dt, dev)
sketch = N.make_sketch(S, dt, dev)
th0 = N.init_theta(args.seed).to(dev, dt)[None]  # (1, P)


def mm(name, shape, dtype):
    path = O(name)
    mode = "r+" if os.path.exists(path) else "w+"
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


bank = mm("bank_vecs.npy", (NB, k, P), np.float32)
th_mm = mm("theta_f32.npy", (NA, P), np.float32)
tb_mm = mm("thetabar_f16.npy", (NA, P), np.float16)

L = {n_: np.full(T_, np.nan, np.float32) for n_ in
     ["loss", "acc", "gnorm", "dtheta_u1", "u1_overlap_prev", "wdist"]}
for n_ in ["evals", "resid", "g_u", "xyz", "proj_moving"]:
    L[n_] = np.full((T_, k), np.nan, np.float32)
L["sketch"] = np.zeros((T_, S), np.float32)
L["u1_sketch"] = np.zeros((T_, S), np.float32)
L["bank_t"] = np.full(NB, -1, np.int64)
L["bank_evals"] = np.full((NB, k), np.nan, np.float32)
L["bank_resid"] = np.full((NB, k), np.nan, np.float32)
L["checks_t"] = np.full(T_ // args.check_every + 1, -1, np.int64)
L["checks_v"] = np.full((T_ // args.check_every + 1, k), np.nan, np.float32)
meta = vars(args) | {"P": P, "eta": eta, "dtype": "float32", "thetabar_dtype": "float16",
                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}


def save_log(t_done, final=False):
    np.savez(O("log.npz.tmp.npz"), **L, steps_done=t_done, final=final, eta=eta, inv=args.inv,
             meta=json.dumps(meta))
    os.replace(O("log.npz.tmp.npz"), O("log.npz"))


ck = O("ckpt.pt")
if os.path.exists(ck):
    C = torch.load(ck, map_location=dev, weights_only=False)
    t_start = C["t"]
    theta, prev_theta, V, u = C["theta"], C["prev_theta"], C["V"], C["u"]
    ring_th, ring_u, ring_sum = C["ring_th"], C["ring_u"], C["ring_sum"]
    L = C["L"]
    wall_prev = C["wall"]
    print(f"resumed at t={t_start}", flush=True)
else:
    t_start, wall_prev = 0, 0.0
    theta = th0.clone()
    prev_theta = theta.clone()
    ring_th = torch.zeros(W, 1, P, dtype=dt, device=dev)
    ring_u = torch.zeros(W, 1, k, P, dtype=dt, device=dev)
    ring_sum = torch.zeros(1, P, dtype=torch.float64, device=dev)
    V = N.orth(torch.randn(1, k, P, dtype=dt, device=dev))
    for _ in range(60):  # burn-in (source protocol)
        HV = N.grad_hvps(theta, V, X, Y)[3]
        ev, V, HV = N.rayleigh_ritz(V, HV)
        V = N.orth(HV)
    u = None


def subspace(th, V0, iters):
    Vc = V0
    for _ in range(iters):
        HV = N.grad_hvps(th, Vc, X, Y)[3]
        ev, Vr, HV = N.rayleigh_ritz(Vc, HV)
        Vc = N.orth(HV)
    res = (HV - ev[..., None] * Vr).norm(dim=-1)
    return ev, Vr, res


t0 = time.time()
done_training = t_start >= T_
for t in range(t_start, T_):
    lv, out, gr, HV = N.grad_hvps(theta, V, X, Y)
    ev, Vr, HVr = N.rayleigh_ritz(V, HV)
    res = (HVr - ev[..., None] * Vr).norm(dim=-1)
    unew = Vr.clone()
    if u is not None:
        sgn = torch.sign(torch.einsum("mjp,mjp->mj", unew, u))
        sgn[sgn == 0] = 1
        unew = unew * sgn[..., None]
        L["u1_overlap_prev"][t] = torch.einsum("mp,mp->m", unew[:, 0], u[:, 0]).abs().item()
    u = unew
    V = N.orth(HVr)
    L["evals"][t] = ev[0].cpu().numpy()
    L["resid"][t] = res[0].cpu().numpy()
    if not torch.isfinite(lv).all():
        print(f"diverged at {t}", flush=True)
        save_log(t, final=False)
        sys.exit(2)
    L["loss"][t] = lv.item()
    L["acc"][t] = (out.argmax(-1) == yl).to(dt).mean().item()
    L["gnorm"][t] = gr.norm().item()
    L["g_u"][t] = torch.einsum("mp,mjp->mj", gr, u)[0].cpu().numpy()
    disp = theta - th0
    L["wdist"][t] = disp.norm().item()
    L["proj_moving"][t] = torch.einsum("mp,mjp->mj", disp, u)[0].cpu().numpy()
    if t > 0:
        L["dtheta_u1"][t] = torch.einsum("mp,mp->m", theta - prev_theta, u[:, 0]).item()
    L["sketch"][t] = sketch(disp)[0].cpu().numpy()
    L["u1_sketch"][t] = sketch(u[:, 0])[0].cpu().numpy()
    if t % TE == 0:
        th_mm[t // TE] = theta[0].cpu().numpy()
    # centred moving average (float64 running sum, as in the source)
    if t >= W:
        ring_sum -= ring_th[t % W].double()
    ring_sum += theta.double()
    ring_th[t % W] = theta
    ring_u[t % W] = u
    if t >= W - 1:
        c = t - args.half
        mean = (ring_sum / W).to(dt)
        d = ring_th[c % W] - mean
        L["xyz"][c] = torch.einsum("mp,mjp->mj", d, ring_u[c % W])[0].cpu().numpy()
        if c % TE == 0:
            tb_mm[c // TE] = mean[0].cpu().numpy().astype(np.float16)
    if t % args.check_every == 0:
        i = t // args.check_every
        Vc = N.orth(torch.randn(1, k, P, dtype=dt, device=dev))
        evc, _, _ = subspace(theta, Vc, args.check_iters)
        L["checks_t"][i], L["checks_v"][i] = t, evc[0].cpu().numpy()
    if t % args.bank_every == 0:
        i = t // args.bank_every
        evb, Vb, resb = subspace(theta, Vr, args.bank_iters)
        sg = torch.sign(torch.einsum("mjp,mjp->mj", Vb, u))
        sg[sg == 0] = 1
        Vb = Vb * sg[..., None]
        bank[i] = Vb[0].cpu().numpy()
        L["bank_t"][i], L["bank_evals"][i], L["bank_resid"][i] = t, evb[0].cpu().numpy(), resb[0].cpu().numpy()
    prev_theta = theta
    theta = theta - eta * gr
    if t % 100 == 0:
        el = time.time() - t0
        print(f"t={t} {wall_prev + el:.0f}s lam1={ev[0, 0].item():.2f} lam1*eta/2={ev[0, 0].item() * eta / 2:.4f} "
              f"loss={lv.item():.4f} res1={res[0, 0].item() / ev[0, 0].item():.3f}", flush=True)
    last = t == T_ - 1
    if (t + 1) % args.ckpt_every == 0 or last:
        for a in (bank, th_mm, tb_mm):
            a.flush()
        wall = wall_prev + time.time() - t0
        meta["wall_train_s"] = wall
        torch.save({"t": t + 1, "theta": theta, "prev_theta": prev_theta, "V": V, "u": u,
                    "ring_th": ring_th, "ring_u": ring_u, "ring_sum": ring_sum, "L": L, "wall": wall},
                   ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        save_log(t + 1, final=last)
        print(f"checkpoint t={t + 1} wall={wall:.0f}s", flush=True)
    if args.stop_after >= 0 and t >= args.stop_after:
        print(f"stop-after {t}", flush=True)
        sys.exit(0)
del ring_th, ring_u
torch.cuda.empty_cache()

if args.replay:
    # ---- Pass 2: per-step projections onto the whole fixed bank, replayed from float32 anchors ----
    D = torch.tensor(np.asarray(bank).reshape(NB * k, P), device=dev)  # (NB*k, P)
    rp = O("replay_partial.npz")
    proj = np.full((T_, NB * k), np.nan, np.float32)
    err = np.full(NA, np.nan, np.float32)
    a0 = 0
    if os.path.exists(rp):
        R = np.load(rp)
        proj, err, a0 = R["proj_bank"], R["anchor_err"], int(R["a_next"])
        print(f"replay resumed at anchor {a0}", flush=True)
    tr = time.time()
    for a in range(a0, NA):
        th = torch.tensor(np.asarray(th_mm[a]), device=dev)[None]
        for j in range(TE):
            t = a * TE + j
            if t >= T_:
                break
            proj[t] = (D @ (th - th0)[0]).cpu().numpy()
            gr = N.grad_hvps(th, None, X, Y)[2]
            th = th - eta * gr
        if a + 1 < NA:
            nxt = torch.tensor(np.asarray(th_mm[a + 1]), device=dev)[None]
            err[a] = ((th - nxt).norm() / (nxt - th0).norm()).item()
        if (a + 1) % 50 == 0 or a == NA - 1:
            np.savez(rp + ".tmp.npz", proj_bank=proj, anchor_err=err, a_next=a + 1)
            os.replace(rp + ".tmp.npz", rp)
            print(f"replay anchor {a + 1}/{NA} {time.time() - tr:.0f}s max_err={np.nanmax(err):.2e}", flush=True)
    np.savez(O("replay.npz"), proj_bank=proj, anchor_err=err, bank_t=L["bank_t"], k=k,
             wall_replay_s=time.time() - tr)
    print("replay done", flush=True)
print("stage_b done", flush=True)
