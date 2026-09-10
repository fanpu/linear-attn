"""
Day 1: does the fla chunked Gated DeltaNet kernel compute the right thing on GB10?

Deliverable: a table of forward and backward agreement between the chunked kernel
and a naive fp32 sequential reference, across dtypes / head dims / sequence lengths.

This script has no dependency on a training loop, a dataset, or a model config.
That is deliberate. Kernel risk and framework risk are separate failure surfaces
and should not be debugged on the same day.

Run:  python day1_kernel_check.py
"""

import inspect
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

DEV = "cuda"


def gdn_recurrent_ref(q, k, v, g, beta, scale):
    """
    Naive fp32 sequential Gated DeltaNet reference.

    Shapes (fla's layout, head-last):
        q, k : [B, T, H, Dk]
        v    : [B, T, H, Dv]
        g    : [B, T, H]        log-space decay, <= 0
        beta : [B, T, H]        write strength

    Recurrence (Yang et al., GDN), with state S : [Dk, Dv] per (b, h):
        S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T
        o_t = S_t^T (scale * q_t)
    i.e. decay first, then read the *decayed* state, then correct.
    The order matters and is the usual place to get this wrong.
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))

    S = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=torch.float32)
    o = torch.zeros(B, T, H, Dv, device=q.device, dtype=torch.float32)

    for t in range(T):
        a_t = g[:, t].exp()[..., None, None]  # [B, H, 1, 1]
        b_t = beta[:, t][..., None]  # [B, H, 1]
        k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]  # [B, H, D*]

        S = S * a_t
        v_old = torch.einsum("bhkd,bhk->bhd", S, k_t)  # read
        S = S + torch.einsum("bhk,bhd->bhkd", k_t, b_t * (v_t - v_old))
        o[:, t] = torch.einsum("bhkd,bhk->bhd", S, q_t * scale)

    return o, S


def make_inputs(B, T, H, Dk, Dv, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, Dk, device=DEV, dtype=dtype)
    k = torch.randn(B, T, H, Dk, device=DEV, dtype=dtype)
    k = F.normalize(k.float(), dim=-1).to(dtype)  # GDN L2-normalizes keys
    v = torch.randn(B, T, H, Dv, device=DEV, dtype=dtype)
    # decay near 1 (log near 0) is the realistic regime; sample log-uniformly
    g = -torch.rand(B, T, H, device=DEV, dtype=torch.float32).exp() * 0.05
    beta = torch.rand(B, T, H, device=DEV, dtype=torch.float32).sigmoid()
    for x in (q, k, v):
        x.requires_grad_(True)
    g.requires_grad_(True)
    beta.requires_grad_(True)
    return q, k, v, g, beta


def report(name, a, b):
    a, b = a.float(), b.float()
    denom = a.abs().max().clamp_min(1e-6)
    max_rel = (a - b).abs().max().item() / denom.item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    flag = "ok  " if (max_rel < 2e-2 and cos > 0.999) else "FAIL"
    print(f"  {flag} {name:12s} max_rel={max_rel:.3e}  cos={1 - cos:.3e} (1-cos)")
    return max_rel, cos


def run_case(B, T, H, Dk, Dv, dtype):
    print(f"\nT={T:5d} Dk={Dk:3d} Dv={Dv:3d} dtype={str(dtype).split('.')[-1]}")
    scale = Dk**-0.5

    # generat random inputs
    q, k, v, g, beta = make_inputs(B, T, H, Dk, Dv, dtype)
    q2, k2, v2, g2, beta2 = (
        x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
    )

    o_fla, S_fla = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=True,
    )
    o_ref, S_ref = gdn_recurrent_ref(q2, k2, v2, g2, beta2, scale)

    report("fwd out", o_ref, o_fla)
    report("fwd state", S_ref, S_fla)

    # backward: same random cotangent through both paths
    torch.manual_seed(1234)
    d_o = torch.randn_like(o_ref)
    o_fla.float().backward(d_o)
    o_ref.backward(d_o)

    for nm, x, y in [
        ("dq", q2.grad, q.grad),
        ("dk", k2.grad, k.grad),
        ("dv", v2.grad, v.grad),
        ("dg", g2.grad, g.grad),
        ("dbeta", beta2.grad, beta.grad),
    ]:
        report(nm, x, y)


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0))
    import fla

    print("fla", getattr(fla, "__version__", "unknown"))
    # Trust your installed source over any remembered signature:
    print(inspect.signature(chunk_gated_delta_rule))

    for dtype in (torch.float32, torch.bfloat16):
        for T in (256, 1024, 4096):
            for Dk, Dv in ((64, 64), (128, 128), (64, 128)):
                run_case(B=2, T=T, H=4, Dk=Dk, Dv=Dv, dtype=dtype)
