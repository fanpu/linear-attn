"""Cross-check pure-torch recurrences against flash-linear-attention kernels (fwd + bwd), and check the
DeltaNet <-> online-SGD identity.  Run: .venv/bin/python 08-icl-linear-attention/test_seqmodels.py"""
import os; os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # Triton tl.dot defaults to TF32 otherwise
import torch
import torch.nn.functional as F
from seqmodels import linear_attn_ref, linear_attn_par, delta_ref, delta_fast
from icl_core import lms_prefix_w

torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def check(name, a, b, tol):
    r = rel(a, b)
    print(f"[{'ok' if r < tol else 'FAIL'}] {name}: rel err {r:.2e}")
    assert r < tol


B, T, H, Dk, Dv = 3, 50, 2, 16, 16
q, k, v = (torch.randn(B, T, H, D, device=dev) for D in (Dk, Dk, Dv))
check("linear attn: causal-mask parallel == recurrence (float64)",
      linear_attn_par(q.double(), k.double(), v.double()), linear_attn_ref(q.double(), k.double(), v.double()), 1e-12)

from fla.ops.linear_attn import fused_recurrent_linear_attn
o_fla, _ = fused_recurrent_linear_attn(q, k, v, scale=1.0, normalize=False)
o_ref = linear_attn_ref(q, k, v) * torch.arange(1, T + 1, device=dev)[:, None, None]   # undo our 1/t readout
check("linear attn: pure torch == fla fused_recurrent_linear_attn (fp32)", o_fla, o_ref, 1e-4)

for gated in (False, True):
    qq, kk = F.normalize(q, dim=-1).requires_grad_(), F.normalize(k, dim=-1).requires_grad_()
    vv = v.clone().requires_grad_()
    beta = torch.rand(B, T, H, device=dev).requires_grad_()
    la = (-torch.rand(B, T, H, device=dev) * 0.3).requires_grad_() if gated else None
    ins = [qq, kk, vv, beta] + ([la] if gated else [])
    o1 = delta_fast(qq, kk, vv, beta, la)
    g1 = torch.autograd.grad((o1 * torch.cos(o1)).sum(), ins)
    o2 = delta_ref(qq, kk, vv, beta, la)
    g2 = torch.autograd.grad((o2 * torch.cos(o2)).sum(), ins)
    nm = "gated delta" if gated else "delta"
    check(f"{nm}: fla chunk kernel == pure torch, forward (fp32)", o1, o2, 1e-3)
    for n_, a_, b_ in zip(["dq", "dk", "dv", "dbeta", "dlogalpha"], g1, g2):
        check(f"{nm}: backward {n_}", a_, b_, 3e-3)

# DeltaNet with k = x/|x|, v = y/|x|, q = x_q, one head, is normalized LMS on in-context regression.
d, n = 5, 30
X = torch.randn(B, n, d, dtype=torch.float64)
w = torch.randn(B, d, dtype=torch.float64)
y = torch.einsum("bnd,bd->bn", X, w)
nrm = X.norm(dim=-1, keepdim=True)
kx, vx = (X / nrm)[:, :, None], (y[..., None] / nrm)[:, :, None]
# append d probe tokens with beta = 0 and q = e_j, so o = S_n e_j = w_n[j]
eye = torch.eye(d, dtype=torch.float64).expand(B, d, d)[:, :, None]
K_ = torch.cat([kx, eye], 1); V_ = torch.cat([vx, torch.zeros(B, d, 1, 1, dtype=torch.float64)], 1)
Q_ = torch.cat([kx, eye], 1)
beta = torch.cat([torch.full((B, n, 1), 0.7, dtype=torch.float64), torch.zeros(B, d, 1, dtype=torch.float64)], 1)
o = delta_ref(Q_, K_, V_, beta)[:, n:, 0, 0]
W_lms = lms_prefix_w(X, y, beta=0.7)
check("DeltaNet state (k=x/|x|, v=y/|x|, beta=0.7) == normalized-LMS weights", o, W_lms[:, -1], 1e-12)
print("all seqmodel checks passed")
