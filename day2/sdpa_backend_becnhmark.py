import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

for H in (8, 12, 16):                       # your three head counts
    q = torch.randn(32, H, 1024, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        with sdpa_kernel([b]):
            for _ in range(5):              # warmup
                F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(50):
                F.scaled_dot_product_attention(q, q, q, is_causal=True).sum().backward()
            e.record(); torch.cuda.synchronize()
            print(f"H={H:2d} {b.name:20s} {s.elapsed_time(e)/50:7.2f} ms")
