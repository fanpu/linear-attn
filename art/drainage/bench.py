import time, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False
B, nkv, nh, hd, T, t = 1024, 8, 16, 128, 513, 256
K = torch.randn(B, nkv, T, hd, device="cuda", dtype=torch.bfloat16)
V = torch.randn_like(K)
q = torch.randn(B, nh, 1, hd, device="cuda", dtype=torch.bfloat16)
def sdpa():
    return F.scaled_dot_product_attention(q, K[:, :, :t], V[:, :, :t], enable_gqa=True)
def manual():
    qq = q.view(B, nkv, nh // nkv, hd)
    s = torch.matmul(qq, K[:, :, :t].transpose(-1, -2)).float() * hd ** -0.5
    a = torch.softmax(s, -1).to(q.dtype)
    return torch.matmul(a, V[:, :, :t]).view(B, nh, 1, hd)
def manual32():
    qq = q.view(B, nkv, nh // nkv, hd)
    s = torch.matmul(qq, K[:, :, :t].transpose(-1, -2)).float() * hd ** -0.5
    a = torch.softmax(s, -1)
    return torch.matmul(a, V[:, :, :t].float()).to(q.dtype).view(B, nh, 1, hd)
for f in (sdpa, manual, manual32, sdpa, manual):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(28): o = f()
    torch.cuda.synchronize(); print(f.__name__, f"{(time.time()-t0)*1000:.1f} ms per 28 layers")
print("max diff manual vs sdpa", (sdpa().float() - manual().float()).abs().max().item(), (sdpa().float() - manual32().float()).abs().max().item())
gb = 28 * 2 * B * nkv * t * hd * 2 / 1e9
print(f"KV bytes read per step {gb:.1f} GB")
